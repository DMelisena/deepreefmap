"""No-auth browser prototype for turning reef survey videos into 3D point clouds."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_JOB_ID_RE = re.compile(r"[a-f0-9]{12}")


class UploadValidationError(ValueError):
    """Raised when an uploaded file cannot safely enter reconstruction."""


@dataclass
class ReconstructionJob:
    id: str
    filename: str
    input_path: Path
    output_dir: Path
    status: str = "queued"
    error: str | None = None
    created: float = 0.0

    def public(self) -> dict[str, object]:
        done = self.status == "complete"
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "created": self.created,
            "video_url": f"/api/jobs/{self.id}/video" if self.input_path.is_file() else None,
            "model_url": f"/api/jobs/{self.id}/model.ply" if done else None,
            "preview_url": f"/api/jobs/{self.id}/preview.ply" if done else None,
            "download_url": f"/api/jobs/{self.id}/download" if done else None,
        }


def validate_upload(path: Path, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> Path:
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise UploadValidationError(f"Choose a supported video ({supported}).")
    if not path.is_file() or path.stat().st_size == 0:
        raise UploadValidationError("Uploaded video is empty.")
    if path.stat().st_size > max_bytes:
        raise UploadValidationError(f"Video exceeds {max_bytes // (1024 * 1024)} MB prototype limit.")
    return path


def build_reconstruction_command(video_path: Path, output_dir: Path, camera_profile: str) -> list[str]:
    """Return safe no-auth prototype command. Geometry only avoids gated segmentation."""
    return [
        "uv",
        "run",
        "deepreefmap",
        "reconstruct",
        "--videos",
        str(video_path),
        "--camera-profile",
        camera_profile,
        "--mapping",
        "scsfmlearner",
        "--skip-segmentation",
        "--out",
        str(output_dir),
    ]


PREVIEW_TARGET_POINTS = 200_000
_PREVIEW_LOCKS: dict[str, threading.Lock] = {}
_PREVIEW_LOCKS_GUARD = threading.Lock()


def build_preview_ply(model_path: Path, preview_path: Path, target: int = PREVIEW_TARGET_POINTS) -> Path | None:
    """Downsample a large PLY to a small binary preview so the browser loads fast.

    Returns the preview path when ready. Falls back to None (caller serves full PLY)
    if open3d is unavailable or the cloud is already small.
    """
    if preview_path.is_file() and preview_path.stat().st_mtime >= model_path.stat().st_mtime:
        return preview_path
    try:
        import open3d as o3d
    except ImportError:
        return None
    cloud = o3d.io.read_point_cloud(str(model_path))
    n = len(cloud.points)
    if n == 0:
        return None
    if n > target:
        cloud = cloud.uniform_down_sample(every_k_points=max(1, n // target))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = preview_path.with_suffix(".tmp.ply")
    o3d.io.write_point_cloud(str(tmp), cloud, write_ascii=False)
    tmp.replace(preview_path)
    return preview_path


def _preview_lock(job_id: str) -> threading.Lock:
    with _PREVIEW_LOCKS_GUARD:
        return _PREVIEW_LOCKS.setdefault(job_id, threading.Lock())



class JobStore:
    def __init__(self, root: Path, camera_profile: str) -> None:
        self.root = root
        self.camera_profile = camera_profile
        self.jobs: dict[str, ReconstructionJob] = {}
        self.lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        """Rehydrate jobs already on disk so a restart keeps prior uploads visible."""
        for job_root in self.root.iterdir():
            if not job_root.is_dir() or not _JOB_ID_RE.fullmatch(job_root.name):
                continue
            inputs = [p for p in (job_root / "input").glob("*") if p.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES]
            if not inputs:
                continue
            input_path = inputs[0]
            output_dir = job_root / "result"
            complete = (output_dir / "geometry_cloud.ply").is_file()
            self.jobs[job_root.name] = ReconstructionJob(
                job_root.name,
                input_path.name,
                input_path,
                output_dir,
                status="complete" if complete else "interrupted",
                error=None if complete else "Reconstruction did not finish before the last server restart.",
                created=job_root.stat().st_mtime,
            )

    def create(self, filename: str, content: bytes) -> ReconstructionJob:
        safe_name = _safe_filename(filename)
        job_id = uuid.uuid4().hex[:12]
        job_root = self.root / job_id
        input_path = job_root / "input" / safe_name
        input_path.parent.mkdir(parents=True)
        input_path.write_bytes(content)
        validate_upload(input_path)
        job = ReconstructionJob(
            job_id, safe_name, input_path, job_root / "result", created=job_root.stat().st_mtime
        )
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> ReconstructionJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[ReconstructionJob]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created, reverse=True)

    def _run(self, job: ReconstructionJob) -> None:
        job.status = "processing"
        command = build_reconstruction_command(job.input_path, job.output_dir, self.camera_profile)
        try:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=60 * 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = "Reconstruction exceeded 60 minute prototype limit."
            return
        except OSError as exc:
            job.status = "failed"
            job.error = str(exc)
            return
        model_path = job.output_dir / "geometry_cloud.ply"
        if completed.returncode == 0 and model_path.is_file():
            job.status = "complete"
            return
        job.status = "failed"
        output = (completed.stderr or completed.stdout or "Reconstruction failed.").strip()
        job.error = output[-500:]


def _safe_filename(filename: str) -> str:
    base = Path(filename).name
    safe = _FILENAME_RE.sub("-", base).strip(".-")
    return safe or "reef-video.mp4"


class ReefModelHandler(BaseHTTPRequestHandler):
    store: ClassVar[JobStore]
    server_version = "DeepReefMapPrototype/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", HTML.encode())
            return
        if parsed.path == "/api/jobs":
            self._json(HTTPStatus.OK, {"jobs": [job.public() for job in self.store.list()]})
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", parsed.path)
        if match:
            job = self.store.get(match.group(1))
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Job not found."})
            else:
                self._json(HTTPStatus.OK, job.public())
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/video", parsed.path)
        if match:
            job = self.store.get(match.group(1))
            if job is None or not job.input_path.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "Video not found."})
                return
            content_type = _VIDEO_CONTENT_TYPES.get(job.input_path.suffix.lower(), "application/octet-stream")
            self._send_file(job.input_path, "inline", content_type)
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/preview\.ply", parsed.path)
        if match:
            job = self.store.get(match.group(1))
            model = job.output_dir / "geometry_cloud.ply" if job else None
            if job is None or job.status != "complete" or model is None or not model.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "Model not ready."})
                return
            preview = job.output_dir / "geometry_preview.ply"
            with _preview_lock(job.id):
                try:
                    ready = build_preview_ply(model, preview)
                except Exception:  # noqa: BLE001 - preview is best-effort; fall back to full PLY
                    ready = None
            self._send_file(ready if ready is not None else model, "inline")
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/(model\.ply|download)", parsed.path)
        if match:
            job = self.store.get(match.group(1))
            model = job.output_dir / "geometry_cloud.ply" if job else None
            if job is None or job.status != "complete" or model is None or not model.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "Model not ready."})
                return
            disposition = "attachment" if match.group(2) == "download" else "inline"
            self._send_file(model, disposition)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/jobs":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > DEFAULT_MAX_UPLOAD_BYTES + 1024 * 1024:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Upload is empty or exceeds 2 GB limit."})
            return
        content_type = self.headers.get("Content-Type", "")
        boundary_match = re.search(r"boundary=([^;]+)", content_type)
        if not boundary_match:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Expected multipart video upload."})
            return
        body = self.rfile.read(length)
        try:
            filename, content = _parse_multipart_file(body, boundary_match.group(1).strip('"').encode())
            job = self.store.create(filename, content)
        except UploadValidationError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.ACCEPTED, job.public())

    def log_message(self, format: str, *args: object) -> None:
        print(f"[webapp] {self.address_string()} {format % args}")

    def _json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send(self, status: HTTPStatus, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_file(self, path: Path, disposition: str, content_type: str = "application/octet-stream") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as file:
                shutil.copyfileobj(file, self.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _parse_multipart_file(body: bytes, boundary: bytes) -> tuple[str, bytes]:
    marker = b"--" + boundary
    for part in body.split(marker):
        if b"Content-Disposition:" not in part or b"filename=" not in part:
            continue
        try:
            headers, content = part.split(b"\r\n\r\n", 1)
        except ValueError as exc:
            raise ValueError("Malformed upload body.") from exc
        name_match = re.search(br'filename="([^\"]+)"', headers)
        if name_match is None:
            continue
        if not content.endswith(b"\r\n"):
            raise ValueError("Malformed upload body.")
        return name_match.group(1).decode("utf-8", "replace"), content[:-2]
    raise ValueError("Video field missing from upload.")


def serve(host: str, port: int, jobs_dir: Path, camera_profile: str) -> None:
    ReefModelHandler.store = JobStore(jobs_dir, camera_profile)
    server = ThreadingHTTPServer((host, port), ReefModelHandler)
    print(f"DeepReefMap prototype listening on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4190)
    parser.add_argument("--jobs-dir", type=Path, default=Path("webapp_jobs"))
    parser.add_argument("--camera-profile", default="gopro_hero_10")
    args = parser.parse_args()
    serve(args.host, args.port, args.jobs_dir, args.camera_profile)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeaLens Reef Model Lab</title><style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Manrope:wght@400;500;600;700&display=swap');
:root{--ink:#082f31;--water:#0a5b63;--foam:#eaf5ef;--sand:#f3eee2;--coral:#ff7558;--line:#bdd9d2}*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:Manrope,sans-serif;background:radial-gradient(circle at 95% 4%,#9cded2 0,transparent 29%),var(--foam)}.shell{max-width:1180px;margin:auto;padding:25px 26px 42px}.top{display:flex;align-items:center;justify-content:space-between;padding-bottom:48px}.brand{font:650 25px Fraunces,serif;letter-spacing:-1px}.brand b{color:var(--coral)}.tag{font:500 11px 'DM Mono',monospace;border:1px solid var(--line);padding:8px 10px;border-radius:99px;background:#ffffff8a}.hero{display:grid;grid-template-columns:1.08fr .92fr;gap:32px;align-items:stretch}.intro{padding:18px 0}.kicker{font:500 11px 'DM Mono',monospace;letter-spacing:.1em;color:var(--water);text-transform:uppercase}.intro h1{font:650 clamp(46px,6vw,78px)/.93 Fraunces,serif;letter-spacing:-.065em;margin:18px 0 23px}.intro p{max-width:535px;font-size:17px;line-height:1.65;margin:0;color:#31575a}.facts{display:flex;gap:30px;margin-top:42px}.facts span{display:block;font:500 11px 'DM Mono',monospace;color:#4e7071;text-transform:uppercase}.facts strong{font:650 22px Fraunces,serif}.card{background:#fff;border:1px solid var(--line);box-shadow:8px 10px 0 #c9e8de;border-radius:22px;padding:22px}.viewer{min-height:360px;background:linear-gradient(145deg,#062c35,#0a6972 60%,#45a9a0);position:relative;overflow:hidden;padding:0}.viewer:before,.viewer:after{content:"";position:absolute;border:1px solid #91e8d73d;border-radius:50%;inset:16%;transform:rotate(25deg)}.viewer:after{inset:27%;transform:rotate(-48deg)}canvas{position:absolute;inset:0;width:100%;height:100%}.viewer .label{position:absolute;z-index:2;left:18px;top:17px;color:#ddfff6;font:500 10px 'DM Mono',monospace;letter-spacing:.12em}.viewer .hint{position:absolute;z-index:2;bottom:16px;left:18px;color:#ddfff6c4;font-size:12px}.upload{margin-top:32px;display:grid;grid-template-columns:1.35fr .65fr;gap:18px;align-items:stretch}.drop{border:1.5px dashed #5ba79f;background:#ffffff8c;border-radius:18px;padding:26px;cursor:pointer;transition:.2s}.drop:hover,.drop.drag{background:#fff;border-color:var(--water);transform:translateY(-2px)}.drop input{display:none}.drop strong{font:650 23px Fraunces,serif;display:block;margin:8px 0}.drop p{margin:0;color:#527476;font-size:13px}.drop i{font-style:normal;color:var(--coral)}button{width:100%;border:0;border-radius:18px;background:var(--coral);color:#fff;font:700 14px Manrope,sans-serif;cursor:pointer;padding:20px;transition:.2s}button:hover:not(:disabled){background:#e55b41;transform:translateY(-2px)}button:disabled{opacity:.48;cursor:not-allowed}.status{min-height:75px;margin-top:20px;border-left:3px solid var(--water);padding:12px 15px;background:#ffffff88;font-size:13px;line-height:1.55}.status a{color:var(--water);font-weight:700}.process{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:46px}.step{padding:16px 4px;border-top:1px solid var(--line);font-size:13px}.step span{font:500 10px 'DM Mono',monospace;color:var(--coral)}.step b{display:block;margin-top:7px}
.localply{display:block;margin-top:14px;padding:13px 16px;border:1.5px dashed #5ba79f;border-radius:14px;background:#ffffff8c;cursor:pointer;font-size:13px;color:#3a5f61;transition:.2s;text-align:center}.localply:hover{background:#fff;border-color:var(--water)}.localply b{color:var(--water)}
.gallery{margin-top:54px}.gallery h2{font:650 30px Fraunces,serif;letter-spacing:-.03em;margin:0 0 4px}.gallery .sub{color:#4e7071;font-size:13px;margin:0 0 20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px}.jobcard{background:#fff;border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:5px 6px 0 #d3ece3;display:flex;flex-direction:column}.jobcard video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#062c35;display:block}.jobcard .novid{width:100%;aspect-ratio:16/9;background:linear-gradient(145deg,#062c35,#0a6972);display:flex;align-items:center;justify-content:center;color:#ddfff6a0;font:500 11px 'DM Mono',monospace}.jobbody{padding:13px 14px 15px;display:flex;flex-direction:column;gap:9px;flex:1}.jobname{font-weight:700;font-size:14px;word-break:break-all}.badge{align-self:flex-start;font:500 10px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.08em;padding:4px 9px;border-radius:99px}.b-complete{background:#d4f3e4;color:#0a6b4b}.b-processing,.b-queued{background:#fdeecf;color:#8a5a13}.b-failed,.b-interrupted{background:#fadcd5;color:#9a3320}.jobactions{margin-top:auto;display:flex;gap:8px}.jobactions button{padding:10px;font-size:12px;border-radius:11px}.jobactions .ghost{background:#eef7f2;color:var(--water);border:1px solid var(--line)}.jobactions .ghost:hover{background:#e0f0e8}.joberr{font-size:11px;color:#9a3320;line-height:1.4}.empty{color:#4e7071;font-size:14px;padding:22px;border:1px dashed var(--line);border-radius:14px;background:#ffffff70}
@media(max-width:750px){.hero,.upload{grid-template-columns:1fr}.top{padding-bottom:28px}.facts{margin-top:28px}.process{grid-template-columns:1fr 1fr}.viewer{min-height:300px}}
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body><main class="shell"><header class="top"><div class="brand">Sea<b>Lens</b> × DeepReefMap</div><div class="tag">NO AUTH · PROTOTYPE</div></header><section class="hero"><div class="intro"><div class="kicker">Reef model laboratory</div><h1>Turn a dive into terrain.</h1><p>Upload reef footage. DeepReefMap samples frames, estimates camera motion and depth, then returns an explorable 3D point cloud. Built for GoPro Hero 10 Linear footage in this prototype.</p><div class="facts"><div><strong>PLY</strong><span>export format</span></div><div><strong>3D</strong><span>point cloud</span></div><div><strong>60 min</strong><span>job ceiling</span></div></div></div><div class="card viewer"><div class="label" id="viewerLabel">PROCEDURAL REEF PREVIEW</div><canvas id="reef"></canvas><div class="hint">Drag to orbit · scroll to zoom · right-drag to pan</div></div></section><section class="upload"><label class="drop" id="drop"><input id="file" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/x-m4v"><span class="kicker">01 / Select footage</span><strong id="fileName">Drop reef video here</strong><p>MP4, MOV, AVI, MKV or M4V · max 2 GB · <i>video stays on this prototype host</i></p></label><button id="start" disabled>Build 3D model<br><small>Geometry-only reconstruction</small></button></section><label class="localply" id="localplyLabel"><input id="localply" type="file" accept=".ply" style="display:none"><span>Already have a .ply? <b>View it locally — no upload</b></span></label><div class="status" id="status">Choose a reef video. Output becomes a downloadable and browser-viewable PLY point cloud.</div><section class="process"><div class="step"><span>01</span><b>Rectify frames</b>Camera profile corrects lens geometry.</div><div class="step"><span>02</span><b>Estimate depth</b>SC-SfMLearner maps scene geometry.</div><div class="step"><span>03</span><b>Fuse cloud</b>Frames become one colored reef surface.</div><div class="step"><span>04</span><b>Inspect / export</b>Rotate result here or download PLY.</div></section><section class="gallery"><h2>Reef library</h2><p class="sub">Every uploaded dive and its reconstruction. Reconstructions survive server restarts.</p><div class="grid" id="gallery"></div></section></main><script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';
const canvas=document.querySelector('#reef'),file=document.querySelector('#file'),start=document.querySelector('#start'),drop=document.querySelector('#drop'),status=document.querySelector('#status'),name=document.querySelector('#fileName'),label=document.querySelector('#viewerLabel'),gallery=document.querySelector('#gallery');
let chosen;
const MAX_POINTS=600000;
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(55,1,0.01,5000);
camera.position.set(0,0,6);
const controls=new OrbitControls(camera,canvas);
controls.enableDamping=true;controls.dampingFactor=0.08;
function fit(){const w=canvas.clientWidth,h=canvas.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
addEventListener('resize',fit);fit();
let cloud;
function decimate(geometry){
  const pos=geometry.getAttribute('position');if(!pos)return geometry;
  const n=pos.count;if(n<=MAX_POINTS)return geometry;
  const step=Math.ceil(n/MAX_POINTS),keep=Math.floor(n/step);
  const col=geometry.getAttribute('color');
  const np=new Float32Array(keep*3),nc=col?new Float32Array(keep*3):null;
  for(let i=0,j=0;j<keep;i+=step,j++){
    np[j*3]=pos.getX(i);np[j*3+1]=pos.getY(i);np[j*3+2]=pos.getZ(i);
    if(nc){nc[j*3]=col.getX(i);nc[j*3+1]=col.getY(i);nc[j*3+2]=col.getZ(i)}
  }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(np,3));
  if(nc)g.setAttribute('color',new THREE.BufferAttribute(nc,3));
  return g;
}
function setCloud(geometry,hasColor){
  if(cloud){scene.remove(cloud);cloud.geometry.dispose();cloud.material.dispose()}
  geometry.computeBoundingSphere();
  const s=geometry.boundingSphere;
  geometry.translate(-s.center.x,-s.center.y,-s.center.z);
  const mat=new THREE.PointsMaterial({size:s.radius/260,sizeAttenuation:true});
  if(hasColor)mat.vertexColors=true;else mat.color=new THREE.Color('#7fd8bd');
  cloud=new THREE.Points(geometry,mat);scene.add(cloud);
  controls.target.set(0,0,0);
  camera.position.set(0,0,s.radius*2.4);controls.update();
}
function placeholder(){
  const n=1400,pos=new Float32Array(n*3),col=new Float32Array(n*3),c1=new THREE.Color('#82d8b7'),c2=new THREE.Color('#ff9a77');
  for(let i=0;i<n;i++){const a=i*2.399,r=1+Math.sqrt(i)*0.12,co=(i%9?c1:c2);
    pos[i*3]=Math.cos(a)*r;pos[i*3+1]=(Math.random()-.5)*7;pos[i*3+2]=Math.sin(a)*r;
    col[i*3]=co.r;col[i*3+1]=co.g;col[i*3+2]=co.b}
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  setCloud(g,true);
}
placeholder();
(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera)})();
function loadPly(url,displayName){
  label.textContent='LOADING MODEL…';
  new PLYLoader().load(url,raw=>{
    const total=raw.getAttribute('position')?.count||0;
    const g=decimate(raw);
    const hasColor=!!g.getAttribute('color');
    setCloud(g,hasColor);
    const shown=g.getAttribute('position').count;
    const suffix=shown<total?` (of ${total.toLocaleString()})`:'';
    label.textContent=`${(displayName||'MODEL').toUpperCase()} · ${shown.toLocaleString()} PTS${suffix}`;
    canvas.scrollIntoView({behavior:'smooth',block:'center'});
  },undefined,()=>{status.textContent='Model loaded but viewer could not parse the PLY.'});
}
function loadLocalPly(f){
  label.textContent='LOADING LOCAL FILE…';status.textContent=`Reading ${f.name} (${(f.size/1048576).toFixed(1)} MB) — nothing leaves your browser.`;
  const rd=new FileReader();
  rd.onload=()=>{try{
    const raw=new PLYLoader().parse(rd.result);
    const total=raw.getAttribute('position')?.count||0;
    const g=decimate(raw);setCloud(g,!!g.getAttribute('color'));
    const shown=g.getAttribute('position').count,suffix=shown<total?` (of ${total.toLocaleString()})`:'';
    label.textContent=`${f.name.toUpperCase()} · ${shown.toLocaleString()} PTS${suffix}`;
    status.textContent=`Viewing local file ${f.name}. Drag to orbit.`;
    canvas.scrollIntoView({behavior:'smooth',block:'center'});
  }catch(_){status.textContent='Could not parse that .ply file.'}};
  rd.onerror=()=>{status.textContent='Could not read that file.'};
  rd.readAsArrayBuffer(f);
}
const localply=document.querySelector('#localply');
localply.addEventListener('change',()=>localply.files[0]&&loadLocalPly(localply.files[0]));
function choose(f){chosen=f;name.textContent=f.name;start.disabled=false;status.textContent=`Ready: ${f.name} (${(f.size/1048576).toFixed(1)} MB).`}
file.addEventListener('change',()=>file.files[0]&&choose(file.files[0]));
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,v=>{v.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>e.dataTransfer.files[0]&&choose(e.dataTransfer.files[0]));
const BADGE={complete:'b-complete',processing:'b-processing',queued:'b-queued',failed:'b-failed',interrupted:'b-interrupted'};
function card(job){
  const el=document.createElement('div');el.className='jobcard';
  const media=job.video_url?`<video src="${job.video_url}" muted loop playsinline preload="metadata" onmouseover="this.play()" onmouseout="this.pause()"></video>`:'<div class="novid">NO VIDEO</div>';
  const cls=BADGE[job.status]||'b-queued';
  const err=job.error?`<div class="joberr">${job.error.replace(/[<>&]/g,'')}</div>`:'';
  let actions='';
  if(job.status==='complete'){actions=`<button data-view="${job.preview_url||job.model_url}" data-name="${job.filename}">View 3D</button><a href="${job.download_url}"><button class="ghost">PLY</button></a>`}
  else if(job.status==='processing'||job.status==='queued'){actions='<button class="ghost" disabled>Reconstructing…</button>'}
  else{actions='<button class="ghost" disabled>No model</button>'}
  el.innerHTML=`${media}<div class="jobbody"><div class="jobname">${job.filename}</div><span class="badge ${cls}">${job.status}</span>${err}<div class="jobactions">${actions}</div></div>`;
  const view=el.querySelector('button[data-view]');
  if(view)view.onclick=()=>loadPly(view.dataset.view,view.dataset.name);
  return el;
}
async function refresh(){
  try{
    const {jobs}=await (await fetch('/api/jobs')).json();
    gallery.innerHTML='';
    if(!jobs.length){gallery.innerHTML='<div class="empty">No dives yet. Upload a reef video above to build the first model.</div>';return}
    for(const job of jobs)gallery.appendChild(card(job));
  }catch(_){/* leave existing gallery on transient error */}
}
refresh();setInterval(refresh,5000);
start.onclick=async()=>{start.disabled=true;status.textContent='Uploading footage…';const body=new FormData();body.append('video',chosen);const r=await fetch('/api/jobs',{method:'POST',body}),j=await r.json();if(!r.ok){status.textContent=j.error;start.disabled=false;return}status.textContent='Queued. DeepReefMap is reconstructing geometry…';refresh();const poll=setInterval(async()=>{const q=await (await fetch('/api/jobs/'+j.id)).json();if(q.status==='complete'){clearInterval(poll);status.innerHTML=`Model complete. <a href="${q.model_url}">View PLY data</a> · <a href="${q.download_url}">Download model</a>`;loadPly(q.model_url,q.filename);refresh();start.disabled=false;start.textContent='Build another model'}else if(q.status==='failed'){clearInterval(poll);status.textContent='Failed: '+q.error;refresh();start.disabled=false}else{status.textContent='Reconstructing geometry… This can take several minutes.';refresh()}},2500)};
</script></body></html>'''


if __name__ == "__main__":
    main()
