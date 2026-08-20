from pathlib import Path

import pytest

from deepreefmap.webapp import (
    JobStore,
    UploadValidationError,
    build_preview_ply,
    build_reconstruction_command,
    validate_upload,
)


def _seed_job(root: Path, job_id: str, video: str, with_model: bool) -> None:
    (root / job_id / "input").mkdir(parents=True)
    (root / job_id / "input" / video).write_bytes(b"reefdata")
    if with_model:
        (root / job_id / "result").mkdir(parents=True)
        (root / job_id / "result" / "geometry_cloud.ply").write_bytes(b"ply")


def test_jobstore_restores_disk_jobs_with_status(tmp_path: Path) -> None:
    _seed_job(tmp_path, "aaaaaaaaaaaa", "DJI252.mp4", with_model=True)
    _seed_job(tmp_path, "bbbbbbbbbbbb", "DJI231.mp4", with_model=False)

    store = JobStore(tmp_path, camera_profile="gopro_hero_10")
    jobs = {job.id: job for job in store.list()}

    assert jobs["aaaaaaaaaaaa"].status == "complete"
    assert jobs["aaaaaaaaaaaa"].public()["model_url"] == "/api/jobs/aaaaaaaaaaaa/model.ply"
    assert jobs["bbbbbbbbbbbb"].status == "interrupted"
    assert jobs["bbbbbbbbbbbb"].public()["model_url"] is None
    assert jobs["bbbbbbbbbbbb"].public()["video_url"] == "/api/jobs/bbbbbbbbbbbb/video"


def test_jobstore_ignores_non_job_dirs(tmp_path: Path) -> None:
    (tmp_path / "not-a-job").mkdir()
    (tmp_path / "zzzzzzzzzzzz" / "input").mkdir(parents=True)
    (tmp_path / "zzzzzzzzzzzz" / "input" / "notes.txt").write_bytes(b"x")

    store = JobStore(tmp_path, camera_profile="gopro_hero_10")

    assert store.list() == []


def test_build_preview_ply_downsamples_and_caches(tmp_path: Path) -> None:
    o3d = pytest.importorskip("open3d")
    import numpy as np

    pts = np.random.default_rng(0).random((5000, 3)).astype("float32")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    model = tmp_path / "geometry_cloud.ply"
    o3d.io.write_point_cloud(str(model), cloud, write_ascii=False)

    preview = tmp_path / "geometry_preview.ply"
    result = build_preview_ply(model, preview, target=1000)

    assert result == preview
    assert preview.is_file()
    downsampled = o3d.io.read_point_cloud(str(preview))
    assert 0 < len(downsampled.points) <= 5000
    # Second call reuses the cached preview (mtime not older than the model).
    first_mtime = preview.stat().st_mtime
    assert build_preview_ply(model, preview, target=1000) == preview
    assert preview.stat().st_mtime == first_mtime


def test_validate_upload_accepts_supported_video_within_limit(tmp_path: Path) -> None:
    clip = tmp_path / "reef-survey.mp4"
    clip.write_bytes(b"reef")

    assert validate_upload(clip, max_bytes=4) == clip


def test_validate_upload_rejects_non_video_and_oversized_uploads(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_bytes(b"reef")
    too_large = tmp_path / "survey.mov"
    too_large.write_bytes(b"12345")

    with pytest.raises(UploadValidationError, match="supported video"):
        validate_upload(document, max_bytes=10)
    with pytest.raises(UploadValidationError, match="exceeds"):
        validate_upload(too_large, max_bytes=4)


def test_build_reconstruction_command_uses_geometry_only_safe_defaults(tmp_path: Path) -> None:
    command = build_reconstruction_command(
        video_path=tmp_path / "reef.mp4",
        output_dir=tmp_path / "run",
        camera_profile="gopro_hero_10",
    )

    assert command[:3] == ["uv", "run", "deepreefmap"]
    assert "reconstruct" in command
    assert "--skip-segmentation" in command
    assert command[command.index("--mapping") + 1] == "scsfmlearner"
    assert command[command.index("--camera-profile") + 1] == "gopro_hero_10"
    assert command[command.index("--out") + 1] == str(tmp_path / "run")
