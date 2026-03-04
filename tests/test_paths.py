from pathlib import Path

from core.pipeline import collect_images


def test_collect_images_single_file(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    out = collect_images(img)
    assert out == [img]


def test_collect_images_folder(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("ignore")
    (tmp_path / "c.png").write_bytes(b"x")
    out = collect_images(tmp_path)
    assert [p.name for p in out] == ["a.jpg", "c.png"]
