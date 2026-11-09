"""End-to-end wiring test with a fake backend (no network, no GPU)."""
from PIL import Image

import image_to_3d.pipeline as pipeline
from image_to_3d import convert_image_to_3d
from image_to_3d.providers import Provider


class FakeSession:
    @classmethod
    def connect(cls, space_id, token, work_dir, ctx):
        return cls()

    def close(self):
        pass


class FakeProvider(Provider):
    key, label = "trellis", "fake"

    def generate(self, image_path, params):
        out = image_path.parent / "model.glb"
        out.write_bytes(b"glTF" + b"0" * 200)
        return out


def test_convert_with_fake_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SpaceSession", FakeSession)
    monkeypatch.setattr(pipeline, "resolve_chain", lambda p, s: [(FakeProvider, "fake/space")])
    img = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "blue").save(img)
    logs = []
    res = convert_image_to_3d(img, remove_bg=False, seed=1, output_dir=tmp_path / "out", on_status=logs.append)
    assert res.preview.exists() and res.preview.suffix == ".glb"
    assert res.space_id == "fake/space" and res.seed == 1
    assert any("Connected" in m for m in logs)
