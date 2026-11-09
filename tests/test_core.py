import pytest
from PIL import Image

from image_to_3d.config import AUTO_ORDER
from image_to_3d.errors import (AuthError, BackendAPIChangedError, BackendUnavailableError, InputError,
                                QuotaError)
from image_to_3d.export.mesh import validate_mesh_file
from image_to_3d.errors import BackendError
from image_to_3d.preprocess.image_io import prepare_image, validate_image_path
from image_to_3d.providers import resolve_chain
from image_to_3d.remote.classify import classify_exception
from image_to_3d.remote.files import find_files
from image_to_3d.utils import env_token


def test_classify_quota():
    assert isinstance(classify_exception(RuntimeError("You have exceeded your GPU quota")), QuotaError)


def test_classify_api_changed():
    assert isinstance(classify_exception(RuntimeError("Cannot find a function with api_name")), BackendAPIChangedError)


def test_classify_unavailable():
    assert isinstance(classify_exception(RuntimeError("Space is sleeping")), BackendUnavailableError)


def test_classify_auth():
    assert isinstance(classify_exception(RuntimeError("Invalid user token")), AuthError)


def test_resolve_chain_auto_order():
    keys = [cls.key for cls, _ in resolve_chain("auto", None)]
    assert keys[0] == AUTO_ORDER[0]


def test_resolve_chain_rejects_unknown_and_auto_with_space():
    with pytest.raises(InputError):
        resolve_chain("nope", None)
    with pytest.raises(InputError):
        resolve_chain("auto", "a/b")


def test_validate_image_path(tmp_path):
    with pytest.raises(InputError):
        validate_image_path(tmp_path / "missing.png")
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    with pytest.raises(InputError):
        validate_image_path(bad)


def test_prepare_image_downscales(tmp_path):
    src = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 1500), "red").save(src)
    out, has_alpha = prepare_image(validate_image_path(src), tmp_path)
    assert max(Image.open(out).size) == 1024 and not has_alpha


def test_find_files(tmp_path):
    f = tmp_path / "m.glb"
    f.write_bytes(b"glTF" + b"0" * 100)
    assert find_files({"a": [str(f), "not a path"]}, (".glb",)) == [f]


def test_validate_mesh_file(tmp_path):
    bad = tmp_path / "m.glb"
    bad.write_bytes(b"<html>" + b"0" * 100)
    with pytest.raises(BackendError):
        validate_mesh_file(bad)
    good = tmp_path / "g.glb"
    good.write_bytes(b"glTF" + b"0" * 100)
    assert validate_mesh_file(good) == good


def test_no_hardcoded_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert env_token() is None
