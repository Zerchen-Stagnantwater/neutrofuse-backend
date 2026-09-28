import numpy as np

from neutrofuse.core.patches import (
    pad_to_multiple,
    unpad,
    extract_patches,
    reconstruct_from_patches,
    grid_shape_for,
)


def test_pad_to_multiple_no_op_when_already_aligned():
    img = np.zeros((16, 16), dtype=np.uint8)
    padded, orig_shape = pad_to_multiple(img, 8)
    assert padded.shape == (16, 16)
    assert orig_shape == (16, 16)


def test_pad_to_multiple_pads_correctly():
    img = np.zeros((10, 13), dtype=np.uint8)
    padded, orig_shape = pad_to_multiple(img, 8)
    assert padded.shape == (16, 16)
    assert orig_shape == (10, 13)


def test_unpad_recovers_original_content():
    img = np.random.randint(0, 256, (10, 13), dtype=np.uint8)
    padded, orig_shape = pad_to_multiple(img, 8)
    recovered = unpad(padded, orig_shape)
    assert recovered.shape == orig_shape
    np.testing.assert_array_equal(recovered, img)


def test_extract_reconstruct_roundtrip_grayscale():
    img = np.random.randint(0, 256, (32, 24), dtype=np.uint8)
    patches = extract_patches(img, 8)
    assert patches.shape == (4, 3, 8, 8)
    reconstructed = reconstruct_from_patches(patches)
    np.testing.assert_array_equal(reconstructed, img)


def test_extract_reconstruct_roundtrip_color():
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    patches = extract_patches(img, 4)
    assert patches.shape == (4, 4, 4, 4, 3)
    reconstructed = reconstruct_from_patches(patches)
    np.testing.assert_array_equal(reconstructed, img)


def test_patch_spatial_correspondence():
    """Patch (i, j) must correspond to image[i*ps:(i+1)*ps, j*ps:(j+1)*ps]."""
    img = np.arange(64).reshape(8, 8).astype(np.uint8)
    patches = extract_patches(img, 4)
    np.testing.assert_array_equal(patches[0, 1], img[0:4, 4:8])
    np.testing.assert_array_equal(patches[1, 0], img[4:8, 0:4])


def test_grid_shape_for():
    img = np.zeros((32, 24), dtype=np.uint8)
    assert grid_shape_for(img, 8) == (4, 3)


def test_extract_raises_on_unaligned_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    try:
        extract_patches(img, 8)
        assert False, "expected ValueError"
    except ValueError:
        pass
