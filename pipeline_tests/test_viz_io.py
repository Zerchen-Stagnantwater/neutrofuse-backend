import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neutrofuse.viz.io import save_figure


def test_save_figure_writes_real_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    path = tmp_path / "out.png"
    result = save_figure(fig, path)

    assert result == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_figure_creates_parent_directories(tmp_path):
    fig, ax = plt.subplots()
    path = tmp_path / "nested" / "dirs" / "out.png"
    save_figure(fig, path)
    assert path.exists()


def test_save_figure_closes_by_default(tmp_path):
    fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())

    save_figure(fig, tmp_path / "out.png")

    n_after = len(plt.get_fignums())
    assert n_after == n_before - 1


def test_save_figure_close_false_keeps_figure_open(tmp_path):
    fig, ax = plt.subplots()
    n_before = len(plt.get_fignums())

    save_figure(fig, tmp_path / "out.png", close=False)

    n_after = len(plt.get_fignums())
    assert n_after == n_before  # unchanged, figure still open
    plt.close(fig)  # manual cleanup since we opted out of auto-close
