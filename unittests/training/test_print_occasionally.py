import pytest

from lipschitz.training.trainer import PrintProgressOccasionally


def test_len_is_copied_from_iterator():

    class DummyIterable:
        def __len__(self):
            return 42

        def __iter__(self):
            return iter(range(10))

    iterable = DummyIterable()
    progress = PrintProgressOccasionally(iterable)

    assert len(progress) == 42, "Length should be copied from the iterable."


def test_len_raise_error_if_iterator_has_no_len():

    class DummyIterableWithoutLen:
        def __iter__(self):
            return iter(range(10))

    iterable = DummyIterableWithoutLen()
    progress = PrintProgressOccasionally(iterable)

    with pytest.raises(TypeError):
        len(progress)


def test_safe_len_str(capfd):
    class DummyIterableWithoutLen:
        def __iter__(self):
            return iter(range(5))

    iterable = DummyIterableWithoutLen()
    progress = PrintProgressOccasionally(iterable)

    for _ in progress:
        pass

    std_out, err = capfd.readouterr()

    assert "Started iterating." in std_out
    assert "Completed 1 / ? steps" in std_out
    assert "Completed 2 / ? steps" in std_out
    assert "Completed 3 / ? steps" not in std_out
    assert "Completed 4 / ? steps" in std_out
    assert "Completed 5 / ? steps" not in std_out
    assert "Finished iterating" in std_out
