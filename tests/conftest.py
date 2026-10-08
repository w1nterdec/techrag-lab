import pytest


@pytest.fixture
def standard_sample():
    return {
        "id": "sample", "instruction": "回答问题", "input": "如何查看 Linux 端口？",
        "output": "可以使用 ss -tlnp 查看系统监听端口。",
        "metadata": {"category": "linux", "language": "zh", "source": "manual"},
    }
