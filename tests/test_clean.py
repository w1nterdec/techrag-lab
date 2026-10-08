
from data.pipeline.clean import clean_text, clean_dataset


def test_clean_text_removes_whitespace():
    text = "   Hello Linux   "

    result = clean_text(text)

    assert result == "Hello Linux"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_dataset_removes_empty_input():
    dataset = [
        {
            "input": "",
            "output": "Valid answer",
        },
        {
            "input": "What is Docker?",
            "output": "Docker is a container platform.",
        },
    ]

    result = clean_dataset(dataset)

    assert len(result) == 1
    assert result[0]["input"] == "What is Docker?"


def test_clean_dataset_removes_empty_output():
    dataset = [
        {
            "input": "What is Linux?",
            "output": "",
        }
    ]

    result = clean_dataset(dataset)

    assert len(result) == 0


def test_clean_dataset_strips_whitespace():
    dataset = [
        {
            "input": "  What is WSL2?  ",
            "output": "  A Linux compatibility environment.  ",
        }
    ]

    result = clean_dataset(dataset)

    assert result[0]["input"] == "What is WSL2?"
    assert result[0]["output"] == "A Linux compatibility environment."
