from modality_router import ModalityRouter


def test_image_route() -> None:
    router = ModalityRouter()
    assert router.predict("What color is the bag in the image?").label == "image"


def test_video_route() -> None:
    router = ModalityRouter()
    assert router.predict("Describe the final motion in the video sequence.").label == "video"


def test_table_route() -> None:
    router = ModalityRouter()
    assert router.predict("Compare values in this spreadsheet table").label == "table"
