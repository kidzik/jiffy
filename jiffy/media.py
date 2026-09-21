"""Embedded image helpers shared by inference backends."""
import base64
import io


def image_data_url(path):
    """Explicit client-side file loading; inference accepts embedded images."""
    from PIL import Image
    with Image.open(path) as image:
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def decode_image(value):
    from PIL import Image, UnidentifiedImageError
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("image exceeds 16 million pixels")
            return image.convert("RGB")
    except (IndexError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("invalid embedded image") from error
