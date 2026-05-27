from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import re
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from app.core.errors import RenderError
from app.schemas.state import RenderResult


class AssetStore:
    """Persist rendered PNGs and HTML source to disk.

    Files are written under *base_dir* and served via *public_path* URL prefix.
    """

    def __init__(self, base_dir: str | Path = "generated", public_path: str = "/assets") -> None:
        self.base_dir = Path(base_dir)
        self.public_path = public_path.rstrip("/")

    def _safe_job_id(self, job_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise RenderError("job_id contains unsafe characters")
        return job_id

    async def save_render(self, result: RenderResult, *, job_id: str, iteration: int) -> RenderResult:
        if not result.image_base64:
            return result

        self.base_dir.mkdir(parents=True, exist_ok=True)
        extension = "jpg" if result.mime_type == "image/jpeg" else "png"
        job_id = self._safe_job_id(job_id)
        filename = f"{job_id}_{iteration}.{extension}"
        target = self.base_dir / filename
        try:
            target.write_bytes(base64.b64decode(result.image_base64))
        except ValueError as exc:
            raise RenderError("render result contains invalid base64 image data") from exc

        return result.model_copy(update={"image_url": f"{self.public_path}/{filename}"})

    async def save_html(self, html: str, *, job_id: str, iteration: int) -> str:
        """Persist the HTML source and return its public URL."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        job_id = self._safe_job_id(job_id)
        filename = f"{job_id}_{iteration}.html"
        target = self.base_dir / filename
        target.write_text(html, encoding="utf-8")
        return f"{self.public_path}/{filename}"

    async def save_reference_image(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        mime_type: str,
    ) -> str:
        """Persist a user-uploaded reference image and return its public URL."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        upload_dir = self.base_dir / "reference_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(filename).suffix.lower()
        extension = suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else _extension_from_mime(mime_type)
        target_name = f"ref_{uuid4().hex}{extension}"
        target = upload_dir / target_name

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise RenderError("uploaded file is not a valid image") from exc

        target.write_bytes(image_bytes)
        return f"{self.public_path}/reference_uploads/{target_name}"


    async def load_html_by_url(self, html_url: str) -> str:
        """Read an HTML file that was previously saved under *base_dir*.

        Only accepts paths that resolve within ``self.base_dir`` and end with
        ``.html``.  Rejects paths with ``..`` traversal or absolute paths.
        """
        if not html_url:
            raise RenderError("html_url is empty")

        if not html_url.startswith(self.public_path + "/"):
            raise RenderError(f"html_url must start with {self.public_path}/")

        relative = html_url[len(self.public_path):].lstrip("/")

        # Reject dangerous patterns.
        if ".." in relative or relative.startswith("/") or "\\" in relative:
            raise RenderError("html_url contains unsafe path components")

        if not relative.lower().endswith(".html"):
            raise RenderError("html_url must point to an .html file")

        base = self.base_dir.resolve()
        target = (self.base_dir / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise RenderError("html_url escapes the asset directory") from exc

        if not target.is_file():
            raise RenderError(f"HTML file not found: {relative}")

        return target.read_text(encoding="utf-8")

    async def save_refined_html(self, html: str, *, job_id: str, iteration: int) -> str:
        """Persist a refined HTML and return its public URL.

        Filename: ``{job_id}_refine_{iteration}.html``
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        job_id = self._safe_job_id(job_id)
        filename = f"{job_id}_refine_{iteration}.html"
        target = self.base_dir / filename
        target.write_text(html, encoding="utf-8")
        return f"{self.public_path}/{filename}"

    async def save_refined_png(self, image_base64: str, *, job_id: str, iteration: int) -> str:
        """Persist a refined PNG and return its public URL.

        Filename: ``{job_id}_refine_{iteration}.png``
        """
        if not image_base64:
            raise RenderError("refined result image data is empty")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        job_id = self._safe_job_id(job_id)
        filename = f"{job_id}_refine_{iteration}.png"
        target = self.base_dir / filename
        try:
            target.write_bytes(base64.b64decode(image_base64))
        except ValueError as exc:
            raise RenderError("refined result contains invalid base64 image data") from exc
        return f"{self.public_path}/{filename}"


def _extension_from_mime(mime_type: str) -> str:
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    return mapping.get(mime_type.lower(), ".png")
