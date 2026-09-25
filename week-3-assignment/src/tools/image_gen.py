"""image_generator -- generate a visual matching the topic.

Three-tier fallback ladder, each tried only if the previous one failed, so
the UI can never show a broken image even offline:
  1. Cloudflare Workers AI, FLUX.1-schnell -- 10k neurons/day free, no card,
     confirmed live during design (HTTP 200, ~3s, good output).
  2. Pollinations.ai -- keyless GET, no signup, but ~1 req/15s anonymously
     and no uptime guarantee.
  3. A locally rendered typographic card (PIL, no network at all).

Before any of that: a fast LLM call turns the topic (or, if the planner wired
image_generator to depend on content_generator, the actual post) into a
concrete visual brief. Diffusion models render a bare topic string badly --
there's no subject, no style, and no instruction to avoid rendering text
(which they render badly). One cheap call fixes that.
"""

from __future__ import annotations

import asyncio
import base64
import io
import textwrap
import urllib.parse

import httpx
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_IMAGE_MODEL,
    CLOUDFLARE_STEPS,
    CLOUDFLARE_TIMEOUT,
    IMAGE_SIZE,
    MODEL_FAST,
    POLLINATIONS_TIMEOUT,
    POLLINATIONS_URL,
)
from src import llm
from src.tools._util import find_dep, tool_ok
from src.types import StepResult

NAME = "image_generator"
DESCRIPTION = (
    "Generate a visual matching the topic (or, if it depends on "
    "content_generator, the actual post). Always returns an image -- falls "
    "back through Cloudflare FLUX -> Pollinations -> a local rendered card."
)

ART_DIRECTOR_SYSTEM = (
    "You turn a topic or LinkedIn post into a concrete, concise image-generation "
    "prompt for a diffusion model. Describe a specific visual subject, composition, "
    "and style (e.g. 'isometric illustration', 'flat vector', 'muted teal and slate "
    "palette'). Never ask for readable text in the image -- diffusion models render "
    "it badly. One or two sentences, no preamble."
)


class ImageGenArgs(BaseModel):
    style: str | None = Field(
        default=None, description="optional art-direction hint, e.g. 'flat vector, teal palette'"
    )


async def _art_brief(topic: str, deps: dict[int, StepResult], style: str | None) -> str:
    gen_dep = find_dep(deps, "content_generator")
    if gen_dep and gen_dep.ok and gen_dep.output and not gen_dep.output.get("refused"):
        subject = f"{gen_dep.output.get('headline', '')} -- {gen_dep.output.get('body', '')[:300]}"
    else:
        subject = topic

    style_line = f"\n\nStyle hint: {style}" if style else ""
    # llm.chat is a sync network call; run it off the event loop so it can't
    # stall the other tools in the same parallel wave (see D9's scheduler).
    brief = await asyncio.to_thread(
        llm.chat,
        ART_DIRECTOR_SYSTEM,
        f"Subject: {subject}{style_line}",
        MODEL_FAST,
        0.7,
        120,
    )
    return brief.strip() or (
        f"minimalist flat-vector illustration representing: {topic}, "
        "professional, LinkedIn-appropriate, no text"
    )


async def _try_cloudflare(prompt: str) -> dict | None:
    if not (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN):
        return None
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}"
        f"/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    )
    try:
        async with httpx.AsyncClient(timeout=CLOUDFLARE_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
                json={"prompt": prompt, "steps": CLOUDFLARE_STEPS},
            )
        data = resp.json()
        if data.get("success") and data.get("result", {}).get("image"):
            return {"tier": "cloudflare", "image_b64": data["result"]["image"], "mime": "image/jpeg"}
    except Exception:  # noqa: BLE001 -- any failure here just means "try the next tier"
        pass
    return None


async def _try_pollinations(prompt: str) -> dict | None:
    url = POLLINATIONS_URL.format(prompt=urllib.parse.quote(prompt))
    try:
        async with httpx.AsyncClient(timeout=POLLINATIONS_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code == 200 and resp.content:
            mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
            return {
                "tier": "pollinations",
                "image_b64": base64.b64encode(resp.content).decode(),
                "mime": mime,
            }
    except Exception:  # noqa: BLE001
        pass
    return None


def _render_local_card(topic: str) -> dict:
    """No network dependency at all -- the guaranteed-to-work last resort.

    Styled to look deliberate rather than broken, using the UI's palette:
    --ink ground, --paper text (light on dark, for actual readability -- teal
    on ink measured far too low-contrast to read), --signal as an accent rule.
    """
    w, h = IMAGE_SIZE
    ink, paper, signal = (14, 26, 34), (237, 241, 244), (11, 110, 110)
    img = Image.new("RGB", (w, h), color=ink)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=44)
    except TypeError:  # older Pillow: load_default() takes no args
        font = ImageFont.load_default()

    wrapped = textwrap.wrap(topic, width=24)[:6]
    line_height = 60
    total_h = len(wrapped) * line_height
    y = (h - total_h) // 2

    # accent rule above the text block, echoing the UI's left-border motif
    draw.rectangle([(w // 2 - 60, y - 40), (w // 2 + 60, y - 36)], fill=signal)

    for line in wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        draw.text(((w - line_w) // 2, y), line, font=font, fill=paper)
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"tier": "local_card", "image_b64": base64.b64encode(buf.getvalue()).decode(), "mime": "image/png"}


async def run(topic: str, args: dict, deps: dict[int, StepResult]) -> dict:
    parsed = ImageGenArgs.model_validate(args or {})
    prompt = await _art_brief(topic, deps, parsed.style)

    for attempt in (_try_cloudflare, _try_pollinations):
        result = await attempt(prompt)
        if result:
            return tool_ok(prompt=prompt, **result)

    return tool_ok(prompt=prompt, **_render_local_card(topic))


if __name__ == "__main__":
    import json
    import sys

    async def _demo():
        topic = " ".join(sys.argv[1:]) or "recent trends in GenAI agents for backend engineers"
        out = await run(topic, {}, {})
        return {k: (v[:60] + "..." if k == "image_b64" else v) for k, v in out.items()}

    print(json.dumps(asyncio.run(_demo()), indent=2))
