from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent / "slides" / "04_agriflow_architecture_loop_en.png"
W, H = 1920, 1080
BG = (3, 22, 16)
CARD = (10, 43, 31)
WHITE = (248, 248, 241)
MINT = (168, 239, 193)
MUTED = (204, 216, 207)
ORANGE = (251, 157, 102)
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def f(path, size):
    return ImageFont.truetype(path, size)


def arrow_right(draw, x1, y, x2, color=MINT, width=5):
    draw.line((x1, y, x2 - 15, y), fill=color, width=width)
    draw.polygon([(x2 - 23, y - 13), (x2, y), (x2 - 23, y + 13)], fill=color)


def arrow_up(draw, x, y1, y2, color=ORANGE, width=6):
    draw.line((x, y1, x, y2 + 18), fill=color, width=width)
    draw.polygon([(x - 14, y2 + 22), (x, y2), (x + 14, y2 + 22)], fill=color)


im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
d.rounded_rectangle((115, 68, 204, 76), radius=4, fill=MINT)
d.text((115, 110), "AGENT ARCHITECTURE", font=f(BOLD, 27), fill=MINT)
d.text((108, 166), "A closed loop for reliable robot data", font=f(BOLD, 68), fill=WHITE)
d.text((115, 260), "Every failed QA check becomes a new simulation job with adjusted conditions.", font=f(REG, 32), fill=MUTED)

# Two input modes converge on one customer specification.
d.rounded_rectangle((115, 345, 404, 400), radius=16, fill=(18, 58, 42), outline=MINT, width=2)
d.rounded_rectangle((115, 410, 404, 465), radius=16, fill=(68, 40, 25), outline=ORANGE, width=2)
d.text((137, 357), "New dataset order", font=f(BOLD, 26), fill=WHITE)
d.text((137, 422), "Failure video / log", font=f(BOLD, 26), fill=WHITE)
d.line((404, 372, 427, 372), fill=MINT, width=3)
d.line((404, 437, 427, 437), fill=ORANGE, width=3)
d.line((427, 372, 427, 437), fill=MINT, width=3)
arrow_right(d, 427, 405, 468)

d.rounded_rectangle((469, 344, 855, 464), radius=21, fill=CARD, outline=(89, 153, 114), width=3)
d.text((495, 358), "01  CUSTOMER AGENT", font=f(BOLD, 30), fill=WHITE)
d.text((495, 407), "DatasetSpec: task, robot,", font=f(REG, 23), fill=MUTED)
d.text((495, 434), "sensors + acceptance rules", font=f(REG, 23), fill=MUTED)

d.rounded_rectangle((923, 344, 1810, 464), radius=21, fill=(12, 50, 35), outline=(89, 153, 114), width=3)
d.text((954, 359), "BRAINBASE ORCHESTRATION", font=f(BOLD, 28), fill=MINT)
d.text((954, 408), "Handoffs  ·  execution state  ·  retries  ·  approvals", font=f(REG, 27), fill=WHITE)
arrow_right(d, 867, 405, 915)

# Main production path.
nodes = [
    ("02", "ENVIRONMENT", ["Scene, robot, camera", "and variation ranges"]),
    ("GPU", "SIMULATOR", ["Isaac Sim on RunPod", "MuJoCo when suitable"]),
    ("03", "DATA COLLECTION", ["RGB-D, state, actions", "labels + metadata"]),
    ("04", "QA AGENT", ["Integrity, task success", "coverage + replay"]),
]
x0, card_w, gap, top, bottom = 115, 388, 70, 528, 736
for i, (num, title, lines) in enumerate(nodes):
    x = x0 + i * (card_w + gap)
    d.rounded_rectangle((x, top, x + card_w, bottom), radius=23, fill=CARD, outline=(89, 153, 114), width=3)
    d.text((x + 27, top + 23), num, font=f(BOLD, 27), fill=MINT)
    d.text((x + 27, top + 67), title, font=f(BOLD, 31 if title != "DATA COLLECTION" else 29), fill=WHITE)
    d.text((x + 27, top + 130), lines[0], font=f(REG, 25), fill=MUTED)
    d.text((x + 27, top + 164), lines[1], font=f(REG, 25), fill=MUTED)
    if i < 3:
        arrow_right(d, x + card_w + 6, 631, x + card_w + gap - 8)

# Customer spec drives the first environment job.
d.line((662, 464, 662, 490), fill=MINT, width=4)
d.line((309, 490, 662, 490), fill=MINT, width=4)
d.line((309, 490, 309, 509), fill=MINT, width=4)
d.polygon([(296, 505), (309, 523), (322, 505)], fill=MINT)

# The distinctive failure feedback loop: QA report routes to Environment.
qa_cx = x0 + 3 * (card_w + gap) + 155
env_cx = x0 + 192
d.line((qa_cx, bottom, qa_cx, 823), fill=ORANGE, width=7)
d.line((env_cx, 823, qa_cx, 823), fill=ORANGE, width=7)
arrow_up(d, env_cx, 823, bottom, color=ORANGE, width=7)
d.rounded_rectangle((522, 781, 1395, 863), radius=18, fill=(67, 40, 25), outline=ORANGE, width=2)
d.text((553, 797), "QA FAIL  →  revise scene constraints / seeds  →  rerun", font=f(BOLD, 28), fill=WHITE)

# Accepted episodes proceed to commercial handoff.
d.line((qa_cx + 90, bottom, qa_cx + 90, 892), fill=MINT, width=6)
d.polygon([(qa_cx + 76, 876), (qa_cx + 90, 896), (qa_cx + 104, 876)], fill=MINT)
d.text((qa_cx + 110, 845), "QA PASS", font=f(BOLD, 23), fill=MINT)
d.rounded_rectangle((1422, 899, 1810, 1004), radius=21, fill=(12, 50, 35), outline=MINT, width=3)
d.text((1451, 916), "05  PAYMENT AGENT", font=f(BOLD, 26), fill=WHITE)
d.text((1451, 960), "Checkout → dataset + QA report", font=f(REG, 23), fill=MUTED)

d.text((115, 938), "Agent logic: Cloudflare Sandboxes", font=f(REG, 25), fill=MUTED)
d.text((115, 978), "Retry until acceptance rules pass or the run budget is reached.", font=f(BOLD, 27), fill=MINT)
d.text((115, 1040), "04  /  CLOSED-LOOP ARCHITECTURE", font=f(BOLD, 19), fill=MUTED)
im.save(OUT)
