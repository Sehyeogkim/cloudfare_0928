from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)
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


def base(label, title, subtitle):
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((115, 80, 203, 88), radius=4, fill=MINT)
    d.text((115, 120), label, font=f(BOLD, 28), fill=MINT)
    d.text((110, 184), title, font=f(BOLD, 75), fill=WHITE)
    d.text((115, 286), subtitle, font=f(REG, 34), fill=MUTED)
    return image, d


def arrow(d, x1, y, x2):
    d.line((x1, y, x2 - 13, y), fill=MINT, width=5)
    d.polygon([(x2 - 20, y - 12), (x2, y), (x2 - 20, y + 12)], fill=MINT)


slide, d = base("SYSTEM ARCHITECTURE", "Five agents. One data pipeline.", "Two customer requests feed the same build → generate → validate workflow.")

d.rounded_rectangle((115, 372, 466, 432), radius=18, fill=(17, 60, 43), outline=MINT, width=2)
d.text((141, 387), "New dataset request", font=f(BOLD, 27), fill=WHITE)
d.rounded_rectangle((490, 372, 869, 432), radius=18, fill=(67, 40, 25), outline=ORANGE, width=2)
d.text((516, 387), "Failure-case request", font=f(BOLD, 27), fill=WHITE)
d.line((290, 432, 290, 475), fill=MINT, width=3)
d.line((680, 432, 680, 455), fill=ORANGE, width=3)
d.line((680, 455, 290, 455), fill=ORANGE, width=3)
d.polygon([(279, 473), (290, 489), (301, 473)], fill=MINT)

cards = [
    ("01", "Customer", ["Request → data spec", "or failure conditions"]),
    ("02", "Environment", ["Builds scenes; selects", "Isaac Sim / MuJoCo"]),
    ("03", "Data Collection", ["Runs episodes; records", "images, states, actions"]),
    ("04", "QA", ["Checks validity, task", "success and coverage"]),
    ("05", "Payment", ["Quotes, charges and", "releases the dataset"]),
]
x0, cw, gap = 115, 306, 58
for i, (number, title, body) in enumerate(cards):
    x = x0 + i * (cw + gap)
    d.rounded_rectangle((x, 490, x + cw, 739), radius=24, fill=CARD, outline=(92, 156, 117), width=3)
    d.text((x + 26, 514), number, font=f(BOLD, 30), fill=MINT)
    d.text((x + 26, 563), title, font=f(BOLD, 32 if len(title) < 14 else 26), fill=WHITE)
    for j, line in enumerate(body):
        d.text((x + 26, 637 + 36 * j), line, font=f(REG, 23), fill=MUTED)
    if i < 4:
        arrow(d, x + cw + 7, 615, x + cw + gap - 6)

d.rounded_rectangle((115, 790, 1804, 946), radius=22, fill=(7, 34, 24), outline=(77, 132, 99), width=2)
d.text((148, 817), "ORCHESTRATION", font=f(BOLD, 25), fill=MINT)
d.text((148, 856), "Brainbase agents + Cloudflare Sandboxes", font=f(BOLD, 31), fill=WHITE)
d.text((1065, 817), "SIMULATION EXECUTION", font=f(BOLD, 25), fill=MINT)
d.text((1065, 856), "Isaac Sim on RunPod GPU", font=f(BOLD, 31), fill=WHITE)
d.text((115, 1010), "03  /  ARCHITECTURE", font=f(BOLD, 22), fill=MUTED)
slide.save(OUT / "03_agriflow_architecture_en.png")


slide, d = base("CUSTOMERS & MARKET", "Start with greenhouse robotics.", "Our first buyer is the team building the robot, not the grower operating it.")

d.rounded_rectangle((115, 378, 940, 918), radius=28, fill=CARD, outline=(92, 156, 117), width=3)
d.text((157, 416), "BEACHHEAD CUSTOMER", font=f(BOLD, 28), fill=MINT)
d.text((157, 479), "Greenhouse harvesting", font=f(BOLD, 51), fill=WHITE)
d.text((157, 544), "robot companies", font=f(BOLD, 51), fill=WHITE)
d.text((157, 635), "Tomato harvesting first", font=f(BOLD, 32), fill=ORANGE)
d.text((157, 697), "Buyers: robotics, perception and data leads", font=f(REG, 28), fill=MUTED)
d.text((157, 743), "Use cases: new datasets + recurring failures", font=f(REG, 28), fill=MUTED)
d.line((157, 813, 895, 813), fill=(83, 131, 100), width=2)
d.text((157, 842), "Category example: Four Growers GR-100", font=f(REG, 25), fill=MUTED)

d.rounded_rectangle((974, 378, 1805, 918), radius=28, fill=(12, 50, 35), outline=(92, 156, 117), width=3)
d.text((1016, 416), "BROADER MARKET SIGNAL", font=f(BOLD, 28), fill=MINT)
d.text((1008, 472), "$14.7B", font=f(BOLD, 118), fill=WHITE)
d.text((1016, 612), "Global agricultural robots market", font=f(BOLD, 35), fill=WHITE)
d.text((1016, 665), "2024 estimate · Grand View Research", font=f(REG, 27), fill=MUTED)
d.line((1016, 729, 1765, 729), fill=(83, 131, 100), width=2)
d.text((1016, 756), "~19.5k agricultural service robots", font=f(BOLD, 29), fill=MINT)
d.text((1016, 803), "reported sold in 2024 · IFR supplier sample", font=f(REG, 25), fill=MUTED)

d.text((115, 950), "Market figures include many robot categories; AgriFlow's simulation-data spend is not yet measured.", font=f(REG, 27), fill=MUTED)
d.text((115, 1010), "04  /  TARGET CUSTOMERS & MARKET  |  Sources: Four Growers, Grand View Research, IFR World Robotics 2025", font=f(REG, 20), fill=MUTED)
slide.save(OUT / "04_agriflow_customers_market_en.png")
