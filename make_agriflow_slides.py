from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)
SOURCES = [
    Path("/Users/jeff/.codex/generated_images/01a0e915-a461-7220-9f5e-244c015c00a7/exec-8ff43525-3c67-4a5c-85b3-c6ee4903c149.png"),
    Path("/Users/jeff/.codex/generated_images/01a0e915-a461-7220-9f5e-244c015c00a7/exec-6081cffa-732b-4cdf-8ce0-45eecded8889.png"),
]
W, H = 1920, 1080
WHITE = (248, 248, 241)
MINT = (168, 239, 193)
MUTED = (204, 216, 207)
KOREAN = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
ENGLISH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def background(source, shade=0.62):
    img = Image.open(source).convert("RGB")
    iw, ih = img.size
    scale = max(W / iw, H / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    img = img.crop(((nw - W) // 2, (nh - H) // 2, (nw + W) // 2, (nh + H) // 2)).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    px = overlay.load()
    for x in range(W):
        # Left-side contrast while preserving the robot and greenhouse detail.
        alpha = int(255 * shade * max(0.0, 1.0 - (x / (W * 0.78)) ** 1.5))
        for y in range(H):
            px[x, y] = (0, 15, 10, alpha)
    return Image.alpha_composite(img, overlay)


def line(draw, y):
    draw.rounded_rectangle((118, y, 206, y + 8), radius=4, fill=MINT)


slide = background(SOURCES[0], 0.53)
d = ImageDraw.Draw(slide)
line(d, 115)
d.text((118, 155), "AGRICULTURAL ROBOTICS  /  DATA INFRASTRUCTURE", font=font(ENGLISH, 27), fill=MINT, stroke_width=0)
d.text((109, 285), "AgriFlow", font=font(ENGLISH, 159), fill=WHITE)
d.text((118, 512), "농업 로봇을 위한", font=font(KOREAN, 67), fill=WHITE)
d.text((118, 604), "시뮬레이션 데이터 생성 플랫폼", font=font(KOREAN, 67), fill=WHITE)
d.rounded_rectangle((118, 837, 868, 921), radius=26, fill=(13, 50, 35, 210), outline=(120, 175, 142, 165), width=2)
d.text((151, 854), "환경 생성  →  데이터 수집  →  품질 검증", font=font(KOREAN, 34), fill=WHITE)
d.text((118, 1010), "01  /  INTRODUCTION", font=font(ENGLISH, 22), fill=MUTED)
slide.convert("RGB").save(OUT / "01_agriflow_cover.png", quality=95)


slide = background(SOURCES[1], 0.76)
d = ImageDraw.Draw(slide)
line(d, 83)
d.text((118, 119), "PROBLEM STATEMENT", font=font(ENGLISH, 28), fill=MINT)
d.text((112, 205), "농업 로봇 데이터에는", font=font(KOREAN, 78), fill=WHITE)
d.text((112, 305), "두 개의 빈틈이 있습니다", font=font(KOREAN, 78), fill=WHITE)

card_x1, card_x2 = 118, 1105
for y1, y2 in [(465, 641), (662, 838)]:
    d.rounded_rectangle((card_x1, y1, card_x2, y2), radius=22, fill=(5, 27, 20, 225), outline=(116, 158, 132, 130), width=2)

d.text((154, 490), "01", font=font(ENGLISH, 43), fill=MINT)
d.text((236, 487), "기본 데이터셋 생성", font=font(KOREAN, 47), fill=WHITE)
d.text((237, 563), "새 로봇·작업에 맞는 시뮬레이션 데이터가 필요하다", font=font(KOREAN, 30), fill=MUTED)

d.text((154, 687), "02", font=font(ENGLISH, 43), fill=MINT)
d.text((236, 684), "특정 실패 상황 재현", font=font(KOREAN, 47), fill=WHITE)
d.text((237, 760), "가림·조명·시점·배치를 바꾼 사례가 부족하다", font=font(KOREAN, 30), fill=MUTED)

d.text((118, 889), "필요한 환경과 제약조건을 바꿔가며 데이터를 생성·검증합니다", font=font(KOREAN, 35), fill=MINT)
d.text((118, 1010), "농업 로봇 사례: Four Growers (토마토) · Synphony (딸기)  |  Source: Y Combinator", font=font(KOREAN, 21), fill=MUTED)
slide.convert("RGB").save(OUT / "02_agriflow_problem.png", quality=95)


# English versions for the linked Google Slides deck.
slide = background(SOURCES[0], 0.58)
d = ImageDraw.Draw(slide)
line(d, 115)
d.text((118, 155), "AGRICULTURAL ROBOTICS  /  DATA INFRASTRUCTURE", font=font(ENGLISH, 27), fill=MINT)
d.text((109, 285), "AgriFlow", font=font(ENGLISH, 159), fill=WHITE)
d.text((118, 526), "Simulation data for", font=font(ENGLISH, 65), fill=WHITE)
d.text((118, 607), "agricultural robots", font=font(ENGLISH, 65), fill=WHITE)
d.rounded_rectangle((118, 837, 854, 921), radius=26, fill=(13, 50, 35, 210), outline=(120, 175, 142, 165), width=2)
d.text((151, 856), "Build scenes  →  Generate data  →  Verify quality", font=font(ENGLISH, 29), fill=WHITE)
d.text((118, 1010), "01  /  INTRODUCTION", font=font(ENGLISH, 22), fill=MUTED)
slide.convert("RGB").save(OUT / "01_agriflow_cover_en.png", quality=95)

slide = background(SOURCES[1], 0.76)
d = ImageDraw.Draw(slide)
line(d, 83)
d.text((118, 119), "PROBLEM STATEMENT", font=font(ENGLISH, 28), fill=MINT)
d.text((112, 213), "Agricultural robots face", font=font(ENGLISH, 73), fill=WHITE)
d.text((112, 308), "two data gaps", font=font(ENGLISH, 73), fill=WHITE)
for y1, y2 in [(465, 641), (662, 838)]:
    d.rounded_rectangle((118, y1, 1105, y2), radius=22, fill=(5, 27, 20, 225), outline=(116, 158, 132, 130), width=2)
d.text((154, 493), "01", font=font(ENGLISH, 42), fill=MINT)
d.text((238, 492), "Baseline data generation", font=font(ENGLISH, 44), fill=WHITE)
d.text((238, 566), "New robots and tasks need tailored simulation datasets.", font=font("/System/Library/Fonts/Supplemental/Arial.ttf", 28), fill=MUTED)
d.text((154, 690), "02", font=font(ENGLISH, 42), fill=MINT)
d.text((238, 689), "Failure-case reproduction", font=font(ENGLISH, 44), fill=WHITE)
d.text((238, 763), "Occlusion, lighting and viewpoint failures are hard to sample.", font=font("/System/Library/Fonts/Supplemental/Arial.ttf", 27), fill=MUTED)
d.text((118, 891), "Vary scene constraints. Generate targeted data. Verify quality.", font=font(ENGLISH, 32), fill=MINT)
d.text((118, 1010), "Agricultural robotics examples: Four Growers (tomatoes) · Synphony (strawberries)  |  Source: YC", font=font("/System/Library/Fonts/Supplemental/Arial.ttf", 20), fill=MUTED)
slide.convert("RGB").save(OUT / "02_agriflow_problem_en.png", quality=95)
