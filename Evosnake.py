"""
EvoSnake — Neuroevolution with PyTorch Policy Networks
=======================================================
Requirements:
    pip install pygame numpy torch

Architecture:
    Each agent carries a PyTorch nn.Module (SnakeNet).
    Evolution is done via weight cloning + Gaussian mutation
    (no gradient-based training — pure genetic algorithm).

Controls:
    SPACE      Pause / Resume
    R          Reset simulation
    N          Force next generation
    UP / DOWN  Speed up / slow down
    M          Cycle mutation rate
    A          Toggle architecture info overlay
    D          Toggle debug (fitness labels)
    ESC / Q    Quit
"""

import sys, math, random
import numpy as np
import pygame
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────────────────
W, H           = 960, 640
SIM_W, SIM_H   = 640, 640
PANEL_W        = W - SIM_W

POP_SIZE       = 32
FOOD_COUNT     = 22
STEPS_PER_GEN  = 650
AGENT_SPEED    = 2.8
ENERGY_DRAIN   = 0.00065
ENERGY_FOOD    = 0.42
EAT_RADIUS     = 11
ELITE_FRAC     = 0.20
TRAIL_LEN      = 28
FPS_CAP        = 120

INPUT_DIM   = 8
HIDDEN_DIM  = 16
OUTPUT_DIM  = 2

MUT_RATES  = [0.02, 0.07, 0.18]
MUT_LABELS = ["Low  0.02", "Med  0.07", "High 0.18"]

DEVICE = torch.device("cpu")   

# ──────────────────────────────────────────────────────────
# COLOURS
# ──────────────────────────────────────────────────────────
BG          = ( 10,  14,  22)
FOOD_OUT    = (245, 158,  11)
FOOD_IN     = (252, 211,  77)
ELITE_COL   = ( 74, 222, 128)
NORMAL_COL  = ( 96, 165, 250)
DEAD_COL    = ( 55,  55,  75)
VEC_COL     = (239,  68,  68)
PANEL_BG    = ( 16,  20,  32)
PANEL_LINE  = ( 38,  48,  70)
TEXT_HI     = (220, 220, 235)
TEXT_LO     = (115, 125, 152)
COL_GREEN   = ( 74, 222, 128)
COL_BLUE    = ( 96, 165, 250)
COL_AMBER   = (245, 158,  11)


# ──────────────────────────────────────────────────────────
# PYTORCH POLICY NETWORK
# ──────────────────────────────────────────────────────────
class SnakeNet(nn.Module):
    """
    Lightweight MLP policy:
      Input  (8):  food_dx, food_dy, food_dist, vx, vy,
                   energy, wall_dx, wall_dy
      Hidden (16): tanh
      Output (2):  steering (dx, dy) via tanh  ∈ [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
            nn.Tanh(),
        )
        # Xavier init for cleaner early behaviour
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Genetic operators ──────────────────────────────
    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view_as(p))
            idx += n

    def clone(self) -> "SnakeNet":
        child = SnakeNet().to(DEVICE)
        child.set_flat_params(self.get_flat_params().clone())
        return child

    def mutate(self, rate: float) -> "SnakeNet":
        child = self.clone()
        flat  = child.get_flat_params()
        # Sparse perturbation: only ~30 % of weights mutated
        mask  = torch.rand_like(flat) < 0.30
        noise = torch.randn_like(flat) * rate
        child.set_flat_params(flat + noise * mask)
        return child

    @staticmethod
    def crossover(a: "SnakeNet", b: "SnakeNet") -> "SnakeNet":
        """Uniform crossover — randomly pick each weight from either parent."""
        fa, fb = a.get_flat_params(), b.get_flat_params()
        mask   = torch.rand_like(fa) > 0.5
        child  = SnakeNet().to(DEVICE)
        child.set_flat_params(torch.where(mask, fa, fb))
        return child

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────
class Agent:
    def __init__(self, net: SnakeNet = None):
        self.x  = random.uniform(30, SIM_W - 30)
        self.y  = random.uniform(30, SIM_H - 30)
        ang     = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(ang) * AGENT_SPEED
        self.vy = math.sin(ang) * AGENT_SPEED
        self.energy  = 1.0
        self.fitness = 0
        self.alive   = True
        self.trail: list = []
        self.net    = net if net is not None else SnakeNet().to(DEVICE)

    # ── Sensing ───────────────────────────────────────
    def _nearest_food(self, foods):
        bx, by, bd = 0.0, 0.0, float("inf")
        for f in foods:
            d = math.hypot(f[0] - self.x, f[1] - self.y)
            if d < bd:
                bd, bx, by = d, f[0], f[1]
        return bx, by, bd

    # ── One simulation tick ───────────────────────────
    def step(self, foods) -> list | None:
        if not self.alive:
            return None

        fx, fy, dist = self._nearest_food(foods)

        # Wall repulsion signal (normalised distance to nearest wall)
        wx = min(self.x, SIM_W - self.x) / (SIM_W / 2)
        wy = min(self.y, SIM_H - self.y) / (SIM_H / 2)

        inv_d   = 1.0 / max(dist, 1.0)
        dx_norm = (fx - self.x) * inv_d
        dy_norm = (fy - self.y) * inv_d

        inp = torch.tensor([
            dx_norm,
            dy_norm,
            min(dist / 450.0, 1.0),
            self.vx / AGENT_SPEED,
            self.vy / AGENT_SPEED,
            self.energy,
            wx,
            wy,
        ], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            out = self.net(inp).cpu().numpy()  # shape (2,)

        # Blend: neural steering + soft food-seeking bias
        blend   = 0.50
        tx      = out[0] * (1 - blend) + dx_norm * blend
        ty      = out[1] * (1 - blend) + dy_norm * blend
        self.vx = self.vx * 0.55 + tx * AGENT_SPEED * 0.45
        self.vy = self.vy * 0.55 + ty * AGENT_SPEED * 0.45
        spd = math.hypot(self.vx, self.vy) or 1.0
        self.vx = (self.vx / spd) * AGENT_SPEED
        self.vy = (self.vy / spd) * AGENT_SPEED

        self.x = max(6, min(SIM_W - 6, self.x + self.vx))
        self.y = max(6, min(SIM_H - 6, self.y + self.vy))

        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)

        self.energy -= ENERGY_DRAIN
        if self.energy <= 0:
            self.alive = False
            return None

        if dist < EAT_RADIUS:
            self.fitness += 1
            self.energy   = min(1.0, self.energy + ENERGY_FOOD)
            return foods[foods.index([fx, fy])]   # return the eaten food object

        return None


# ──────────────────────────────────────────────────────────
# EVOLUTION ENGINE
# ──────────────────────────────────────────────────────────
def evolve(agents: list[Agent], mut_rate: float, pop_size: int) -> list[Agent]:
    ranked  = sorted(agents, key=lambda a: a.fitness, reverse=True)
    n_elite = max(2, int(len(ranked) * ELITE_FRAC))
    elite   = ranked[:n_elite]

    new_agents: list[Agent] = []

    # 1) Preserve top 2 unchanged
    for e in elite[:2]:
        new_agents.append(Agent(net=e.net.clone()))

    # 2) Crossover offspring (~25 % of remaining slots)
    n_cross = max(0, (pop_size - 2) // 4)
    for _ in range(n_cross):
        p1, p2 = random.sample(elite, 2)
        child_net = SnakeNet.crossover(p1.net, p2.net)
        new_agents.append(Agent(net=child_net))

    # 3) Mutated offspring fill the rest
    while len(new_agents) < pop_size:
        parent    = random.choice(elite)
        child_net = parent.net.mutate(mut_rate)
        new_agents.append(Agent(net=child_net))

    return new_agents


def spawn_food(count: int) -> list:
    m = 15
    return [[random.uniform(m, SIM_W - m), random.uniform(m, SIM_H - m)]
            for _ in range(count)]


# ──────────────────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────────────────
def draw_grid(surf):
    s = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    for x in range(0, SIM_W, 30):
        pygame.draw.line(s, (255, 255, 255, 7), (x, 0), (x, SIM_H))
    for y in range(0, SIM_H, 30):
        pygame.draw.line(s, (255, 255, 255, 7), (0, y), (SIM_W, y))
    surf.blit(s, (0, 0))


def draw_food(surf, foods):
    for f in foods:
        x, y = int(f[0]), int(f[1])
        pygame.draw.circle(surf, FOOD_OUT, (x, y), 6)
        pygame.draw.circle(surf, FOOD_IN,  (x, y), 3)


def draw_agent(surf, agent: Agent, is_elite: bool, debug: bool):
    col = ELITE_COL if is_elite else NORMAL_COL
    if not agent.alive:
        col = DEAD_COL
    ax, ay = int(agent.x), int(agent.y)

    # Trail
    if len(agent.trail) > 1:
        ts = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        n  = len(agent.trail)
        r, g, b = col
        for i in range(1, n):
            alpha = int((i / n) * (85 if is_elite else 55))
            pygame.draw.line(ts, (r, g, b, alpha),
                             (int(agent.trail[i-1][0]), int(agent.trail[i-1][1])),
                             (int(agent.trail[i][0]),   int(agent.trail[i][1])), 2)
        surf.blit(ts, (0, 0))

    # Body
    pygame.draw.circle(surf, col,     (ax, ay), 7)
    pygame.draw.circle(surf, (0,0,0), (ax, ay), 7, 1)

    # Energy arc
    if agent.alive and agent.energy > 0.02:
        arc_r = pygame.Rect(ax - 10, ay - 10, 20, 20)
        angle = agent.energy * 2 * math.pi
        arc_s = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        r, g, b = col
        pygame.draw.arc(arc_s, (r, g, b, 160),
                        arc_r, math.pi / 2 - angle, math.pi / 2, 2)
        surf.blit(arc_s, (0, 0))

    # Direction vector
    if agent.alive:
        spd = math.hypot(agent.vx, agent.vy) or 1.0
        ex  = ax + int((agent.vx / spd) * 14)
        ey  = ay + int((agent.vy / spd) * 14)
        pygame.draw.line(surf, VEC_COL, (ax, ay), (ex, ey), 1)

    if debug and agent.fitness > 0:
        ft = pygame.font.SysFont("monospace", 10)
        surf.blit(ft.render(str(agent.fitness), True, (200, 200, 200)), (ax + 9, ay - 8))


def draw_chart(surf, x, y, w, h, best_h, avg_h, font_s):
    pygame.draw.rect(surf, (22, 28, 44), (x, y, w, h), border_radius=4)
    if len(best_h) < 2:
        return
    mx = max(max(best_h), 1)

    def pts(data, col):
        points = [(x + int(i / (len(data)-1) * (w-8)) + 4,
                   y + h - 4 - int((v / mx) * (h-10)))
                  for i, v in enumerate(data)]
        if len(points) > 1:
            pygame.draw.lines(surf, col, False, points, 2)

    pts(best_h, COL_GREEN)
    pts(avg_h,  COL_BLUE)
    pygame.draw.line(surf, COL_GREEN, (x+4,  y+h+6), (x+16, y+h+6), 2)
    surf.blit(font_s.render("best", True, TEXT_LO), (x+19, y+h+1))
    pygame.draw.line(surf, COL_BLUE,  (x+52, y+h+6), (x+64, y+h+6), 2)
    surf.blit(font_s.render("avg",  True, TEXT_LO), (x+67, y+h+1))


def draw_arch_overlay(surf, net: SnakeNet, font_s, font_n):
    """Semi-transparent architecture info card."""
    ow, oh = 310, 170
    ox, oy = (SIM_W - ow) // 2, (SIM_H - oh) // 2
    card = pygame.Surface((ow, oh), pygame.SRCALPHA)
    card.fill((16, 20, 36, 220))
    pygame.draw.rect(card, (60, 80, 120, 180), (0, 0, ow, oh), 1, border_radius=8)

    lines = [
        ("PyTorch SnakeNet", TEXT_HI, font_n),
        ("", TEXT_LO, font_s),
        (f"  Input   → {INPUT_DIM}  neurons  (sensing)", TEXT_LO, font_s),
        (f"  Hidden1 → {HIDDEN_DIM}  neurons  (tanh)", COL_GREEN, font_s),
        (f"  Hidden2 → {HIDDEN_DIM}  neurons  (tanh)", COL_GREEN, font_s),
        (f"  Output  → {OUTPUT_DIM}   neurons  (tanh, dx/dy)", COL_BLUE, font_s),
        ("", TEXT_LO, font_s),
        (f"  Total params : {net.param_count()}", COL_AMBER, font_s),
        (f"  Evolution     : clone + crossover + mutate", TEXT_LO, font_s),
        ("  [A] to close", TEXT_LO, font_s),
    ]
    ty = 12
    for text, col, fnt in lines:
        card.blit(fnt.render(text, True, col), (12, ty))
        ty += 14
    surf.blit(card, (ox, oy))


def draw_panel(surf, state, fonts, best_h, avg_h):
    fn, fs, ft = fonts
    px = SIM_W + 12
    pygame.draw.rect(surf, PANEL_BG, (SIM_W, 0, PANEL_W, H))
    pygame.draw.line(surf, PANEL_LINE, (SIM_W, 0), (SIM_W, H), 1)

    y = 16
    surf.blit(ft.render("EvoSnake", True, TEXT_HI), (px, y)); y += 8
    surf.blit(fs.render("PyTorch edition", True, COL_AMBER), (px, y + 16)); y += 34
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 20, y), 1); y += 10

    def row(label, val, col=TEXT_HI):
        nonlocal y
        surf.blit(fs.render(label, True, TEXT_LO), (px, y))
        surf.blit(fn.render(str(val), True, col), (px + 108, y))
        y += 20

    row("Generation",  state["gen"])
    row("Step",        f"{state['step']} / {STEPS_PER_GEN}")
    row("Alive",       f"{state['alive']} / {state['pop']}")
    row("Food left",   state["food"])
    row("Best fit",    state["best"],  col=COL_GREEN)
    row("Avg fit",     f"{state['avg']:.1f}")
    row("Params/agent", state["params"], col=COL_AMBER)
    y += 4
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 20, y), 1); y += 10

    status     = "⏸  Paused" if state["paused"] else "▶  Running"
    status_col = COL_AMBER if state["paused"] else COL_GREEN
    surf.blit(fs.render(status, True, status_col), (px, y)); y += 22

    row("Speed",    f"{state['speed']} fps")
    row("Mutation", MUT_LABELS[state["mut_idx"]])
    y += 6
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 20, y), 1); y += 10

    surf.blit(fs.render("Fitness history", True, TEXT_LO), (px, y)); y += 16
    chart_h = 80
    draw_chart(surf, px, y, PANEL_W - 20, chart_h, best_h, avg_h, fs)
    y += chart_h + 26
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 20, y), 1); y += 10

    for line in ["SPACE  Pause/Resume",
                 "R      Reset",
                 "N      Next gen",
                 "↑↓     Speed",
                 "M      Mutation",
                 "A      Architecture",
                 "D      Debug labels",
                 "ESC/Q  Quit"]:
        surf.blit(fs.render(line, True, TEXT_LO), (px, y)); y += 16


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("EvoSnake — PyTorch Neuroevolution")
    clock = pygame.time.Clock()

    font_n = pygame.font.SysFont("segoeui", 14)
    font_s = pygame.font.SysFont("segoeui", 12)
    font_t = pygame.font.SysFont("segoeui", 20, bold=True)
    fonts  = (font_n, font_s, font_t)

    # ── Initial population ──────────────────────────────
    agents    = [Agent() for _ in range(POP_SIZE)]
    foods     = spawn_food(FOOD_COUNT)
    gen       = 1
    step      = 0
    paused    = False
    debug     = False
    show_arch = False
    speed     = 30
    mut_idx   = 1
    best_hist: list[float] = []
    avg_hist:  list[float] = []

    sim_surf = pygame.Surface((SIM_W, SIM_H))
    ref_net  = agents[0].net   # for param count display

    running = True
    while running:
        clock.tick(FPS_CAP)

        # ── Events ────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    agents    = [Agent() for _ in range(POP_SIZE)]
                    foods     = spawn_food(FOOD_COUNT)
                    ref_net   = agents[0].net
                    gen, step = 1, 0
                    best_hist.clear(); avg_hist.clear()
                elif k == pygame.K_n:
                    fits = [a.fitness for a in agents]
                    best_hist.append(max(fits, default=0))
                    avg_hist.append(sum(fits) / max(len(fits), 1))
                    if len(best_hist) > 50: best_hist.pop(0); avg_hist.pop(0)
                    agents = evolve(agents, MUT_RATES[mut_idx], POP_SIZE)
                    foods  = spawn_food(FOOD_COUNT)
                    gen += 1; step = 0
                elif k == pygame.K_UP:
                    speed = min(120, speed + 5)
                elif k == pygame.K_DOWN:
                    speed = max(5, speed - 5)
                elif k == pygame.K_m:
                    mut_idx = (mut_idx + 1) % len(MUT_RATES)
                elif k == pygame.K_d:
                    debug = not debug
                elif k == pygame.K_a:
                    show_arch = not show_arch

        # ── Simulation ticks ──────────────────────────
        if not paused:
            ticks = max(1, speed // 30)
            for _ in range(ticks):
                if step >= STEPS_PER_GEN:
                    fits = [a.fitness for a in agents]
                    best_hist.append(max(fits, default=0))
                    avg_hist.append(sum(fits) / max(len(fits), 1))
                    if len(best_hist) > 50: best_hist.pop(0); avg_hist.pop(0)
                    agents = evolve(agents, MUT_RATES[mut_idx], POP_SIZE)
                    foods  = spawn_food(FOOD_COUNT)
                    gen += 1; step = 0

                to_replace = []
                for a in agents:
                    eaten = a.step(foods)
                    if eaten is not None:
                        to_replace.append(eaten)

                for f in to_replace:
                    if f in foods:
                        foods.remove(f)
                    foods.append([random.uniform(15, SIM_W-15),
                                  random.uniform(15, SIM_H-15)])

                if len(foods) < FOOD_COUNT:
                    foods += spawn_food(FOOD_COUNT - len(foods))

                step += 1

        # ── Render simulation canvas ──────────────────
        sim_surf.fill(BG)
        draw_grid(sim_surf)
        draw_food(sim_surf, foods)

        ranked  = sorted(agents, key=lambda a: a.fitness, reverse=True)
        n_elite = max(1, int(len(ranked) * ELITE_FRAC))
        elite_s = {id(a) for a in ranked[:n_elite]}

        for a in agents:
            draw_agent(sim_surf, a, id(a) in elite_s, debug)

        if show_arch:
            draw_arch_overlay(sim_surf, ref_net, font_s, font_n)

        # ── Compose final frame ───────────────────────
        screen.fill(BG)
        screen.blit(sim_surf, (0, 0))

        fits  = [a.fitness for a in agents]
        state = {
            "gen":    gen,
            "step":   step,
            "alive":  sum(1 for a in agents if a.alive),
            "pop":    POP_SIZE,
            "food":   len(foods),
            "best":   max(fits, default=0),
            "avg":    sum(fits) / max(len(fits), 1),
            "paused": paused,
            "speed":  speed,
            "mut_idx": mut_idx,
            "params": ref_net.param_count(),
        }
        draw_panel(screen, state, fonts, best_hist, avg_hist)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
    print(f"SnakeNet params per agent: {SnakeNet().param_count()}")
    main()