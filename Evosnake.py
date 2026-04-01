"""
EvoSnake — Neuroevolution Artificial Life Simulation
=====================================================
Requirements:  pip install pygame numpy

Controls:
  SPACE      Pause / Resume
  R          Reset simulation
  N          Force next generation
  UP / DOWN  Change simulation speed
  M          Toggle mutation rate (low / med / high)
  D          Toggle debug overlay
  ESC / Q    Quit
"""

import sys
import math
import random
import numpy as np
import pygame

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
W, H          = 900, 620      # window size
SIM_W, SIM_H  = 620, 620      # simulation canvas (left panel)
PANEL_W       = W - SIM_W     # right info panel

POP_SIZE      = 20
FOOD_COUNT    = 20
STEPS_PER_GEN = 1000
AGENT_SPEED   = 2.8
ENERGY_DRAIN  = 0.001
ENERGY_FOOD   = 0.40
EAT_RADIUS    = 10
ELITE_FRAC    = 0.10
MUTATION_RATE = 0.08
TRAIL_LEN     = 24
FPS_CAP       = 120

# Neural net shape: 7 inputs → 12 hidden → 2 outputs
NN_SHAPE = (11,256, 256,256,256, 2)

# Colour palette
BG          = (10,  14,  22)
GRID_COL    = (255, 255, 255,  8)
FOOD_OUTER  = (245, 158,  11)
FOOD_INNER  = (252, 211,  77)
ELITE_COL   = ( 74, 222, 128)
NORMAL_COL  = ( 96, 165, 250)
DEAD_COL    = ( 60,  60,  80)
VEC_COL     = (239,  68,  68)
PANEL_BG    = ( 18,  22,  32)
PANEL_LINE  = ( 40,  50,  70)
TEXT_HI     = (220, 220, 235)
TEXT_LO     = (120, 130, 155)
GREEN_CHART = ( 74, 222, 128)
BLUE_CHART  = ( 96, 165, 250)

MUT_PRESETS = [0.03, 0.08, 0.20]
MUT_LABELS  = ["Low (0.03)", "Med (0.08)", "High (0.20)"]


# ─────────────────────────────────────────────
# NEURAL NETWORK  (pure numpy, no torch needed)
# ─────────────────────────────────────────────
def nn_weights_size(shape):
    n = 0
    for i in range(len(shape) - 1):
        n += shape[i] * shape[i+1] + shape[i+1]
    return n

def random_weights(shape):
    return np.random.randn(nn_weights_size(shape)).astype(np.float32) * 0.8

def mutate_weights(w, rate):
    noise = np.random.randn(*w.shape).astype(np.float32)
    mask  = np.random.random(w.shape) < 0.35
    return w + noise * rate * mask

def nn_forward(flat_w, inp, shape):
    x = np.array(inp, dtype=np.float32)
    idx = 0
    for i in range(len(shape) - 1):
        r, c = shape[i+1], shape[i]
        W_ = flat_w[idx : idx + r*c].reshape(r, c); idx += r*c
        b_ = flat_w[idx : idx + r];                  idx += r
        x  = np.tanh(W_ @ x + b_)
    return x


# ─────────────────────────────────────────────
# FOOD
# ─────────────────────────────────────────────
def spawn_food(count):
    margin = 15
    return [
        [random.uniform(margin, SIM_W - margin),
         random.uniform(margin, SIM_H - margin)]
        for _ in range(count)
    ]


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────
class Agent:
    def __init__(self, weights=None):
        self.x  = random.uniform(30, SIM_W - 30)
        self.y  = random.uniform(30, SIM_H - 30)
        ang     = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(ang) * AGENT_SPEED
        self.vy = math.sin(ang) * AGENT_SPEED
        self.energy  = 1.0
        self.fitness = 0
        self.alive   = True
        self.trail   = []
        self.weights = weights if weights is not None else random_weights(NN_SHAPE)

    def nearest_food(self, foods):
        best_d, best_f = math.inf, None
        for f in foods:
            d = math.hypot(f[0] - self.x, f[1] - self.y)
            if d < best_d:
                best_d, best_f = d, f
        return best_f, best_d

    def step(self, foods):
        if not self.alive:
            return None

        food, dist = self.nearest_food(foods)
        if food is None:
            return None

        fx, fy  = food
        inv_d   = 1.0 / max(dist, 1.0)
        dx_norm = (fx - self.x) * inv_d
        dy_norm = (fy - self.y) * inv_d
        spd_inv = 1.0 / AGENT_SPEED
        dist_left   = self.x / SIM_W
        dist_right  = (SIM_W - self.x) / SIM_W
        dist_top    = self.y / SIM_H
        dist_bottom = (SIM_H - self.y) / SIM_H

        inp = [
            dx_norm,
            dy_norm,
            min(dist / 400.0, 1.0),
            self.vx * spd_inv,
            self.vy * spd_inv,
            self.energy,
            dist_left,
            dist_right,
            dist_top,
            dist_bottom,
            1.0 if dist < 60 else 0.0,  # "food close" signal
        ]

        out = nn_forward(self.weights, inp, NN_SHAPE)
        # Blend neural steering with a soft food-seeking bias
        #blend = 0.55
        #tx = out[0] * (1 - blend) + dx_norm * blend
        #ty = out[1] * (1 - blend) + dy_norm * blend
        tx, ty = out[0], out[1]
        # Smooth velocity update
        self.vx = self.vx * 0.6 + tx * AGENT_SPEED * 0.4
        self.vy = self.vy * 0.6 + ty * AGENT_SPEED * 0.4
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
            return food

        return None


# ─────────────────────────────────────────────
# EVOLUTION ENGINE
# ─────────────────────────────────────────────
def evolve(agents, mut_rate, pop_size):
    ranked  = sorted(agents, key=lambda a: a.fitness, reverse=True)
    n_elite = max(2, int(len(ranked) * ELITE_FRAC))
    elite   = ranked[:n_elite]

    new_agents = []
    # Keep top 2 unchanged
    for e in elite[:2]:
        new_agents.append(Agent(weights=e.weights.copy()))
    # Fill rest with mutated elite offspring
    while len(new_agents) < pop_size:
        parent = random.choice(elite)
        child_w = mutate_weights(parent.weights, mut_rate)
        new_agents.append(Agent(weights=child_w))

    return new_agents


# ─────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────
def draw_grid(surf):
    grid_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    for x in range(0, SIM_W, 30):
        pygame.draw.line(grid_surf, (255, 255, 255, 8), (x, 0), (x, SIM_H))
    for y in range(0, SIM_H, 30):
        pygame.draw.line(grid_surf, (255, 255, 255, 8), (0, y), (SIM_W, y))
    surf.blit(grid_surf, (0, 0))


def draw_food(surf, foods):
    for f in foods:
        x, y = int(f[0]), int(f[1])
        pygame.draw.circle(surf, FOOD_OUTER, (x, y), 6)
        pygame.draw.circle(surf, FOOD_INNER, (x, y), 3)


def draw_agent(surf, agent, is_elite, debug):
    col = ELITE_COL if is_elite else NORMAL_COL
    if not agent.alive:
        col = DEAD_COL

    ax, ay = int(agent.x), int(agent.y)

    # Trail
    if len(agent.trail) > 1:
        trail_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        n = len(agent.trail)
        for i in range(1, n):
            alpha = int((i / n) * (80 if is_elite else 55))
            r, g, b = col
            pygame.draw.line(trail_surf, (r, g, b, alpha),
                             (int(agent.trail[i-1][0]), int(agent.trail[i-1][1])),
                             (int(agent.trail[i][0]),   int(agent.trail[i][1])), 2)
        surf.blit(trail_surf, (0, 0))

    # Body circle
    pygame.draw.circle(surf, col, (ax, ay), 7)
    pygame.draw.circle(surf, (0, 0, 0), (ax, ay), 7, 1)

    # Energy arc
    if agent.alive:
        arc_rect = pygame.Rect(ax - 10, ay - 10, 20, 20)
        angle    = agent.energy * 2 * math.pi
        if angle > 0.05:
            r, g, b = col
            arc_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
            pygame.draw.arc(arc_surf, (r, g, b, 160),
                            arc_rect, math.pi / 2 - angle, math.pi / 2, 2)
            surf.blit(arc_surf, (0, 0))

    # Direction vector
    if agent.alive:
        spd = math.hypot(agent.vx, agent.vy) or 1.0
        ex  = ax + int((agent.vx / spd) * 14)
        ey  = ay + int((agent.vy / spd) * 14)
        pygame.draw.line(surf, VEC_COL, (ax, ay), (ex, ey), 1)

    # Fitness label (debug)
    if debug and agent.fitness > 0:
        font_tiny = pygame.font.SysFont("monospace", 10)
        lbl = font_tiny.render(str(agent.fitness), True, (200, 200, 200))
        surf.blit(lbl, (ax + 9, ay - 8))


def draw_chart(surf, x, y, w, h, best_hist, avg_hist, font_s):
    pygame.draw.rect(surf, (25, 30, 45), (x, y, w, h), border_radius=4)
    if len(best_hist) < 2:
        return
    mx = max(max(best_hist), 1)

    def pts(data, col):
        points = []
        for i, v in enumerate(data):
            px = x + int(i / (len(data) - 1) * (w - 8)) + 4
            py = y + h - 4 - int((v / mx) * (h - 10))
            points.append((px, py))
        if len(points) > 1:
            pygame.draw.lines(surf, col, False, points, 2)

    pts(best_hist, GREEN_CHART)
    pts(avg_hist,  BLUE_CHART)

    # Legend
    pygame.draw.line(surf, GREEN_CHART, (x + 4,  y + h + 6), (x + 16, y + h + 6), 2)
    surf.blit(font_s.render("best", True, TEXT_LO), (x + 19, y + h + 1))
    pygame.draw.line(surf, BLUE_CHART,  (x + 50, y + h + 6), (x + 62, y + h + 6), 2)
    surf.blit(font_s.render("avg",  True, TEXT_LO), (x + 65, y + h + 1))


def draw_panel(surf, state, fonts):
    fn, fs, ft = fonts  # normal, small, title
    px = SIM_W + 12
    pygame.draw.rect(surf, PANEL_BG, (SIM_W, 0, PANEL_W, H))
    pygame.draw.line(surf, PANEL_LINE, (SIM_W, 0), (SIM_W, H), 1)

    y = 18
    # Title
    surf.blit(ft.render("EvoSnake", True, TEXT_HI), (px, y)); y += 30
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 24, y), 1); y += 10

    def row(label, val, highlight=False):
        nonlocal y
        surf.blit(fs.render(label, True, TEXT_LO), (px, y))
        vcol = ELITE_COL if highlight else TEXT_HI
        surf.blit(fn.render(str(val), True, vcol), (px + 100, y))
        y += 20

    row("Generation",  state["gen"])
    row("Step",        f"{state['step']} / {STEPS_PER_GEN}")
    row("Alive",       f"{state['alive']} / {state['pop_size']}")
    row("Food",        state["food"])
    row("Best fit",    state["best"],  highlight=True)
    row("Avg fit",     f"{state['avg']:.1f}")
    y += 4
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 24, y), 1); y += 10

    # Status tag
    status     = "⏸ Paused" if state["paused"] else "▶ Running"
    status_col = (245, 158, 11) if state["paused"] else (74, 222, 128)
    surf.blit(fs.render(status, True, status_col), (px, y)); y += 22

    row("Speed",    f"{state['speed']} fps")
    row("Mutation", MUT_LABELS[state["mut_idx"]])
    y += 8
    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 24, y), 1); y += 10

    # Fitness chart
    surf.blit(fs.render("Fitness history", True, TEXT_LO), (px, y)); y += 16
    chart_h = 80
    draw_chart(surf, px, y, PANEL_W - 24, chart_h,
               state["best_hist"], state["avg_hist"], fs)
    y += chart_h + 26

    pygame.draw.line(surf, PANEL_LINE, (px, y), (px + PANEL_W - 24, y), 1); y += 10

    # Controls help
    for line in ["SPACE  Pause/Resume",
                 "R      Reset",
                 "N      Next gen",
                 "↑↓     Speed",
                 "M      Mutation",
                 "D      Debug",
                 "ESC/Q  Quit"]:
        surf.blit(fs.render(line, True, TEXT_LO), (px, y)); y += 16


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("EvoSnake — Neuroevolution Simulation")
    clock = pygame.time.Clock()

    font_n = pygame.font.SysFont("segoeui",    14)
    font_s = pygame.font.SysFont("segoeui",    12)
    font_t = pygame.font.SysFont("segoeui",    20, bold=True)
    fonts  = (font_n, font_s, font_t)

    # Simulation state
    agents     = [Agent() for _ in range(POP_SIZE)]
    foods      = spawn_food(FOOD_COUNT)
    gen        = 1
    step       = 0
    paused     = False
    debug      = False
    speed      = 30
    mut_idx    = 1
    best_hist  = []
    avg_hist   = []

    sim_surf = pygame.Surface((SIM_W, SIM_H))

    running = True
    while running:
        clock.tick(FPS_CAP)

        # ── Events ──────────────────────────────
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
                    gen       = 1
                    step      = 0
                    best_hist = []
                    avg_hist  = []
                elif k == pygame.K_n:
                    # Force next generation
                    fits      = [a.fitness for a in agents]
                    best_hist.append(max(fits, default=0))
                    avg_hist.append(sum(fits) / max(len(fits), 1))
                    if len(best_hist) > 50: best_hist.pop(0); avg_hist.pop(0)
                    agents = evolve(agents, MUT_PRESETS[mut_idx], POP_SIZE)
                    foods  = spawn_food(FOOD_COUNT)
                    gen   += 1
                    step   = 0
                elif k == pygame.K_UP:
                    speed = min(120, speed + 5)
                elif k == pygame.K_DOWN:
                    speed = max(5, speed - 5)
                elif k == pygame.K_m:
                    mut_idx = (mut_idx + 1) % len(MUT_PRESETS)
                elif k == pygame.K_d:
                    debug = not debug

        # ── Simulation steps ────────────────────
        if not paused:
            steps_this_frame = max(1, speed // 30)
            for _ in range(steps_this_frame):
                if step >= STEPS_PER_GEN:
                    fits      = [a.fitness for a in agents]
                    best_hist.append(max(fits, default=0))
                    avg_hist.append(sum(fits) / max(len(fits), 1))
                    if len(best_hist) > 50: best_hist.pop(0); avg_hist.pop(0)
                    agents = evolve(agents, MUT_PRESETS[mut_idx], POP_SIZE)
                    foods  = spawn_food(FOOD_COUNT)
                    gen   += 1
                    step   = 0

                eaten_foods = set()
                for a in agents:
                    f = a.step(foods)
                    if f is not None:
                        idx = foods.index(f)
                        eaten_foods.add(idx)

                for idx in sorted(eaten_foods, reverse=True):
                    foods.pop(idx)
                    foods.append([
                        random.uniform(15, SIM_W - 15),
                        random.uniform(15, SIM_H - 15)
                    ])

                if len(foods) < FOOD_COUNT:
                    foods += spawn_food(FOOD_COUNT - len(foods))

                step += 1

        # ── Rendering ───────────────────────────
        sim_surf.fill(BG)
        draw_grid(sim_surf)
        draw_food(sim_surf, foods)

        # Determine elite set
        ranked  = sorted(agents, key=lambda a: a.fitness, reverse=True)
        n_elite = max(1, int(len(ranked) * ELITE_FRAC))
        elite_s = set(id(a) for a in ranked[:n_elite])

        for a in agents:
            draw_agent(sim_surf, a, id(a) in elite_s, debug)

        screen.fill(BG)
        screen.blit(sim_surf, (0, 0))

        fits = [a.fitness for a in agents]
        state = {
            "gen":       gen,
            "step":      step,
            "alive":     sum(1 for a in agents if a.alive),
            "pop_size":  POP_SIZE,
            "food":      len(foods),
            "best":      max(fits, default=0),
            "avg":       sum(fits) / max(len(fits), 1),
            "paused":    paused,
            "speed":     speed,
            "mut_idx":   mut_idx,
            "best_hist": best_hist,
            "avg_hist":  avg_hist,
        }
        draw_panel(screen, state, fonts)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()