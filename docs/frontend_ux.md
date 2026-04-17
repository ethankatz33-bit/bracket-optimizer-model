# Frontend UX Plan — March Madness Bracket Predictor

**Core value proposition:** "Stop picking your heart. Start winning your pool."

---

## The Problem We're Solving

Most bracket tools optimize for accuracy — picking the most likely outcome.
That's the wrong goal for pool play. A bracket that matches 60 million other
brackets wins you nothing even if it's "correct."

This tool solves a different problem: **maximize your expected payout**, which
requires differentiation from the field, not just good picks.

---

## User Flow

```
Pool size input
     ↓
Team data input (CSV upload or manual bracket)
     ↓
Strategy recommendation (single vs portfolio, bold vs safe)
     ↓
Generated bracket(s) with champion rationale
     ↓
Export (printable / ESPN-importable)
```

---

## Page Structure

### 1. Homepage / Landing

**Headline:** "Win your March Madness pool — not just predict it"

**Sub-headline:**
> Most people pick the popular team. We find the team that wins you money.

**Three-card hook:**
- Card 1 → "For office pools (≤50 people): best single bracket, champion by win probability"
- Card 2 → "For medium pools (50–500): value-weighted champion, semi-contrarian path"
- Card 3 → "For large pools (500+): portfolio of 3–8 brackets, each chasing a different champion"

**CTA button:** "Build my bracket →"

---

### 2. Pool Setup (Step 1 of 3)

**Input fields:**
- Pool size (numeric, required) — with labels: "Office pool", "ESPN group", "Large bracket challenge"
- Pool type (radio): Single entry / Multiple entries allowed
- Scoring format (radio): Standard (round multiplier) / Upset bonus / ESPN-style points

**Pool size thresholds + strategy copy that updates live:**

| Pool size | Strategy label | Shown copy |
|-----------|---------------|------------|
| 1–20 | Safe | "Small pool: pick the likely winner. Contrarian picks are risky here." |
| 21–200 | Balanced | "Medium pool: mix probability and value. One differentiated champion pick." |
| 201–1000 | Value-first | "Large pool: champion pick is everything. Avoid the public's #1 favorite." |
| 1000+ | Portfolio | "Mega pool: you need multiple brackets with different champion picks to have any shot." |

---

### 3. Team Data Input (Step 2 of 3)

**Two input modes:**

**Mode A — CSV Upload**
- Drag-and-drop area: "Drop your KenPom/Barttorvik export here"
- Column mapping: auto-detect `team_name`, `efficiency_margin`, `seed`, etc.
- Inline validation: highlight missing columns before proceeding
- Link to schema reference

**Mode B — Use Our Defaults**
- "We'll use the current bracket seedings with historical efficiency estimates"
- User can edit seed assignments for known matchups

**Region assignment (if not in CSV):**
- Four-column visual with seed slots (East/West/South/Midwest)
- Drag teams to assign regions, or auto-fill from CSV

---

### 4. Bracket Output (Step 3 of 3)

#### Single Bracket View

**Left panel — Bracket visual:**
- Standard left-right bracket layout
- Winners highlighted in green
- Upsets flagged with a small badge (e.g. "11-seed upset")
- Champion at center with large callout

**Right panel — Strategy card:**
```
YOUR CHAMPION PICK
─────────────────────────────────
  Kansas  |  #1 West
  Win probability:   31.4%
  Public pick share: 18.2%
  Value score:       1.7×

  Rationale:
  Top efficiency margin in the field (+32.6), elite defense,
  AP Top 12 all season. At 18% public pick share, this pick
  has positive expected value in most pool sizes.
─────────────────────────────────
FINAL FOUR
  Kansas (#1 West)     ← champion path
  Duke   (#1 East)
  Auburn (#1 South)
  Tennessee (#2 South) ← contrarian

NOTABLE UPSETS PICKED
  R64: #11 NC State over #6 Xavier
  E8:  #2 Gonzaga over #1 Kansas  ← other half
```

#### Portfolio View (when n > 1)

**Table at top:**
```
 #   Champion           Seed  Region   Win%   Public%  Value   Composite
 1   NC State            11   East     3.1%    0.1%    31.0×   0.248
 2   Kansas               1   West    31.4%   18.2%    1.7×    0.142
 3   Auburn               1   South   28.6%   20.1%    1.4×    0.131
...
```

**Expandable row:** click any bracket to see its full tree.

**EV callout box:**
> "In a 10,000-person pool, if NC State wins, ~10 people share the pot vs ~100 for a 1-seed pick.
> That's 10× the payout for a pick with 1/5th the chance — positive expected value at scale."

---

### 5. Export / Share

- **Print bracket** — styled PDF with champion highlighted
- **Copy bracket picks** — text list for manual ESPN entry
- **JSON export** — for API/developer use
- **Share link** — save bracket state to URL params

---

## Recommendations by Pool Type

### Small pool (≤20 people)

**Recommendation:** Single bracket, pick the highest win-probability champion.

> In a small pool, everyone has a real shot at the pot. Avoid contrarian picks —
> if you pick an 11-seed and they don't win, you gave up a bracket that would
> have won you money.

Champion strategy: pick the #1 seed with the best efficiency margin.

---

### Medium pool (21–200 people)

**Recommendation:** Single bracket, one differentiated champion pick.

> The pot is big enough that you need to stand out, but not so crowded that
> you need multiple brackets. Pick a 2-seed or undervalued 1-seed with
> positive value score (>1.5×).

Champion strategy: value_score > 1.5, seed ≤ 3.

---

### Large pool (201–1000 people)

**Recommendation:** 2–3 brackets with different champions.

> At this scale, picking the same champion as 200 other people is near-worthless
> even if they win. You need at least one contrarian champion pick — a 2-seed
> or lower — in your portfolio.

Champion strategy: at least one pick with value_score > 2.5.

---

### Mega pool (1000+ people)

**Recommendation:** 5–10 brackets spanning 4–6 different champion picks.

> The only winning strategy at this scale is to hold a "position" in multiple
> likely-but-underrepresented outcomes. You can't outguess 10,000 people on
> game-level picks — but you can own champion picks that 1–3% of the pool
> chose.

Champion strategy: include at least one seed 5–12 pick. The path-win
probability is low, but the payout multiplier is massive.

---

## Key UX Principles

1. **Lead with pool size, not bracket picks.** The first question is "how many people are you playing against?" — everything flows from that.

2. **Explain the "why" for every champion pick.** Show win probability, public pick share, and value score together. Users need to feel confident defending their contrarian pick.

3. **The word "upset" is bad UX.** Reframe as "value pick" or "differentiated pick" — it makes users more likely to follow the recommendation.

4. **Show the cost of consensus.** On the champion card, show how many people (estimated) are picking the same team. "~1,800 of 10,000 people in your pool are also picking Kansas — you'd split the pot with them."

5. **Mobile-first bracket display.** Most users view brackets on their phone. Use a vertical scrolling tournament tree, not the traditional left-right layout.
