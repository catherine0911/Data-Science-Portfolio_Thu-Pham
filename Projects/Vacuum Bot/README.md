# Magic Sweeper – Intelligent Vacuum Bot (Python)

Magic Sweeper is an autonomous vacuum-cleaning robot designed for an AI module at Tilburg University.  
The bot navigates a grid world using only local vision and internal memory, without access to the full map.  
This project received a grade of **8/10**.

---

## Overview

Magic Sweeper receives limited information each turn:

- `currentCell`: current robot position  
- `currentEnergy`: remaining energy  
- `vision`: a 3×3 grid around the robot  
- `remainingStainCells`: how many stains are left  

Using this, it must clean the entire map while avoiding obstacles (`x`) and walls.

---

## Key Features

### 1. Zig-zag sweeping pattern
- Moves top → bottom, left → right.
- Skips 2 rows (except the first) to save energy.
- Only fully checks rows where `row % 3 == 2`.

### 2. Vertical cleaning logic
- If a stain appears in the top or bottom cell of the vision grid, the bot moves up/down to clean it.
- Before cleaning, it stores its position in `resumeCell`.
- After cleaning, it returns to that saved cell and continues its zig-zag route.

### 3. Obstacle avoidance
- Detects pillars (`x`) and ignores the map boundary walls.
- When facing a pillar, the bot:
  - Stores its current row in `resumeRow`
  - Moves downward until the pillar is no longer blocking its path
  - Returns to the correct row and continues sweeping from the column after the pillar

### 4. Edge-aware movement
- Handles movement near left and right walls without getting stuck.
- Adjusts direction only when actually encountering an obstacle.

### 5. Completion check
- When `remainingStainCells == 0`, the bot stops.

---

## Performance Summary

Magic Sweeper was tested on several difficulty levels:

| Map Grade | Result |
|----------|--------|
| 6        | Cleaned successfully |
| 7        | Cleaned successfully |
| 8        | Solved 3/4 maps (struggles with very large pillars) |
| 9        | Too complex |
| 10       | Too complex |

It performs well on medium-complexity maps and demonstrates strong rule-based logic.

---

## Skills Demonstrated

- Python programming  
- Rule-based agent design  
- Handling partial observability  
- Obstacle detection and avoidance  
- Memory implementation in agents  
- Debugging multi-step movement logic  
- Energy-efficient navigation  
---

## Potential Improvements

Future work could include:

- A* search for advanced obstacle avoidance  
- Better handling of large clusters of pillars  
- Path recovery logic when trapped  
- Local map reconstruction from visited cells  
- More energy-efficient planning  
