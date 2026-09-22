import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from ortools.sat.python import cp_model

def print_progress(current, total, prefix='Progress'):
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(50 * current // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% Complete')
    sys.stdout.flush()
    if current == total: sys.stdout.write('\n')

# --- 1. DATA EXTRACTION ---
def get_map_data(img_path):
    print("Step 1: Analyzing Map...")
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError("Could not find dots.png")
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    colors = {
        'red':    ((0, 150, 150),   (10, 255, 255)),
        'yellow': ((20, 150, 150),  (30, 255, 255)),
        'white':  ((0, 0, 200),     (180, 30, 255)),
        'green':  ((40, 150, 150),  (70, 255, 255)),
        'gray':   ((0, 0, 40),      (180, 50, 150))
    }

    nodes, cats = [], {'red': [], 'yellow': [], 'white': [], 'green': []}
    for color in cats:
        mask = cv2.inRange(hsv, colors[color][0], colors[color][1])
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                nodes.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
                cats[color].append(len(nodes) - 1)
                
    obs_mask = cv2.inRange(hsv, colors['gray'][0], colors['gray'][1])
    print(f"Found {len(nodes)} nodes. Processing distance matrix...")
    return np.array(nodes), cats, (obs_mask > 0), (h, w)

# --- 2. SOLVER LOGIC ---
def solve_with_rules(nodes, cats, size):
    model = cp_model.CpModel()
    n = len(nodes)
    red, green, whites = cats['red'][0], cats['green'][0], set(cats['white'])
    
    x = {(i, j): model.NewBoolVar(f'x_{i}_{j}') for i in range(n) for j in range(n) if i != j}
    is_tp = {(i, j): model.NewBoolVar(f't_{i}_{j}') for i in range(n) for j in range(n) if i != j}
    u = [model.NewIntVar(0, n - 1, f'u_{i}') for i in range(n)]

    # 1. Flow Constraints
    for i in range(n):
        model.Add(sum(x[i, j] for j in range(n) if i != j) == 1)
        model.Add(sum(x[j, i] for j in range(n) if i != j) == 1)

    # 2. MTZ Order (Visitation)
    model.Add(u[red] == 0)
    for (i, j), var in x.items():
        if j != red:
            model.Add(u[j] >= u[i] + 1).OnlyEnforceIf(var)

    # 3. Exit Condition: All nodes visited before Green
    for i in range(n):
        if i != green: model.Add(u[green] > u[i])

    # 4. Teleport Logic: Destination j must be visited before current i
    for (i, j), var in x.items():
        if j in whites:
            model.Add(u[j] < u[i]).OnlyEnforceIf(is_tp[i, j])
            model.Add(is_tp[i, j] <= var)
        else:
            model.Add(is_tp[i, j] == 0)

    # 5. Objective: Sum of physical distances
    obj = []
    total_edges = len(x)
    for idx, ((i, j), var) in enumerate(x.items()):
        dist = int(np.linalg.norm(nodes[i] - nodes[j]))
        cost_var = model.NewIntVar(0, dist, f'c_{i}_{j}')
        
        model.Add(cost_var == 0).OnlyEnforceIf(is_tp[i, j])
        model.Add(cost_var == dist).OnlyEnforceIf([is_tp[i, j].Not(), var])
        model.Add(cost_var == 0).OnlyEnforceIf(var.Not())
        obj.append(cost_var)
        if idx % 10 == 0: print_progress(idx, total_edges, prefix='Solving Geometry')
            
    print_progress(total_edges, total_edges, prefix='Solving Geometry')
    model.Minimize(sum(obj))
    
    print("\nStep 3: Finding optimal route (CP-SAT)...")
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        res = sorted(range(n), key=lambda i: solver.Value(u[i]))
        path_data = []
        for k in range(len(res)-1):
            u_f, v_t = res[k], res[k+1]
            path_data.append(((u_f, v_t), bool(solver.Value(is_tp[u_f, v_t]))))
        return path_data, solver.ObjectiveValue()
    return None, 0

# --- 3. RENDERING ---
def render(nodes, path_data, cats, total_dist, size):
    h, w = size
    # Create figure with exact pixel sizing
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    total_steps = len(path_data)
    for idx, ((i, j), tp_active) in enumerate(path_data):
        p1, p2 = nodes[i], nodes[j]
        color = 'white' if tp_active else 'cyan'
        style = '--' if tp_active else '-'
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, ls=style, lw=2, zorder=1)
        ax.text(p2[0], p2[1], str(idx+1), color='yellow', weight='bold', fontsize=9,
                ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, lw=0))
        print_progress(idx + 1, total_steps, prefix='Rendering Path ')

    plt.text(w*0.02, h*0.04, f"TOTAL DISTANCE: {total_dist}", color='cyan', fontsize=12, weight='bold')
    plt.savefig('route.png', transparent=True, dpi=100, pad_inches=0)
    plt.close()

# --- EXECUTION ---
try:
    nodes, cats, obs, size = get_map_data('dots.png')
    path, total_dist = solve_with_rules(nodes, cats, size)
    if path:
        render(nodes, path, cats, total_dist, size)
        print(f"\nSuccess! Dimensions: {size[1]}x{size[0]}. Saved to route.png")
    else:
        print("\nSolver failed to find a valid route.")
except Exception as e:
    print(f"\nAn error occurred: {e}")