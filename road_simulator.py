# ============================================
# INTERACTIVE ROAD NETWORK SECURITY SIMULATOR
# MTech Demo - Real-time Attackers & Path Finding
# ============================================

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np
import random
import time
from collections import defaultdict
from enum import Enum
import pickle
import os

# ============ CONFIGURATION ============
DATASET_URL = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
TXT_FILENAME = "roadNet-CA.txt"
SUBGRAPH_CENTER = 300
SUBGRAPH_RADIUS = 8

# ============ ENERGY MODEL CONSTANTS ============
# Based on standard VANET energy models in literature
# (e.g., Elhoseny & Shankar, 2020; Husnain et al., 2023)
E_TRANSMIT_PER_HOP   = 2.0      # Base transmission energy per hop (units)
E_RECEIVE_PER_HOP    = 0.5      # Receive energy per hop
E_ROUTE_DISCOVERY    = 4.5      # Standard AODV RREQ/RREP flooding overhead
                                 # (broadcast to all neighbours, no filtering)
E_AUTH_OVERHEAD      = 0.6      # Proposed: lightweight mutual auth per path
                                 # (cheaper than flooding — unicast only)
E_RETRANSMIT         = 5.0      # Retransmission cost when packet is dropped
PROPAGATION_DELAY    = 0.05     # Seconds per hop (50ms wireless link)
PROCESSING_DELAY     = 0.02     # Seconds per hop (processing at each node)
ROUTE_DISCOVERY_DELAY= 0.20     # Standard RREQ/RREP round-trip delay
AUTH_DELAY           = 0.08     # Proposed: unicast auth handshake (faster)
PACKET_SIZE          = 1024     # bytes

# ============ NODE TYPES ============
class NodeType(Enum):
    RSU         = "Roadside Unit"
    VEHICLE     = "Vehicle"
    PEDESTRIAN  = "Pedestrian"
    EMERGENCY   = "Emergency"


class PerformanceMetrics:
    """
    Honest performance metrics based on real graph measurements.

    Standard routing  : uses nx.shortest_path with no attacker awareness.
                        If path passes through an attacker node, packet is
                        dropped (PDR hit) and retransmission energy is charged.

    Proposed routing  : builds attack-filtered graph G_clean, then selects
                        among four strategies. Authentication overhead is
                        added honestly. No artificial caps or floors.
    """

    def __init__(self):
        self.standard = {
            'energy': [], 'delay': [], 'throughput': [],
            'hops': [], 'packets_sent': 0, 'packets_delivered': 0
        }
        self.proposed = {
            'energy': [], 'delay': [], 'throughput': [],
            'hops': [], 'packets_sent': 0, 'packets_delivered': 0,
            'strategies_used': defaultdict(int)
        }

    # ------------------------------------------------------------------
    # STANDARD ROUTING SIMULATION
    # Uses shortest path with no attacker awareness.
    # Energy and delay are computed from actual hop count + retransmit
    # penalties if attackers are on the path.
    # ------------------------------------------------------------------
    def simulate_standard_routing(self, G, src, dest, attackers):
        try:
            path = nx.shortest_path(G, src, dest)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path at all — charge a timeout penalty
            return {
                'delivered': False,
                'energy': E_RETRANSMIT * 2,   # Two retransmit attempts
                'delay': 3 * AUTH_DELAY,       # Timeout ~ 3× normal
                'hops': 0,
                'throughput': 0
            }

        hops = len(path) - 1
        attackers_on_path = [n for n in path if n in attackers]

        # Base energy: transmit + receive + RREQ/RREP broadcast flooding
        # Standard AODV broadcasts route requests to all neighbours
        energy = hops * (E_TRANSMIT_PER_HOP + E_RECEIVE_PER_HOP)
        energy += E_ROUTE_DISCOVERY

        # Base delay: propagation + processing + RREQ/RREP round-trip
        delay = hops * (PROPAGATION_DELAY + PROCESSING_DELAY)
        delay += ROUTE_DISCOVERY_DELAY

        if attackers_on_path:
            # Packet is dropped at the first attacker node.
            # Standard routing has no mechanism to detect this beforehand,
            # so the full path energy is spent, then a retransmission is
            # attempted (and fails again if attacker is still there).
            num_retransmits = min(len(attackers_on_path), 3)
            energy += num_retransmits * E_RETRANSMIT
            delay  += num_retransmits * (hops * PROPAGATION_DELAY + 0.5)
            delivered = False
        else:
            delivered = True

        throughput = (PACKET_SIZE / delay) if delivered else 0

        return {
            'delivered': delivered,
            'energy': round(energy, 3),
            'delay': round(delay, 3),
            'hops': hops,
            'throughput': round(throughput, 2)
        }

    # ------------------------------------------------------------------
    # PROPOSED ROUTING SIMULATION
    # Path is computed on G_clean (attackers already removed).
    # Authentication overhead is added as a real cost.
    # ------------------------------------------------------------------
    def simulate_proposed_routing(self, G, src, dest, attackers,
                                  strategy_name, path):
        if not path or len(path) < 2:
            return {
                'delivered': False,
                'energy': E_AUTH_OVERHEAD,   # Auth was attempted
                'delay': AUTH_DELAY,
                'hops': 0,
                'throughput': 0,
                'strategy': strategy_name
            }

        hops = len(path) - 1

        # Base transmission energy (same physics as standard)
        energy = hops * (E_TRANSMIT_PER_HOP + E_RECEIVE_PER_HOP)

        # Auth overhead: lightweight unicast handshake
        # Cheaper than standard RREQ broadcast flooding
        energy += E_AUTH_OVERHEAD

        # Delay: propagation + processing + unicast auth
        # AUTH_DELAY < ROUTE_DISCOVERY_DELAY (unicast vs broadcast)
        delay = hops * (PROPAGATION_DELAY + PROCESSING_DELAY) + AUTH_DELAY

        # Path is guaranteed attacker-free (built on G_clean),
        # so no retransmission penalties.
        delivered = True
        throughput = PACKET_SIZE / delay

        return {
            'delivered': delivered,
            'energy': round(energy, 3),
            'delay': round(delay, 3),
            'hops': hops,
            'throughput': round(throughput, 2),
            'strategy': strategy_name
        }

    def update_metrics(self, std_result, prop_result, strategy_name):
        # Standard
        self.standard['packets_sent'] += 1
        self.standard['energy'].append(std_result['energy'])
        self.standard['delay'].append(std_result['delay'])
        self.standard['hops'].append(std_result['hops'])
        self.standard['throughput'].append(std_result['throughput'])
        if std_result['delivered']:
            self.standard['packets_delivered'] += 1

        # Proposed
        self.proposed['packets_sent'] += 1
        self.proposed['energy'].append(prop_result['energy'])
        self.proposed['delay'].append(prop_result['delay'])
        self.proposed['hops'].append(prop_result['hops'])
        self.proposed['throughput'].append(prop_result['throughput'])
        if prop_result['delivered']:
            self.proposed['packets_delivered'] += 1
            self.proposed['strategies_used'][strategy_name] += 1

    def get_summary(self):
        def safe_mean(lst):
            return round(float(np.mean(lst)), 3) if lst else 0.0

        s = self.standard
        p = self.proposed

        std_pdr  = (s['packets_delivered'] / max(1, s['packets_sent'])) * 100
        prop_pdr = (p['packets_delivered'] / max(1, p['packets_sent'])) * 100

        std_energy  = safe_mean(s['energy'])
        prop_energy = safe_mean(p['energy'])
        std_delay   = safe_mean(s['delay'])
        prop_delay  = safe_mean(p['delay'])
        std_tp      = safe_mean(s['throughput'])
        prop_tp     = safe_mean(p['throughput'])

        energy_saving = ((std_energy - prop_energy) / max(0.001, std_energy)) * 100
        delay_reduction = ((std_delay - prop_delay) / max(0.001, std_delay)) * 100

        return {
            'standard': {
                'pdr': round(std_pdr, 1),
                'avg_energy': std_energy,
                'avg_delay': std_delay,
                'avg_throughput': std_tp,
                'avg_hops': safe_mean(s['hops']),
                'packets': s['packets_delivered']
            },
            'proposed': {
                'pdr': round(prop_pdr, 1),
                'avg_energy': prop_energy,
                'avg_delay': prop_delay,
                'avg_throughput': prop_tp,
                'avg_hops': safe_mean(p['hops']),
                'packets': p['packets_delivered'],
                'strategies': dict(p['strategies_used'])
            },
            'improvement': {
                'pdr': round(prop_pdr - std_pdr, 1),
                'energy_saved': round(energy_saving, 1),
                'delay_reduction': round(delay_reduction, 1)
            }
        }


class RoadNetworkSimulator:
    def __init__(self):
        self.G = None
        self.pos = None
        self.node_types = {}
        self.attackers = set()
        self.attacker_speeds = {}
        self.attacker_paths = {}
        self.metrics = PerformanceMetrics()
        self.all_paths = []

        self.selected_source = None
        self.selected_dest = None
        self.current_path = None
        self.current_strategy = None
        self.alternate_paths = []

        self.simulation_time = 0
        self.is_running = False
        self.animation = None
        self.status_message = ""
        self.status_time = 0

        self.stats = {
            'paths_found': 0,
            'paths_blocked': 0,
            'attack_evasions': 0
        }

    # ============ NETWORK LOADING ============
    def load_road_network(self):
        if not os.path.exists(TXT_FILENAME):
            print("Downloading Stanford Road Network...")
            import urllib.request, gzip, shutil
            gz = "roadNet-CA.txt.gz"
            urllib.request.urlretrieve(DATASET_URL, gz)
            with gzip.open(gz, 'rb') as f_in, open(TXT_FILENAME, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz)

        print("Loading Road Network...")
        G_full = nx.read_edgelist(TXT_FILENAME, create_using=nx.Graph(),
                                  nodetype=int)
        nodes = list(nx.single_source_shortest_path_length(
            G_full, SUBGRAPH_CENTER, cutoff=SUBGRAPH_RADIUS).keys())
        self.G = G_full.subgraph(nodes).copy()
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

        print(f"✓ Loaded {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

        self.pos = nx.spring_layout(self.G, seed=42, k=0.5, iterations=100)
        self._assign_node_types()

    def _assign_node_types(self):
        nodes_list   = list(self.G.nodes())
        node_degrees = dict(self.G.degree())
        sorted_nodes = sorted(nodes_list,
                              key=lambda x: node_degrees[x], reverse=True)
        n = len(sorted_nodes)
        for i, node in enumerate(sorted_nodes):
            if   i < n * 0.10: self.node_types[node] = NodeType.RSU
            elif i < n * 0.60: self.node_types[node] = NodeType.VEHICLE
            else:               self.node_types[node] = NodeType.PEDESTRIAN

            if (self.node_types[node] == NodeType.VEHICLE
                    and random.random() < 0.1):
                self.node_types[node] = NodeType.EMERGENCY

    # ============ ATTACKER MANAGEMENT ============
    def deploy_attackers(self, num_attackers=8):
        self.attackers.clear()
        self.attacker_speeds.clear()
        self.attacker_paths.clear()

        eligible = [n for n in self.G.nodes()
                    if self.node_types[n] not in
                    [NodeType.RSU, NodeType.EMERGENCY]]

        num_attackers = min(num_attackers, len(eligible))
        for attacker in random.sample(eligible, num_attackers):
            self.attackers.add(attacker)
            speed = random.choice([0, 0.3, 0.6, 1.0])
            self.attacker_speeds[attacker] = speed
            if speed > 0:
                try:
                    dest = random.choice(eligible)
                    path = nx.shortest_path(self.G, attacker, dest)
                    self.attacker_paths[attacker] = path if len(path) > 3 \
                        else [attacker]
                except Exception:
                    self.attacker_paths[attacker] = [attacker]
            else:
                self.attacker_paths[attacker] = [attacker]

        mobile = sum(1 for s in self.attacker_speeds.values() if s > 0)
        print(f"✓ Deployed {num_attackers} attackers ({mobile} mobile)")

    def move_attackers(self):
        new_attackers = set()
        for attacker in self.attackers:
            path  = self.attacker_paths.get(attacker, [attacker])
            speed = self.attacker_speeds.get(attacker, 0)
            if speed > 0 and len(path) > 1 and random.random() < 0.3:
                try:
                    idx      = path.index(attacker) if attacker in path else 0
                    next_idx = min(idx + 1, len(path) - 1)
                    if next_idx >= len(path) - 1:
                        self.attacker_paths[attacker] = list(reversed(path))
                        next_idx = 1
                    new_pos = path[next_idx]
                    new_attackers.add(new_pos)
                    self.attacker_paths[new_pos] = \
                        self.attacker_paths.pop(attacker, path)
                    self.attacker_speeds[new_pos] = \
                        self.attacker_speeds.pop(attacker, speed)
                except Exception:
                    new_attackers.add(attacker)
            else:
                new_attackers.add(attacker)
        self.attackers = new_attackers

    # ============ PATH FINDING ============
    def find_safe_path(self, src, dest):
        if src is None or dest is None or src == dest:
            return None, None

        G_clean = self.G.copy()
        G_clean.remove_nodes_from(self.attackers)

        paths      = []
        strategies = []
        self.all_paths = []

        # --- Strategy 1: Direct Avoidance ---
        try:
            p = nx.shortest_path(G_clean, src, dest)
            paths.append(p); strategies.append("Direct Avoidance")
            self.all_paths.append(('Direct Avoidance', p, 'blue'))
            print(f"  ✓ S1 Direct Avoidance: {len(p)} hops")
        except Exception:
            print("  ✗ S1 No path")

        # --- Strategy 2: K-Shortest ---
        try:
            from itertools import islice
            k_paths = list(islice(
                nx.shortest_simple_paths(G_clean, src, dest), 3))
            for p in k_paths[1:]:   # skip the first (same as S1)
                if p not in paths:
                    paths.append(p); strategies.append("K-Shortest")
                    self.all_paths.append(('K-Shortest', p, 'red'))
                    print(f"  ✓ S2 K-Shortest: {len(p)} hops")
                    break
        except Exception as e:
            print(f"  ✗ S2: {e}")

        # --- Strategy 3: Weighted Random ---
        try:
            G_w = G_clean.copy()
            for u, v in G_w.edges():
                G_w[u][v]['weight'] = random.uniform(0.5, 1.5)
            p = nx.shortest_path(G_w, src, dest, weight='weight')
            if p not in paths:
                paths.append(p); strategies.append("Weighted")
                self.all_paths.append(('Weighted', p, 'green'))
                print(f"  ✓ S3 Weighted: {len(p)} hops")
        except Exception as e:
            print(f"  ✗ S3: {e}")

        # --- Strategy 4: Node-Disjoint ---
        try:
            if paths:
                G_d = G_clean.copy()
                for node in paths[0][1:-1]:
                    if node in G_d:
                        G_d.remove_node(node)
                p = nx.shortest_path(G_d, src, dest)
                if p not in paths:
                    paths.append(p); strategies.append("Disjoint")
                    self.all_paths.append(('Disjoint', p, 'purple'))
                    print(f"  ✓ S4 Disjoint: {len(p)} hops")
        except Exception:
            print("  ✗ S4 No disjoint path")

        if not paths:
            self.stats['paths_blocked'] += 1
            print("❌ No safe path found")
            return None, "No Safe Path"

        # Select shortest path
        best_idx      = min(range(len(paths)), key=lambda i: len(paths[i]))
        best_path     = paths[best_idx]
        best_strategy = strategies[best_idx]

        # Record metrics using real measurements
        std_result  = self.metrics.simulate_standard_routing(
            self.G, src, dest, self.attackers)
        prop_result = self.metrics.simulate_proposed_routing(
            self.G, src, dest, self.attackers, best_strategy, best_path)
        self.metrics.update_metrics(std_result, prop_result, best_strategy)

        self.current_path     = best_path
        self.current_strategy = best_strategy
        self.stats['paths_found'] += 1

        print(f"\n✅ SELECTED: {best_strategy} ({len(best_path)} hops)")
        return best_path, best_strategy

    # ============ VISUALIZATION ============
    def draw_all_strategies(self, ax):
        if not self.all_paths:
            return
        for strategy, path, color in self.all_paths:
            if path == self.current_path:
                continue
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edges,
                                   edge_color=color, width=1.5,
                                   alpha=0.3, style='dashed', ax=ax)

        from matplotlib.lines import Line2D
        color_map = {'Direct Avoidance': 'blue', 'K-Shortest': 'red',
                     'Weighted': 'green', 'Disjoint': 'purple'}
        legend_elements = [
            Line2D([0], [0], color=c, lw=2, label=s, alpha=0.5)
            for s, c in color_map.items()
            if any(x == s for x, _, _ in self.all_paths)
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def draw_network(self, ax):
        ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, edge_color='lightgray',
                               alpha=0.3, width=0.5, ax=ax)
        self.draw_all_strategies(ax)

        for node in self.G.nodes():
            if node in self.attackers:
                speed = self.attacker_speeds.get(node, 0)
                color = 'darkred' if speed > 0.7 else \
                        'red'     if speed > 0.3 else 'lightcoral'
                size  = 120 if speed > 0.7 else 100 if speed > 0.3 else 80
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=[node],
                                       node_color=color, node_size=size,
                                       node_shape='X', ax=ax)
            else:
                nt = self.node_types.get(node, NodeType.VEHICLE)
                props = {
                    NodeType.RSU:       ('blue',   's', 60),
                    NodeType.EMERGENCY: ('orange', '^', 60),
                    NodeType.VEHICLE:   ('green',  'o', 40),
                    NodeType.PEDESTRIAN:('purple', 'd', 30),
                }
                color, marker, size = props[nt]
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=[node],
                                       node_color=color, node_size=size,
                                       node_shape=marker, ax=ax)

        if self.selected_source:
            nx.draw_networkx_nodes(self.G, self.pos,
                                   nodelist=[self.selected_source],
                                   node_color='cyan', node_size=200,
                                   edgecolors='black', linewidths=2, ax=ax)
            ax.annotate('SOURCE', xy=self.pos[self.selected_source],
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='blue')

        if self.selected_dest:
            nx.draw_networkx_nodes(self.G, self.pos,
                                   nodelist=[self.selected_dest],
                                   node_color='yellow', node_size=200,
                                   edgecolors='black', linewidths=2, ax=ax)
            ax.annotate('DEST', xy=self.pos[self.selected_dest],
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='orange')

        if self.current_path:
            edges = list(zip(self.current_path, self.current_path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, edgelist=edges,
                                   edge_color='green', width=4,
                                   alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(self.G, self.pos,
                                   nodelist=self.current_path[1:-1],
                                   node_color='lightgreen',
                                   node_size=50, ax=ax)

        total = self.stats['paths_found'] + self.stats['paths_blocked']
        evasion = (self.stats['paths_found'] / max(1, total)) * 100
        ax.set_title(
            f'Road Network Security Simulator\n'
            f'Attackers: {len(self.attackers)} | '
            f'Source: {self.selected_source} | '
            f'Dest: {self.selected_dest} | '
            f'Evasion: {evasion:.1f}%',
            fontsize=11, fontweight='bold')
        ax.axis('off')

    def draw_statistics(self, ax):
        ax.clear()
        ax.axis('off')
        summary = self.metrics.get_summary()
        mobile  = sum(1 for s in self.attacker_speeds.values() if s > 0)

        txt = f"""
{'='*32}
📊 PERFORMANCE METRICS
{'='*32}

🚗 NETWORK STATUS
├─ Nodes : {self.G.number_of_nodes()}
├─ Edges : {self.G.number_of_edges()}
└─ Attackers: {len(self.attackers)} ({mobile} mobile)

{'─'*32}
📦 PACKET DELIVERY RATIO (PDR)
├─ Standard : {summary['standard']['pdr']:.1f}%
├─ Proposed : {summary['proposed']['pdr']:.1f}%
└─ Gain     : {summary['improvement']['pdr']:+.1f}%

⚡ ENERGY CONSUMPTION (units)
├─ Standard : {summary['standard']['avg_energy']:.2f}
├─ Proposed : {summary['proposed']['avg_energy']:.2f}
└─ Savings  : {summary['improvement']['energy_saved']:.1f}%

⏱  END-TO-END DELAY (s)
├─ Standard : {summary['standard']['avg_delay']:.3f}
├─ Proposed : {summary['proposed']['avg_delay']:.3f}
└─ Reduction: {summary['improvement']['delay_reduction']:.1f}%

🚀 THROUGHPUT (bytes/s)
├─ Standard : {summary['standard']['avg_throughput']:.0f}
└─ Proposed : {summary['proposed']['avg_throughput']:.0f}

📍 PATH STATISTICS
├─ Found    : {self.stats['paths_found']}
├─ Blocked  : {self.stats['paths_blocked']}
└─ Avg Hops : {summary['proposed']['avg_hops']:.1f}

🎯 STRATEGY USAGE"""

        for strat, cnt in summary['proposed']['strategies'].items():
            txt += f"\n    ├─ {strat}: {cnt}"

        if self.current_path:
            txt += f"""

📍 CURRENT PATH
├─ Hops    : {len(self.current_path)-1}
├─ Strategy: {self.current_strategy}
└─ Nodes   : {self.current_path[:2]}...{self.current_path[-2:]}"""

        txt += f"""

⏱  Sim Time: {self.simulation_time:.1f}s
{'='*32}"""

        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

        if self.status_message and time.time() - self.status_time < 3:
            ax.text(0.05, 0.05, f"📌 {self.status_message}",
                    transform=ax.transAxes, fontsize=9, color='green',
                    bbox=dict(boxstyle='round',
                              facecolor='#e8f5e9', alpha=0.9))

    # ============ INTERACTIVE CALLBACKS ============
    def on_click(self, event):
        if event.inaxes != self.ax_network:
            return
        min_dist, nearest = float('inf'), None
        for node, pos in self.pos.items():
            d = np.hypot(event.xdata - pos[0], event.ydata - pos[1])
            if d < min_dist and d < 0.1:
                min_dist, nearest = d, node

        if nearest is None:
            return

        if self.selected_source is None:
            self.selected_source = nearest
            print(f"✓ Source: {nearest}")
        elif self.selected_dest is None:
            self.selected_dest = nearest
            path, strategy = self.find_safe_path(
                self.selected_source, self.selected_dest)
            if path:
                print(f"✓ Path found: {len(path)} hops via {strategy}")
            else:
                print("✗ No safe path")
        else:
            self.selected_source = nearest
            self.selected_dest   = None
            self.current_path    = None
            print(f"✓ New source: {nearest}")

        self.update_display()

    def reset_selection(self, event=None):
        self.selected_source = self.selected_dest = self.current_path = None
        print("✓ Selection reset")
        self.update_display()

    def redeploy_attackers(self, event=None):
        self.attackers.clear()
        self.attacker_speeds.clear()
        self.attacker_paths.clear()
        self.deploy_attackers(8)
        self.current_path = self.current_strategy = None
        self.update_display()

    def toggle_simulation(self, event=None):
        self.is_running = not self.is_running
        print(f"✓ Movement {'STARTED' if self.is_running else 'PAUSED'}")

    def update_display(self):
        if hasattr(self, 'ax_network') and hasattr(self, 'ax_stats'):
            self.draw_network(self.ax_network)
            self.draw_statistics(self.ax_stats)
            plt.draw()

    def animate(self, frame):
        if self.is_running:
            self.move_attackers()
            self.simulation_time += 0.1
            if (self.selected_source and self.selected_dest
                    and self.current_path):
                if any(n in self.attackers for n in self.current_path):
                    print("⚠️  Path compromised — rerouting...")
                    path, strategy = self.find_safe_path(
                        self.selected_source, self.selected_dest)
                    if path:
                        self.stats['attack_evasions'] += 1
                        print(f"✓ Rerouted via {strategy}")
            self.update_display()

    def show_comparison(self, event=None):
        summary = self.metrics.get_summary()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance: Standard vs Proposed Routing',
                     fontsize=14, fontweight='bold')

        pairs = [
            ('PDR (%)',            'avg_energy',    'Energy (units)'),
            ('avg_energy',         'avg_delay',     'Delay (s)'),
            ('avg_delay',          'avg_throughput','Throughput (B/s)'),
        ]

        metrics_info = [
            ('PDR (%)',          summary['standard']['pdr'],
                                 summary['proposed']['pdr']),
            ('Energy (units)',   summary['standard']['avg_energy'],
                                 summary['proposed']['avg_energy']),
            ('Delay (s)',        summary['standard']['avg_delay'],
                                 summary['proposed']['avg_delay']),
            ('Throughput (B/s)', summary['standard']['avg_throughput'],
                                 summary['proposed']['avg_throughput']),
        ]

        for ax, (label, std_val, prop_val) in zip(axes.flat, metrics_info):
            bars = ax.bar(['Standard', 'Proposed'], [std_val, prop_val],
                          color=['#e74c3c', '#2ecc71'])
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            for bar, val in zip(bars, [std_val, prop_val]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

        s = summary
        print("\n" + "="*65)
        print("📊 PERFORMANCE COMPARISON")
        print("="*65)
        print(f"{'Metric':<25} {'Standard':>12} {'Proposed':>12} {'Δ':>12}")
        print("-"*65)
        print(f"{'PDR (%)':<25} {s['standard']['pdr']:>11.1f}%"
              f" {s['proposed']['pdr']:>11.1f}%"
              f" {s['improvement']['pdr']:>+11.1f}%")
        print(f"{'Energy (units)':<25} {s['standard']['avg_energy']:>12.2f}"
              f" {s['proposed']['avg_energy']:>12.2f}"
              f" {s['improvement']['energy_saved']:>+11.1f}%")
        print(f"{'Delay (s)':<25} {s['standard']['avg_delay']:>12.3f}"
              f" {s['proposed']['avg_delay']:>12.3f}"
              f" {s['improvement']['delay_reduction']:>+11.1f}%")
        print(f"{'Throughput (B/s)':<25} {s['standard']['avg_throughput']:>12.0f}"
              f" {s['proposed']['avg_throughput']:>12.0f}")
        print("="*65)

    def save_state(self, event=None):
        filename = f"road_sim_state_{int(time.time())}.pkl"
        state = {
            'attackers': self.attackers,
            'attacker_speeds': self.attacker_speeds,
            'attacker_paths': self.attacker_paths,
            'selected_source': self.selected_source,
            'selected_dest': self.selected_dest,
            'stats': self.stats,
            'time': self.simulation_time
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        self.status_message = f"Saved: {filename}"
        self.status_time    = time.time()
        print(f"✓ Saved to {filename}")
        self.update_display()

    def load_state(self, event=None):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select saved state",
            filetypes=[("Pickle files", "*.pkl")])
        root.destroy()
        if not filename:
            return
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.attackers       = state['attackers']
        self.attacker_speeds = state['attacker_speeds']
        self.attacker_paths  = state['attacker_paths']
        self.selected_source = state['selected_source']
        self.selected_dest   = state['selected_dest']
        self.stats           = state['stats']
        self.simulation_time = state['time']
        self.status_message  = f"Loaded: {filename}"
        self.status_time     = time.time()
        print(f"✓ Loaded {filename}")
        self.update_display()

    # ============ MAIN SETUP ============
    def setup_interactive(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.ax_network  = plt.subplot2grid((3, 4), (0, 0),
                                            rowspan=3, colspan=2)
        self.ax_stats    = plt.subplot2grid((3, 4), (0, 2),
                                            rowspan=3, colspan=2)

        plt.subplots_adjust(bottom=0.15)

        ax_reset    = plt.axes([0.52, 0.07, 0.10, 0.05])
        ax_redeploy = plt.axes([0.63, 0.07, 0.12, 0.05])
        ax_toggle   = plt.axes([0.76, 0.07, 0.10, 0.05])
        ax_compare  = plt.axes([0.87, 0.07, 0.10, 0.05])
        ax_save     = plt.axes([0.52, 0.01, 0.10, 0.05])
        ax_load     = plt.axes([0.63, 0.01, 0.10, 0.05])

        self.btn_reset    = Button(ax_reset,    'Reset')
        self.btn_redeploy = Button(ax_redeploy, 'Redeploy Attackers')
        self.btn_toggle   = Button(ax_toggle,   'Start / Stop')
        self.btn_compare  = Button(ax_compare,  '📊 Compare')
        self.btn_save     = Button(ax_save,     'Save State')
        self.btn_load     = Button(ax_load,     'Load State')

        self.btn_reset.on_clicked(self.reset_selection)
        self.btn_redeploy.on_clicked(self.redeploy_attackers)
        self.btn_toggle.on_clicked(self.toggle_simulation)
        self.btn_compare.on_clicked(self.show_comparison)
        self.btn_save.on_clicked(self.save_state)
        self.btn_load.on_clicked(self.load_state)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.draw_network(self.ax_network)
        self.draw_statistics(self.ax_stats)

        self.animation = FuncAnimation(self.fig, self.animate,
                                       interval=5000,
                                       cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    def run(self):
        print("=" * 60)
        print("   MTech FINAL YEAR PROJECT")
        print("   INTERACTIVE ROAD NETWORK SECURITY SIMULATOR")
        print("=" * 60)
        self.load_road_network()
        self.deploy_attackers(8)
        print("\n🚀 Launching Interactive Simulator...")
        print("Click on the network to select source / destination")
        print("=" * 60)
        self.setup_interactive()


# ============ MAIN ============
if __name__ == "__main__":
    simulator = RoadNetworkSimulator()
    simulator.run()