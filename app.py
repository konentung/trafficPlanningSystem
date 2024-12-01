from flask import Flask, request, render_template
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 使用 'Agg' 後端以避免 GUI 錯誤
matplotlib.use('Agg')

app = Flask(__name__)

# 定義包含平均速率的城市道路網絡（簡化版本）
roads = {
    'accident1': [('K', 50), ('N', 40)],
    'hospital1': [('B', 10), ('A', 30), ('L', 25), ('C', 20)],
    'A': [('hospital1', 30), ('N', 50), ('K', 15), ('accident3', 40)],
    'B': [('hospital1', 10), ('M', 20), ('E', 35)],
    'C': [('hospital1', 20), ('E', 25), ('G', 50)],
    'E': [('B', 35), ('C', 25), ('hospital2', 15)],
    'accident3': [('A', 40), ('G', 25), ('J', 30), ('F', 20)],
    'F': [('accident3', 20), ('K', 15), ('J', 25)],
    'hospital2': [('E', 15), ('G', 20)],
    'G': [('C', 50), ('accident3', 25), ('hospital2', 20), ('H', 10)],
    'H': [('G', 10), ('accident2', 45)],
    'accident2': [('H', 45), ('J', 20)],
    'J': [('accident3', 30), ('accident2', 20)],
    'K': [('A', 15), ('accident1', 50)],
    'L': [('hospital1', 25), ('M', 30), ('N', 35)],
    'M': [('B', 20), ('L', 30)],
    'N': [('A', 50), ('L', 35), ('accident1', 40)],
}

# 創建有向圖
original_graph = nx.DiGraph()
for node, neighbors in roads.items():
    for neighbor, speed in neighbors:
        original_graph.add_edge(node, neighbor, weight=speed)

@app.route('/')
def index():
    nodes = list(original_graph.nodes)
    hospitals = [node for node in nodes if 'hospital' in node]
    accident_sites = [node for node in nodes if 'accident' in node]
    nodes_to_block = [node for node in nodes if 'hospital' not in node and 'accident' not in node]
    return render_template('index.html', nodes=nodes_to_block, hospitals=hospitals, accident_sites=accident_sites, all_nodes=nodes)

def visualize_path(graph, path, title):
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=300)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=5000,
        font_size=16,
        font_color="white",
        node_color="#4e79a7",
        edge_color="#999999",
        arrowsize=40,
    )

    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=12, font_color="#ff7f0e")

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=4, connectionstyle='arc3,rad=0.1', arrowstyle='->', arrowsize=60)

    plt.title(title, fontsize=22)

    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64

@app.route('/find_path', methods=['POST'])
def find_path():
    start = request.form['start']
    goal = request.form['goal']
    blocked_nodes = request.form.getlist('blocked_nodes')

    graph = original_graph.copy()
    for node in blocked_nodes:
        if graph.has_node(node):
            graph.remove_node(node)

    method = request.form['method']
    try:
        if method == 'bfs':
            path = nx.shortest_path(graph, source=start, target=goal)
            result = f"使用 BFS 找到從 {start} 到 {goal} 的最短路徑: {' -> '.join(path)}"
        elif method == 'dijkstra':
            path = nx.dijkstra_path(graph, source=start, target=goal, weight='weight')
            total_weight = nx.dijkstra_path_length(graph, source=start, target=goal, weight='weight')
            result = f"使用 Dijkstra 演算法找到從 {start} 到 {goal} 的最短路徑（基於權重）: {' -> '.join(path)}，總權重: {total_weight}"
        else:
            raise ValueError("無效的搜索方法")
    except nx.NetworkXNoPath:
        path = None
        result = "找不到可行的路徑。"

    img_base64 = visualize_path(graph, path, "城市道路網絡視覺化") if path else None
    return render_template('result.html', result=result, map_image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)