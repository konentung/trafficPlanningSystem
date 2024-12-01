from flask import Flask, request, render_template
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 使用 'Agg' 後端以避免 GUI 錯誤
matplotlib.use('Agg')

app = Flask(__name__)

# 定義城市道路
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
    # 繪製圖並以 Base64 格式返回圖片
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

@app.route('/find_nearest_hospital', methods=['POST'])
def find_nearest_hospital():
    current_node = request.form['current_node']
    blocked_nodes = request.form.getlist('blocked_nodes')
    method = request.form['method']

    graph = original_graph.copy()
    for node in blocked_nodes:
        if graph.has_node(node):
            graph.remove_node(node)

    hospitals = [node for node in graph.nodes if 'hospital' in node]

    nearest_hospital = None
    nearest_path = None
    min_weight = float('inf')

    try:
        for hospital in hospitals:
            if method == 'bfs':
                path = nx.shortest_path(graph, source=current_node, target=hospital)
                if nearest_path is None or len(path) < len(nearest_path):
                    nearest_hospital = hospital
                    nearest_path = path
            elif method == 'dfs':
                paths = list(nx.all_simple_paths(graph, source=current_node, target=hospital))
                path = paths[0] if paths else None
                if path and (nearest_path is None or len(path) < len(nearest_path)):
                    nearest_hospital = hospital
                    nearest_path = path
            elif method == 'dijkstra':
                path = nx.dijkstra_path(graph, source=current_node, target=hospital, weight='weight')
                total_weight = nx.dijkstra_path_length(graph, source=current_node, target=hospital, weight='weight')
                if total_weight < min_weight:
                    min_weight = total_weight
                    nearest_hospital = hospital
                    nearest_path = path

        if nearest_path:
            result = f"使用 {method} 找到從 {current_node} 到最近的hospital {nearest_hospital} 的路徑: {' -> '.join(nearest_path)}"
            img_base64 = visualize_path(graph, nearest_path, "最近 hospital 的路徑視覺化")
            return render_template('result.html', result=result, map_image=img_base64)

    except nx.NetworkXNoPath:
        result = "找不到可行的路徑。"

    return render_template('result.html', result=result, map_image=None)

if __name__ == '__main__':
    app.run(debug=True)