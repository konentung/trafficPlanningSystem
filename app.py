from flask import Flask, request, render_template
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os

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

# 創建有向圖（Graph）
original_graph = nx.DiGraph()
for node, neighbors in roads.items():
    for neighbor, speed in neighbors:
        original_graph.add_edge(node, neighbor, weight=speed)

@app.route('/')
def index():
    nodes = list(original_graph.nodes)
    # 選出所有的hospital節點
    hospitals = [node for node in nodes if 'hospital' in node]
    # 選出所有的accident節點
    accident_sites = [node for node in nodes if 'accident' in node]
    # 節點中排除hospital和accident，顯示中間路口以供封鎖選擇
    nodes_to_block = [node for node in nodes if 'hospital' not in node and 'accident' not in node]
    return render_template('index.html', nodes=nodes_to_block, hospitals=hospitals, accident_sites=accident_sites, all_nodes=nodes)

# 當路徑可視化時繪製圖
def visualize_path(graph, path, image_path, title):    
    # 繪製圖並顯示最短路徑
    plt.figure(figsize=(16, 12))  # 增大圖的大小，讓節點間更分散
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=300)  # 增加 k 值和 iterations，確保節點之間間距增大

    # 調整節點和邊的顯示參數，使得圖更加清晰
    nx.draw(
        graph, 
        pos, 
        with_labels=True, 
        node_size=5000,  # 增大節點大小，使其更明顯
        font_size=16,  # 增大字體大小
        font_color="white",  # 字體顏色設為白色，使其在藍色節點上更清晰
        node_color="#4e79a7",  # 使用更深的藍色，使節點與背景的對比更強
        edge_color="#999999",  # 調整邊的顏色為灰色，減少對節點和文字的干擾
        arrowsize=40  # 增大箭頭大小
    )

    # 獲取邊的權重並顯示
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=12, font_color="#ff7f0e")  # 增大權重字體，使用橙色

    # 強調顯示最短路徑
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=4, connectionstyle='arc3,rad=0.1', arrowstyle='->', arrowsize=60)  # 調整最短路徑顏色為紅色，增加寬度

    # 保存圖像到靜態文件夾中
    plt.title(title, fontsize=22)  # 增大標題字體，使其更加顯眼
    plt.savefig(image_path)
    plt.close()

@app.route('/find_path', methods=['POST'])
def find_path():
    # 獲取用戶選擇的起點和終點
    start = request.form['start']
    goal = request.form['goal']
    blocked_nodes = request.form.getlist('blocked_nodes')

    # 創建一個圖的副本，並根據封鎖的路口進行更新
    graph = original_graph.copy()
    for node in blocked_nodes:
        if graph.has_node(node):
            graph.remove_node(node)

    # 根據用戶選擇的方法進行路徑計算
    method = request.form['method']
    if method == 'bfs':
        try:
            path = nx.shortest_path(graph, source=start, target=goal)
            result = f"使用 BFS 找到從 {start} 到 {goal} 的最短路徑: {' -> '.join(path)}"
        except nx.NetworkXNoPath:
            path = None
            result = "找不到可行的路徑。"

    elif method == 'dijkstra':
        try:
            path = nx.dijkstra_path(graph, source=start, target=goal, weight='weight')
            total_weight = nx.dijkstra_path_length(graph, source=start, target=goal, weight='weight')
            result = f"使用 Dijkstra 演算法找到從 {start} 到 {goal} 的最短路徑（基於權重）: {' -> '.join(path)}，總權重: {total_weight}"
        except nx.NetworkXNoPath:
            path = None
            result = "找不到可行的路徑。"
    else:
        result = "無效的搜索方法。"
        path = None

    # 如果需要可視化，則繪製圖
    if method == 'bfs' or method == 'dijkstra':
        visualize_path(graph, path, os.path.join('static', 'city_network_path.png'), "城市道路網絡視覺化")
        return render_template('result.html', result=result, map_image='city_network_path.png')
    else:
        return render_template('result.html', result=result, map_image=None)

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
            if method == 'bfs' or method == 'dijkstra':
                visualize_path(graph, nearest_path, os.path.join('static', 'nearest_hospital_path.png'), "最近hospital的路徑視覺化")
                return render_template('result.html', result=result, map_image='nearest_hospital_path.png')

        else:
            result = "找不到可行的路徑。"
    except nx.NetworkXNoPath:
        result = "找不到可行的路徑。"

    return render_template('result.html', result=result, map_image=None)

if __name__ == '__main__':
    app.run(debug=True)