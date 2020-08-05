
import java.util.LinkedList;

class Graph<T, E> {

	class Node<T> {
		private T state;
		private T id;
		private LinkedList<Node<T>> children;

		public Node(T state, T id) {
			this.state = state;
			this.id = id;
			this.children = (LinkedList<Node<T>>) new LinkedList();
		}
	}

	class Edge<E> {
		private double cost;
		private Node startNode;
		private Node endNode;

		public Edge(double cost, Node startNode, Node endNode) {
			this.cost = cost;
			this.startNode = startNode;
			this.endNode = endNode;

		}
	}

	private LinkedList<Graph<T, E>.Node<T>> nodes;
	private LinkedList<Graph<T, E>.Edge<E>> edges;

	public Graph() {
		this.nodes = (LinkedList<Node<T>>) new LinkedList<Graph<T, E>.Node<T>>();
		this.edges = (LinkedList<Edge<E>>) new LinkedList<Graph<T, E>.Edge<E>>();
	}

	public Node<T> addNode(T state, int id) {
		if (findNode(id) != null) {
			System.out.println("A node with the same id already exist!");
			return null;
		}
		Node node = new Node(state, id);
		nodes.addLast(node);
		return node;
	}

	public Node<T> findNode(int nodeID) {
		if (nodes.size() == 0) {
			return null;
		}
		for (Node<T> i : nodes) {
			if (i.id.equals((Integer) nodeID)) {
				Node temp = i;
				return temp;
			}
		}
		return null;
	}

	public int getNodeIndex(int nodeID) {
		if (nodes.size() == 0)
			return 0;
		for (Node<T> i : nodes) {
			if (i.id == (Integer) nodeID) {
				int t = nodes.indexOf(i);
				return t;
			}
		}
		return 0;
	}

	public Edge<E> addEdge(double cost, int startNodeID, int endNodeID) {
		Node<T> startNode;
		Node<T> endNode;
		if (findNode(startNodeID) == null) {
			startNode = new Node("Oh yeah", startNodeID);
			nodes.addLast(startNode);
		} else {
			startNode = findNode(startNodeID);
		}
		if (findNode(endNodeID) == null) {
			endNode = new Node("Oh yeah", endNodeID);
			nodes.addLast(endNode);
		} else {
			endNode = findNode(endNodeID);
		}
		Edge edge = new Edge(cost, startNode, endNode);
		edges.addLast(edge);
		LinkedList<Node<T>> ls = startNode.children;
		ls.addLast(endNode);
		return edge;
	}

	public void showNodes() {
		for (Node i : nodes) {
			System.out.println(i.state);
		}
	}

	public void showNodeChildren(int id) {
		Node<T> temp = findNode(id);
		for (Node<T> i : temp.children) {
			System.out.println(i.id);
		}
	}

	public void showEdges() {
		for (Edge i : edges) {
			System.out.println("Cost:" + i.cost + ";Start Node:" + i.startNode.id + ";End Node:" + i.endNode.id);
		}
	}

	public LinkedList<Node<T>> expand(Node<T> node) {
		return node.children;
	}

	public LinkedList<LinkedList<Node<T>>> expandNode(Node<T> node) {
		LinkedList<Node<T>> ls = expand(node);

		LinkedList<LinkedList<Node<T>>> expand = (LinkedList<LinkedList<Node<T>>>) new LinkedList();
		int size = ls.size();
		for (int i = 0; i < size; i++) {
			LinkedList<Node<T>> temp = (LinkedList<Node<T>>) new LinkedList();
			;
			temp.addFirst(node);
			temp.addFirst(ls.get(i));
			expand.addLast(temp);
		}
		for (LinkedList<Node<T>> i : expand) {
			for (Node<T> j : i) {
				System.out.print(j.id + "\t");
			}
			System.out.println("\n");
		}

		return expand;
	}

	public LinkedList<Edge<E>> expandPath(Node<T> node) {
		LinkedList<Edge<E>> paths = (LinkedList<Edge<E>>) new LinkedList();
		for (Edge<E> e : edges) {
			if (e.startNode.equals(node)) {
				paths.add(e);
			}
		}
		return paths;
	}

	public void check() {
		edges = revert(edges);
		showEdges();
	}

	public LinkedList<Node<T>> reverse(LinkedList<Node<T>> ls) {
		LinkedList<Node<T>> target = (LinkedList<Node<T>>) new LinkedList();
		for (Node<T> i : ls) {
			target.addFirst(i);
		}
		return target;
	}

	public LinkedList<Edge<E>> revert(LinkedList<Edge<E>> edges) {
		LinkedList<Edge<E>> reverted = (LinkedList<Edge<E>>) new LinkedList();
		for (Edge<E> e : edges) {
			reverted.addFirst(e);
		}
		return reverted;
	}

	public void breadthFirstSearch(Node<T> startNode, Node<T> goalNode) {
		Edge temp = new Edge(0, null, startNode);
		LinkedList<Edge<E>> open = (LinkedList<Edge<E>>) new LinkedList();
		LinkedList<Edge<E>> closed = (LinkedList<Edge<E>>) new LinkedList();
		open.addLast(temp);
		Edge currentPath = open.peekFirst();
		while (currentPath.endNode.equals(goalNode) != true) {
			boolean isThere = false;
			for (Edge<E> e : closed) {
				if (!e.equals(temp)&&currentPath.endNode.equals(e.startNode)) {
					isThere = true;
				}
			}
			if (!isThere) {
				LinkedList<Edge<E>> holder = expandPath(currentPath.endNode);
				for (Edge<E> e : holder) {
					if (open.contains(e) == false && closed.contains(e) == false)
						open.addLast(e);
				}

			}
			open.removeFirst();
			closed.add(currentPath);
			currentPath = open.peekFirst();
			if (currentPath.endNode.children == null && open.size() == 1) {
				break;
			}
		}

		LinkedList<Edge<E>> solutionPath = (LinkedList<Edge<E>>) new LinkedList();
		solutionPath.addFirst(currentPath);
		double cost=0;
		while (currentPath.startNode != startNode) {
			for (Edge<E> e : closed) {
				if (currentPath.startNode.equals(e.endNode)) {
					solutionPath.addFirst(e);
					currentPath = e;
				}
			}
		}
		System.out.print(solutionPath.peekFirst().startNode.id);
		for (Edge<E> e : solutionPath) {
			cost+=e.cost;
			System.out.print("===>");
			System.out.print(e.endNode.id);
		}
		System.out.println(";Cost:"+cost);
		System.out.println();
	}

	public void depthFirstSearch(Node<T> startNode, Node<T> goalNode) {
		Edge<E> edge = new Edge(0, null, startNode);
		LinkedList<Edge<E>> open = (LinkedList<Edge<E>>) new LinkedList();
		LinkedList<Edge<E>> closed = (LinkedList<Edge<E>>) new LinkedList();
		open.add(edge);
		Edge<E> currentPath = open.peekFirst();
		while (currentPath.endNode.id != goalNode.id) {
			boolean expanded = false;

			if (closed.size() != 0) {
				for (Edge<E> e : closed) {
					if (!e.equals(edge) && currentPath.endNode.id == e.startNode.id)
						expanded = true;
					break;
				}
			}
			if (!expanded) {
				LinkedList<Edge<E>> temp = revert(expandPath(currentPath.endNode));
				for (Edge<E> e : temp) {
					if (open.contains(e) == false && closed.contains(e) == false) {
						open.addFirst(e);
					}
				}
			}
			open.remove(currentPath);
			closed.addLast(currentPath);
			currentPath = open.peekFirst();
			if (open.size() == 0) {
				System.out.println("Goal not found!");
				return;
			}
		}
		LinkedList<Edge<E>> solutionPath = (LinkedList<Edge<E>>) new LinkedList();
		solutionPath.addLast(currentPath);
		double cost=0;
		while (currentPath.startNode != startNode) {
			for (Edge<E> e : closed) {
				if (e.endNode.equals(currentPath.startNode)) {
					solutionPath.addFirst(e);
					currentPath = e;
				}
			}
		}
		System.out.print(solutionPath.peekFirst().startNode.id);
		for (Edge<E> e : solutionPath) {
			cost+=e.cost;
			System.out.print("=====>");
			System.out.print(e.endNode.id);
		}
		System.out.println("cost:"+cost);
		System.out.println();
	}

	public void BFS(int startNodeID, int goalNodeID) {
		breadthFirstSearch(findNode(startNodeID), findNode(goalNodeID));
	}

	public void DFS(int startNodeID, int goalNodeID) {
		depthFirstSearch(findNode(startNodeID), findNode(goalNodeID));
	}

}