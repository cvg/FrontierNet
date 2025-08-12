import networkx as nx
from frontier.base import Base
import logging


class FrontierGraph(Base):
    def __init__(self, params=None, log_level=logging.INFO):
        super().__init__(params, log_level=log_level)
        self.graph = nx.Graph()
        self.logger.debug("FrontierGraph initialized")

        self.type_range = {
            "F": (0, 1000000),  # Frontier nodes
            "R": (1000001, 9999999),  # Robot nodes
        }

    def add_node_R(self, robot_id):
        """
        Add a robot node to the graph.
        Args:
            robot_id: ID of the robot node.
        """
        assert not self.graph.has_node(
            robot_id
        ), f"Robot node {robot_id} already exists."
        self.graph.add_node(robot_id, type="R")
        self.logger.debug(f"Robot node {robot_id} added")

    def add_node_F(self, frontier_id):
        """
        Add a frontier node to the graph.
        Args:
            frontier_id: ID of the frontier node.
        """
        assert not self.graph.has_node(
            frontier_id
        ), f"Frontier node {frontier_id} already exists."
        self.graph.add_node(frontier_id, type="F")
        self.logger.debug(f"Frontier node {frontier_id} added")

    def add_edge_FR(self, frontier_id, robot_id, weight=1):
        """
        Add an edge between a frontier and a robot in the graph.
        Args:
            frontier_id: ID of the frontier node.
            robot_id: ID of the robot node.
            weight: Weight of the edge.
        """
        assert self.graph.has_node(robot_id), f"Robot node {robot_id} does not exist."
        assert self.graph.has_node(
            frontier_id
        ), f"Frontier node {frontier_id} does not exist."
        if self.graph.has_edge(frontier_id, robot_id):
            self.logger.debug(
                f"Edge between Frontier {frontier_id} and Robot {robot_id} already exists. Skipping addition."
            )
            return

        self.graph.add_edge(frontier_id, robot_id, weight=weight)
        self.logger.debug(
            f"Edge added between Frontier {frontier_id} and Robot {robot_id} with weight {weight}"
        )

    def add_edge_RR(self, robot_id1, robot_id2, weight=1):
        """
        Add an edge between two robot nodes in the graph.
        Args:
            robot_id1: ID of the first robot node.
            robot_id2: ID of the second robot node.
            weight: Weight of the edge.
        """
        assert self.graph.has_node(robot_id1), f"Robot node {robot_id1} does not exist."
        assert self.graph.has_node(robot_id2), f"Robot node {robot_id2} does not exist."
        if self.graph.has_edge(robot_id1, robot_id2):
            self.logger.debug(
                f"Edge between Robot {robot_id1} and Robot {robot_id2} already exists. Skipping addition."
            )
            return
        self.graph.add_edge(robot_id1, robot_id2, weight=weight)
        self.logger.debug(
            f"Edge added between Robot {robot_id1} and Robot {robot_id2} with weight {weight}"
        )

    def remove_node_F(self, frontier_id):
        """
        Remove a frontier node from the graph, and all associated edges.
        Args:
            frontier_id: ID of the frontier node to remove.
        """
        assert self.graph.has_node(
            frontier_id
        ), f"Frontier node {frontier_id} does not exist."
        self.graph.remove_node(frontier_id)
        self.logger.debug(f"Frontier node {frontier_id} removed")

    def get_node_F(self):
        """
        Get all frontier nodes in the graph.
        Returns:
            List of frontier node IDs.
        """
        frontier_nodes = [n for n, d in self.graph.nodes(data=True) if d["type"] == "F"]
        self.logger.debug(f"Retrieved {len(frontier_nodes)} frontier nodes")
        return frontier_nodes

    def get_node_R(self):
        """
        Get all robot nodes in the graph.
        Returns:
            List of robot node IDs.
        """
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d["type"] == "R"]
        self.logger.debug(f"Retrieved {len(robot_nodes)} robot nodes")
        return robot_nodes

    def get_edge_FR(self):
        """
        Get all edges between frontier and robot nodes in the graph.
        Returns:
            List of tuples (frontier_id, robot_id, weight).
        """
        edges = [
            (u, v, d["weight"])
            for u, v, d in self.graph.edges(data=True)
            if self.graph.nodes[u]["type"] != self.graph.nodes[v]["type"]
        ]
        self.logger.debug(
            f"Retrieved {len(edges)} edges between frontier and robot nodes"
        )
        return edges

    def get_edge_RR(self):
        """
        Get all edges between robot nodes in the graph.
        Returns:
            List of tuples (robot_id1, robot_id2, weight).
        """
        edges = [
            (u, v, d["weight"])
            for u, v, d in self.graph.edges(data=True)
            if self.graph.nodes[u]["type"] == "R" and self.graph.nodes[v]["type"] == "R"
        ]
        self.logger.debug(f"Retrieved {len(edges)} edges between robot nodes")
        return edges

    def get_shortest_R_to_F(self, robot_id: int, frontier_id: int):
        """
        Return the shortest path from a robot node to a frontier node
        while forbidding any other frontier nodes from appearing inside
        the path.
        Args:
            robot_id     : int   (must be type 'R')
            frontier_id  : int   (must be type 'F')
            path      : list[int]      node IDs (robot … robot … frontier)
            distance  : float | int    total weight
        """
        assert self.graph.has_node(robot_id), f"Robot node {robot_id} missing"
        assert self.graph.has_node(frontier_id), f"Frontier node {frontier_id} missing"
        assert self.graph.nodes[robot_id]["type"] == "R", f"{robot_id} is not a robot"
        assert (
            self.graph.nodes[frontier_id]["type"] == "F"
        ), f"{frontier_id} is not a frontier"

        # build a *view* that hides all other frontier nodes
        allowed = {frontier_id} | {
            n for n, d in self.graph.nodes(data=True) if d["type"] == "R"
        }

        G_view = nx.subgraph_view(self.graph, filter_node=lambda n: n in allowed)
        # NOTE: subgraph_view is read-only and O(1) to create.

        # compute shortest path on the restricted graph
        try:
            path = nx.shortest_path(
                G_view, source=robot_id, target=frontier_id, weight="weight"
            )
            distance = nx.shortest_path_length(
                G_view, source=robot_id, target=frontier_id, weight="weight"
            )
        except nx.NetworkXNoPath:
            self.logger.warning(f"No R→F path from {robot_id} to {frontier_id}")
            raise

        # path[0…-2] are guaranteed 'R'; path[-1] is the frontier
        self.logger.debug(f"Robot-only path {path} (d={distance})")
        return path, distance

    def return_current_graph(self):
        """
        Return the current graph.
        Returns:
            current graph as dictionary: {"nodes": list, "edges": list}
        """
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        self.logger.debug(
            f"Returning current graph with {len(nodes)} nodes and {len(edges)} edges"
        )
        return {"nodes": nodes, "edges": edges}
