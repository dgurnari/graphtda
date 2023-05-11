"""Main module."""

import numpy as np
import networkx as nx
from pyrivet import rivet
from pyrivet import hilbert_distance
import ot, ot.bregman


class FilteredGraph:
    """Main class for Filtered Graphs. Basically, a NetworkX graph with filtration parameters on nodes and edges."""

    def __init__(self, G, filtration_function=None, **kwargs):
        """Creates a filtered version of G without changing G.

        Parameters
        ----------
        G : NetworkX Graph or DiGraph
            the input graph, can be directed or not.
        filtration_function : callable, optional
            the filtration function, by default None.
            Use one of the functions specified below or create your own.
        **kwargs : dict
            arguments to be passed to the filtration_function.
        """
        self.Graph = G.copy(as_view=False)
        self.x_label = "x"
        self.y_label = "y"
        self.xreverse = False

        if filtration_function:
            self.Graph = filtration_function(self.Graph, **kwargs)
        # else:
        #    self.Graph = G

        self.check_filtration()

    def check_filtration(self, verbose=False, stop_on_first_error=True):
        """Check whether the graph filtration is consistent, e.g. edges have higher values that the corresponding nodes.

        Parameters
        ----------
        verbose : bool, optional
            _description_, by default False
        stop_on_first_error : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        is_filtration_ok = True

        for u, v, d in self.Graph.edges(data=True):
            multif_e = d["appearance"]
            multif_u = self.Graph.nodes[u]["appearance"]
            multif_v = self.Graph.nodes[v]["appearance"]

            for f_u, f_v, f_e in zip(multif_u, multif_v, multif_e):
                if (f_e < f_u) or (f_e < f_v):
                    is_filtration_ok = False

                    if verbose:
                        print(
                            "wrong filtration in edge ({}, {}) = {}".format(
                                u, v, multif_e
                            )
                        )
                        print("f({}) = {}".format(u, multif_u))
                        print("f({}) = {}\n".format(v, multif_v))

                        if stop_on_first_error:
                            break

        return is_filtration_ok

    def make_filtration_non_decreasing(
        self, verbose=False, increase_edges=True, value=1
    ):
        """_summary_

        Parameters
        ----------
        verbose : bool, optional
            _description_, by default False
        increase_edges : bool, optional
            _description_, by default True
        value : int, optional
            _description_, by default 1
        """
        for u, v, d in self.Graph.edges(data=True):
            multif_e = d["appearance"]
            multif_u = self.Graph.nodes[u]["appearance"]
            multif_v = self.Graph.nodes[v]["appearance"]

            for i, f_u, f_v, f_e in enumerate(zip(multif_u, multif_v, multif_e)):
                if (f_e < f_u) or (f_e < f_v):
                    if verbose:
                        print(
                            "wrong filtration in edge ({}, {}) = {}".format(
                                u, v, multif_e
                            )
                        )
                        print("f({}) = {}".format(u, multif_u))
                        print("f({}) = {}\n".format(v, multif_v))

                    if increase_edges:
                        # increases the edge value
                        self.Graph.edges[(u, v)]["appearance"][i] = (
                            max(f_u, f_v) + value
                        )
                        if verbose:
                            print(
                                "now f({},{}) = {}".format(
                                    u, v, self.Graph.edges[(u, v)]["appearance"]
                                )
                            )
                    else:
                        # decreases the nodes values
                        if f_e < f_u:
                            self.Graph.nodes[u]["appearance"][i] = f_e - value
                            if verbose:
                                print(
                                    "now f({}) = {}".format(
                                        u, self.Graph.nodes[u]["appearance"]
                                    )
                                )

                        if f_e < f_v:
                            self.Graph.nodes[v]["appearance"][i] = f_e - value
                            if verbose:
                                print(
                                    "now f({}) = {}".format(
                                        v, self.Graph.nodes[v]["appearance"]
                                    )
                                )

    def compute_ECP(self):
        """Euler Characteristic Profile

        Euler characteristic profile at degree z is the difference
        (number of vertices alive at z) - (number of edges alive at z).
        Only works for 1-critical filtrations for now.

        Returns
        -------
        list
            list of tuples of (degree, contribution)
        """
        ECP = dict()

        for node in self.Graph.nodes():
            f = self.Graph.nodes[node]["appearance"][0]
            ECP[f] = ECP.get(f, 0) + 1

        for edge in self.Graph.edges():
            f = self.Graph.edges[edge]["appearance"][0]
            ECP[f] = ECP.get(f, 0) - 1

        # remove the contributions that are 0
        to_del = []
        for key in ECP:
            if ECP[key] == 0:
                to_del.append(key)
        for key in to_del:
            del ECP[key]

        return sorted(list(ECP.items()), key=lambda x: x[0])

    def rivet_bifiltration(self):
        """translate to RIVET's bifiltration format


        Returns
        -------
        rivet.Bifiltration
            The rivet representation of this FilteredGraph
        """
        simplices = []
        appearances = []
        for node in self.Graph.nodes:
            simplices.append([node])
            appearances.append(self.Graph.nodes[node]["appearance"])

        for edge in self.Graph.edges:
            simplices.append(list(edge))
            appearances.append(self.Graph.edges[edge]["appearance"])

        return rivet.Bifiltration(
            x_label=self.x_label,
            y_label=self.y_label,
            simplices=simplices,
            appearances=appearances,
            xreverse=self.xreverse,
        )

    def compute_bipersistence(self, dim=0, x=0, y=0):
        """Call RIVET to compute bipersistence

        Coarsening specified by parameters x, y; see RIVET's documentation for further information.

        Parameters
        ----------
        dim : int, optional
            dimension of the homology to compute, by default 0
        x : int, optional
            coarsening in x direction, by default 0
        y : int, optional
            coarsening in y direction, by default 0

        Returns
        -------
        rivet.Betti
            A rivet.Betti object representing the bipersistence module
        """
        self.betti = rivet.betti(self.rivet_bifiltration(), homology=dim, x=x, y=y)
        return self.betti

    # def graded_rank(self):
    #     """Multigraded Betti numbers

    #     Make sure you run compute_bipersistence first before calling this method.

    #     Returns
    #     -------
    #     numpy.array
    #         A numpy array of multigraded Betti numbers.
    #     """
    #     if self.betti is None:
    #         print("compute bipersistence first!")
    #     else:
    #         return self.betti.graded_rank

    def hilbert_function(self):
        """Hilbert function

        Make sure you run compute_bipersistence first before calling this method.


        Returns
        -------
        numpy.array
            numpy array with entries equal the dimension of the homology vector space in the respective degree
        """
        return hilbert_distance.betti_to_splitmat(self.betti)


def degree_filtration(G):
    """filter by node degree

    Sublevelset filtration using lower-star from degree as vertex values: endows G's vertices and edges with an "appearance".
    To be used as a filtration_function when constructing a FilteredGraph


    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration

    """
    for n, degree in G.degree():
        G.nodes[n]["appearance"] = [(degree,)]

    for u, v in G.edges:
        d_u = G.nodes[u]["appearance"][0][0]
        d_v = G.nodes[v]["appearance"][0][0]
        G.edges[(u, v)]["appearance"] = [(max(d_u, d_v),)]

    return G


def in_out_degree_bifiltration(D):
    """filter by in- and out-degree

    Sublevelset bifiltration using lower-star from (indegree,outdegree) as vertex values: endows D's vertices and edges with an "appearance".

    To be used as a filtration_function when constructing a FilteredGraph.


    Parameters
    ----------
    D : networkx.Digraph
        The digraph to be filtered.

    Returns
    -------
    networkx.Digraph
        The input digraph endowed with filtration
    """
    for n in D.nodes:
        D.nodes[n]["appearance"] = (D.in_degree(n), D.out_degree(n))

    for u, v in D.edges:
        u_in, u_out = D.nodes[u]["appearance"]
        v_in, v_out = D.nodes[v]["appearance"]

        D.edges[u, v]["appearance"] = (max(u_in, v_in), max(u_out, v_out))

    return D


def HKS_bifiltration(G, grid=np.linspace(0, 10, 101)):
    """Heat kernel signature bifiltration

    At bidegree (h,t) consists of the subgraph whose vertices have at time t temparature at most h and all edges between them.
    \infty-critical bifiltration.
    To be used as a filtration_function when constructing a FilteredGraph.


    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered
    grid : list or numpy.array, optional
        discretization of the time parameter, by default np.linspace(0,10,101)

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    L = nx.normalized_laplacian_matrix(G)
    values, vectors = np.linalg.eigh(L.todense())
    for n in G.nodes:
        G.nodes[n]["appearance"] = [
            (
                sum(
                    [
                        np.exp(-t * values[i]) * vectors[i][n] ** 2
                        for i in range(len(values))
                    ]
                ),
                t,
            )
            for t in grid
        ]
    for u, v in G.edges:
        G.edges[u, v]["appearance"] = [
            (
                max([G.nodes[u]["appearance"][i][0], G.nodes[v]["appearance"][i][0]]),
                grid[i],
            )
            for i in range(len(grid))
        ]

    return G


def product_bifiltration(G, G1, G2):
    """product of two 1-filtrations

    Assumes G1, G2 are two 1-critical 1-parameter-filtered versions of G.
    The appearance of a simplex in the product bifiltration is the pair of appearances in the two input filtrations.
    To be used as a filtration_function when constructing a FilteredGraph.


    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered
    G1 : FilteredGraph
        1-critical 1-parameter-filtered version of G
    G2 : FilteredGraph
        1-critical 1-parameter-filtered verson of G

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    # if G != G1.Graph or G != G2.Graph:
    #    print("Error: not the same underlying graph")
    #    return
    for n in G.nodes:
        G.nodes[n]["appearance"] = [
            (
                G1.Graph.nodes[n]["appearance"][0][0],
                G2.Graph.nodes[n]["appearance"][0][0],
            )
        ]

    for e in G.edges:
        G.edges[e]["appearance"] = [
            (
                G1.Graph.edges[e]["appearance"][0][0],
                G2.Graph.edges[e]["appearance"][0][0],
            )
        ]

    return G


def interlevel_bifiltration(G, FG, keep_nodes=True):
    """interlevel bifiltration from 1-parameter sublevel filtration

    Assumes FG is a sublevel-filtered version of G.
    The interlevel set filtration at bidegree (x,y) consists of the subset whose wiltration values are in the interval [x,y].
    This is in general not a graph; to fix this, one can either keep the node values fix and increase the appearance of the edges to make sure the endpoints are present,
    or keep the edge values fix and set each node to appear whenever any of its incident edges appears.
    To be used as a filtration_function when constructing a FilteredGraph.
    Need to set xreverse=True in the FilteredGraph for sensible homology.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered
    FG : FilteredGraph
        A sublevel filtered version of G
    keep_nodes : bool, optional
        way to ensure the bifiltration consists of subgraphs. True->increase edge values; False->keep edge values, decrease node values, by default True

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    bifilG = nx.Graph()
    bifilG.add_nodes_from(G.nodes)
    bifilG.add_edges_from(G.edges)
    if (
        keep_nodes
    ):  # nodes appear on diagonal, edges as soon as both endpoints are present
        for n in G.nodes:
            bifilG.nodes[n]["appearance"] = [
                (
                    FG.Graph.nodes[n]["appearance"][0][0],
                    FG.Graph.nodes[n]["appearance"][0][0],
                )
            ]

        for e in G.edges:
            vals = [FG.Graph.nodes[e[i]]["appearance"][0][0] for i in range(2)]
            bifilG.edges[e]["appearance"] = [(min(vals), max(vals))]

    else:  # edges appear on the diagonal, nodes as soon as required for edge
        for e in G.edges:
            bifilG.edges[e]["appearance"] = [
                (
                    FG.Graph.edges[e]["appearance"][0][0],
                    FG.Graph.edges[e]["appearance"][0][0],
                )
            ]

        for n in G.nodes:
            vals = set()
            for e in nx.edges(G, n):
                vals.add(FG.Graph.edges[e]["appearance"][0][0])
            bifilG.nodes[n]["appearance"] = [(val, val) for val in vals]

    return bifilG


def hks(G, t):
    """Heat kernel signature sublevel filtration


    Sublevel filtration, consists at degree h of the subgraph whose vertices have temperature at most h at time fixed by input parameter t.
    (and all edges between them.)
    To be used as a filtration_function when constructing a FilteredGraph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered
    t : float
        time

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    L = nx.normalized_laplacian_matrix(G)
    values, vectors = np.linalg.eigh(L.todense())

    for n in G.nodes:
        G.nodes[n]["appearance"] = [
            (
                sum(
                    [
                        np.exp(-t * values[i]) * vectors[i][n] ** 2
                        for i in range(len(values))
                    ]
                ),
            )
        ]
    for u, v in G.edges:
        G.edges[u, v]["appearance"] = [
            (max([G.nodes[u]["appearance"][0][0], G.nodes[v]["appearance"][0][0]]),)
        ]

    return G


def lazy_random_walk_measure(G, alpha, u):
    """probability distribution of randomly walking away from u
    Starting at node u, stay there with probability alpha;
    with probability (1-alpha)/deg(u) walk to any neighbor.
    Used for Ollivier-Ricci curvature

    Parameters
    ----------
    G : networkx.Graph
        Graph on which the random walk  takes place
    alpha : float
        "laziness" parameter in [0,1] -- 0->always move away from u, 1->always stay at u
    u : int
        starting node

    Returns
    -------
    numpy.array
        array containing the probability of ending up at each node.
    """
    values = np.zeros_like(G.nodes, dtype=float)
    values[u] = alpha
    for v in G.neighbors(u):
        values[v] = (1 - alpha) / G.degree[u]
    return values


def ollivier_ricci_curvature(G, alpha, reg=0):
    """ollivier-ricci curvature sublevel filtration
    At degree z, consists of edges of curvature at most z and all incident nodes.
    Curvature compares neighborhoods of the endpoints using Wasserstein distance of lazy random walk measures starting there.
    To be used as a filtration_function when constructing a FilteredGraph.


    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered
    alpha : float
        laziness parameter for random walk measure
    reg : float, optional
        regularization parameter, 0->solve exactly, 1->solve Sinkhorn relaxation, by default 0

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    for v in G.nodes:
        # default values on the nodes
        G.nodes[v]["appearance"] = [(np.inf,)]

    p = dict(nx.shortest_path_length(G))

    for e in G.edges:
        M = np.zeros((len(G.nodes), len(G.nodes)))
        for i in range(len(G.nodes)):
            for j in range(i):
                # if the two nodes are in different connected components the default value is np.inf
                M[i, j] = p[i].get(j, np.inf)
        M = M + M.T
        if reg == 0:
            Wd = ot.emd2(
                lazy_random_walk_measure(G, alpha, e[0]),
                lazy_random_walk_measure(G, alpha, e[1]),
                M,
            )
        else:
            Wd = ot.bregman.sinkhorn(
                lazy_random_walk_measure(G, alpha, e[0]),
                lazy_random_walk_measure(G, alpha, e[1]),
                M,
                reg,
            )
        G.edges[e]["appearance"] = [(1 - Wd,)]
        if 1 - Wd < G.nodes[e[0]]["appearance"][0][0]:
            G.nodes[e[0]]["appearance"] = [(1 - Wd,)]
        if 1 - Wd < G.nodes[e[1]]["appearance"][0][0]:
            G.nodes[e[1]]["appearance"] = [(1 - Wd,)]

    # if some node still has value +inf, it means that it is isolated, so we set its value to -1
    for v in G.nodes:
        if G.nodes[v]["appearance"] == [(np.inf,)]:
            G.nodes[v]["appearance"] = [(-1,)]

    return G


def forman_ricci_curvature(G):
    """Forman-Ricci curvature sublevel filtration
    At degree z, consists of edges of curvature at most z and all incident nodes.
    FRC((u,v)) = 4-deg(u)-deg(v)+3*|triangles|,
    where triangles have u and v as two of their vertics.
    To be used as a filtration_function when constructing a FilteredGraph.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be filtered

    Returns
    -------
    networkx.Graph
        The input graph endowed with filtration
    """
    for v in G.nodes:
        G.nodes[v]["appearance"] = [(-1,)]

    for e in G.edges:
        n_triangles = len(set(G.neighbors(e[0])) & set(G.neighbors(e[1])))
        G.edges[e]["appearance"] = [
            (4 - G.degree[e[0]] - G.degree[e[1]] + 3 * n_triangles,)
        ]

    return G
