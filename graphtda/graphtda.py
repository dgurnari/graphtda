"""Main module."""

import numpy as np
import networkx as nx


class FilteredGraph:
    """Main class for Filtered Graphs. Basically, a NetworkX graph with filtration parameters on nodes and edges."""

    def __init__(self, G, filtration_function=None):
        """_summary_

        Parameters
        ----------
        G : NetworkX Graph or DiGraph
            the input graph, can be directed or not.
        filtration_function : callable, optional
            the filtration function, by default None
        """
        self.Graph = G

        if filtration_function:
            self.Graph = filtration_function(G)
        else:
            self.Graph = G

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
            multif_e = d["filtration"]
            multif_u = self.Graph.nodes[u]["filtration"]
            multif_v = self.Graph.nodes[v]["filtration"]

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
            multif_e = d["filtration"]
            multif_u = self.Graph.nodes[u]["filtration"]
            multif_v = self.Graph.nodes[v]["filtration"]

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
                        self.Graph.edges[(u, v)]["filtration"][i] = (
                            max(f_u, f_v) + value
                        )
                        if verbose:
                            print(
                                "now f({},{}) = {}".format(
                                    u, v, self.Graph.edges[(u, v)]["filtration"]
                                )
                            )
                    else:
                        # decreases the nodes values
                        if f_e < f_u:
                            self.Graph.nodes[u]["filtration"][i] = f_e - value
                            if verbose:
                                print(
                                    "now f({}) = {}".format(
                                        u, self.Graph.nodes[u]["filtration"]
                                    )
                                )

                        if f_e < f_v:
                            self.Graph.nodes[v]["filtration"][i] = f_e - value
                            if verbose:
                                print(
                                    "now f({}) = {}".format(
                                        v, self.Graph.nodes[v]["filtration"]
                                    )
                                )

    def compute_ECP(self):
        ECP = dict()

        for node in self.Graph.nodes():
            f = self.Graph.nodes[node]["filtration"]
            ECP[f] = ECP.get(f, 0) + 1

        for edge in self.Graph.edges():
            f = self.Graph.edges[edge]["filtration"]
            ECP[f] = ECP.get(f, 0) - 1

        # remove the contributions that are 0
        to_del = []
        for key in ECP:
            if ECP[key] == 0:
                to_del.append(key)
        for key in to_del:
            del ECP[key]

        return sorted(list(ECP.items()), key=lambda x: x[0])


def degree_filtration(G):
    for n, degree in G.degree():
        G.nodes[n]["filtration"] = (degree,)

    for u, v in G.edges:
        d_u = G.nodes[u]["filtration"][0]
        d_v = G.nodes[v]["filtration"][0]
        G.edges[(u, v)]["filtration"] = (max(d_u, d_v),)

    return G


def in_out_degree_bifiltration(D):
    for n in D.nodes:
        D.nodes[n]["filtration"] = (D.in_degree(n), D.out_degree(n))

    for u, v in D.edges:
        u_in, u_out = D.nodes[u]["filtration"]
        v_in, v_out = D.nodes[v]["filtration"]

        D.edges[u, v]["filtration"] = (max(u_in, v_in), max(u_out, v_out))

    return D
