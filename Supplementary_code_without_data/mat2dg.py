#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:17:07 2021

@author: Alex
"""
def mat_to_dg(wh):
    # Converts weight matrix to directed graph
    DG = nx.DiGraph()
    
    val_locs=np.nonzero(wh)
    rows=val_locs[0]
    cols=val_locs[1]
    
    n_con=len(rows)
    for ward in range(n_con):
        DG.add_edge(rows[ward], cols[ward])
    
    return DG