from music21 import *
import networkx as nx
import numpy as np
import os
os.putenv('DISPLAY', ':99.0')

## Load score 

path = 'Mozart_String_Quartet/'
mxl_file = path+'String_Quartet_No._5_in_A_major_Opus_18_No_5.mxl'
k421 = converter.parse(mxl_file, format='musicxml')
score = k421.measures(1,2)

scoreTree = parse_verticality(score)

## Construct a graph 
### Add node first
G = nx.Graph()
G, ver_len = add_node(scoreTree)
### Add horizontal edge with weights
G = add_horizontal_edge(G)
### Add vertical edge with weights
G = add_vertical_edge(G, ver_len)

## Normalized Cut
v_partition = normalized_min_cut(G)


## Ideally
right_hand, left_hand = graph_to_score(G, v_partition, score)




## Functions on graph construction (Details may change)
part_to_idx = {"Violino I":0, "Violino II":1, "Viola":2, "Violoncello":3}

def parse_verticality(score):
    scoreTree = tree.fromStream.asTimespans(score, flatten=True,
       classList=(note.Note, chord.Chord))
    
    return scoreTree

def add_node(scoreTree):
    G = nx.Graph()
    n_idx = 1
    break_flag = False
    
    for i, verticality in enumerate(scoreTree.iterateVerticalities()):
        if break_flag:  
            break
        else:         
            print(f"{i} verticality starts {verticality.offset} - ends {verticality.nextStartOffset}")
            if verticality.nextStartOffset-verticality.offset>=10:
                break_flag = True;
            # all the notes that just started in this verticality
            startedNotes = verticality.startTimespans
            print(startedNotes)
            for snote in startedNotes:
                part_idx = part_to_idx[snote.part.partName]
                G.add_nodes_from([(n_idx, {"note": snote, "start": snote.offset, "end":snote.endTime, "ver": i, "part": part_idx})])
                n_idx += 1        
            print('------------------')
            ver_len = i
    
    return G, ver_len


def add_horizontal_edge(G):
    for instrument in ["Violino I", "Violino II", "Viola", "Violoncello"]:
        notes_same_part = [x for x,y in G.nodes(data=True) if y['note'].part.partName==instrument]
        for i in range(len(notes_same_part)-1):
            w = assign_horizontal_weight(G.nodes[notes_same_part[i]]["note"], G.nodes[notes_same_part[i+1]]["note"])
            G.add_edge(notes_same_part[i], notes_same_part[i+1], weight=w)

    return G

def assign_horizontal_weight(u, v, h_reg=2.0, part_bias=0.2, slur_bias = 0.4):
    pitch1, pitch2 = u.element,v.element
    h_int = abs(float(interval.Interval(noteStart=pitch1, noteEnd=pitch2).cents)/100)
    if h_int == 0:
        return 1000
    else:
        if check_slur(u,v):
            return 1*h_reg/h_int+part_bias+slur_bias
        else:
            return 1*h_reg/h_int+part_bias


def check_slur(u,v):
    if u.element.getSpannerSites() == [] or v.element.getSpannerSites() == []:
        return True

def add_vertical_edge(G, ver_len):
    for ver in range(ver_len):
        notes_same_vert = [x for x,y in G.nodes(data=True) if y['ver']==ver]
        for i in range(len(notes_same_vert)-1):
            w = assign_vertical_weight(G.nodes[notes_same_vert[i]]["note"], G.nodes[notes_same_vert[i+1]]["note"])
            G.add_edge(notes_same_vert[i], notes_same_vert[i+1], weight=w)
    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            node1, node2 = G.nodes[i+1], G.nodes[j+1]
            if node1["start"] <= node2["start"] and node1["end"] >= node2["end"]:
                if abs(node1["part"]-node2["part"]==1):
                    print(i,j)
                    w = assign_vertical_weight(node1["note"], node2["note"])
                    G.add_edge(i, j, weight=w)

    return G

def assign_vertical_weight(u, v, h_reg=2.0, concur_bias = 0.4):
    pitch1, pitch2 = u.element,v.element
    h_int = abs(float(interval.Interval(noteStart=pitch1, noteEnd=pitch2).cents)/100)
    if h_int == 0:
        return 1000
    else:
        if h_int <= 6:
            return 1*h_reg/h_int + concur_bias
        else:
            return 1*h_reg/h_int

### Graph Normalized cut

def normalized_min_cut(graph):

    m_adjacency = np.array(nx.to_numpy_matrix(graph))

    D = np.diag(np.sum(m_adjacency, 0))
    D_half_inv = np.diag(1.0 / np.sqrt(np.sum(m_adjacency, 0)))
    M = np.dot(D_half_inv, np.dot((D - m_adjacency), D_half_inv))

    (w, v) = np.linalg.eig(M)
    #find index of second smallest eigenvalue
    index = np.argsort(w)[1]

    v_partition = v[:, index]
    v_partition = np.sign(v_partition)
    return v_partition

























































