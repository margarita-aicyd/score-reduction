from music21 import *
import networkx as nx
import numpy as np
import copy

# ## Functions on graph construction (Details may change)
# part_to_idx = {"Violino I":0, 
#                 "Violino II":1, 
#                 "Viola":2, 
#                 "Violoncello":3,
#                 "Violin I":0, 
#                 "Violin II":1, 
#                 }

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

class ScoreGraph():
    def __init__(self, score,
                 h_reg = 2.0, concur_bias = 0.4,
                 part_bias = 0.2, slur_bias = 0.4):

        self.score = score
        self.part_to_idx = {part.partName: i for i, part in enumerate(score.parts)}
        self.scoreTree = tree.fromStream.asTimespans(self.score, flatten=True,
                                            classList=(note.Note, chord.Chord))
            
        self.h_reg = h_reg
        self.concur_bias = concur_bias
        self.part_bias = part_bias
        self.slur_bias = slur_bias

        ## Construct the graph 
        self.construct_graph()        

    def construct_graph(self):
        self.G = nx.Graph()
        ### Add node first
        self.add_nodes()
        ### Add horizontal edge with weights
        self.add_horizontal_edges()
        ### Add vertical edge with weights
        self.add_vertical_edges()

    def __len__(self):
        return len(self.G.nodes)

    def add_nodes(self):
        # G = nx.Graph()
        n_idx = 1 # TODO: change to 0
        break_flag = False
        
        for i, verticality in enumerate(self.scoreTree.iterateVerticalities()):
            if break_flag:  
                break
            else:         
                print(f"{i} verticality starts {verticality.offset} - ends {verticality.nextStartOffset}")
                # if verticality.nextStartOffset-verticality.offset>=10:
                #     break_flag = True
                # TODO: I commented this out because for some reason some 
                # TODO: verticality.nextStartOffset can be None

                # all the notes that just started in this verticality
                startedNotes = verticality.startTimespans
                print(startedNotes)
                for snote in startedNotes:
                    # NOTE: snote is a timespan object and not a note
                    # NOTE: snote.element is either a Note or a Chord object
                    # NOTE: however, for monophonic parts though, snote.element is a Note object
                    # part_idx = part_to_idx[snote.part.partName]
                    part_idx = self.part_to_idx[snote.part.partName]
                    self.G.add_nodes_from([(n_idx, {"note": snote, 
                                                "start": snote.offset, 
                                                "end":snote.endTime, 
                                                "ver": i, 
                                                'm21_id' : snote.element.id,
                                                "part": part_idx})])
                    n_idx += 1        
                print('------------------')
        self.ver_len = i + 1
        

    def assign_vertical_weight(self, u, v):
        pitch1, pitch2 = u.element, v.element
        h_int = abs(float(interval.Interval(noteStart=pitch1, noteEnd=pitch2).cents)/100)
        if h_int == 0:
            return 1000
        else:
            if h_int <= 6: # TODO: if this '6' is a tunable parameter then it should be a class variable
                return 1 * self.h_reg / h_int + self.concur_bias
            else:
                return 1 * self.h_reg / h_int

    def add_vertical_edges(self):
        for ver in range(self.ver_len):
            notes_same_vert = [x for x,y in self.G.nodes(data=True) if y['ver']==ver]
            for i in range(len(notes_same_vert)-1):
                w = self.assign_vertical_weight(self.G.nodes[notes_same_vert[i]]["note"], self.G.nodes[notes_same_vert[i+1]]["note"])
                self.G.add_edge(notes_same_vert[i], notes_same_vert[i+1], weight=w)
        
        for i in range(len(self.G.nodes)):
            for j in range(len(self.G.nodes)):
                node1, node2 = self.G.nodes[i+1], self.G.nodes[j+1]
                if node1["start"] <= node2["start"] and node1["end"] >= node2["end"]:
                    if abs(node1["part"]-node2["part"]==1):
                        print(i,j)
                        w = self.assign_vertical_weight(node1["note"], node2["note"])
                        self.G.add_edge(i, j, weight=w)

    def add_horizontal_edges(self):
        for instrument in self.part_to_idx.keys():
            notes_same_part = [x for x,y in self.G.nodes(data=True) if y['note'].part.partName==instrument]
            for i in range(len(notes_same_part)-1):
                w = self.assign_horizontal_weight(self.G.nodes[notes_same_part[i]]["note"], self.G.nodes[notes_same_part[i+1]]["note"])
                self.G.add_edge(notes_same_part[i], notes_same_part[i+1], weight=w)

    def assign_horizontal_weight(self, u, v):
        pitch1, pitch2 = u.element,v.element
        h_int = abs(float(interval.Interval(noteStart=pitch1, noteEnd=pitch2).cents)/100)
        if h_int == 0:
            return 1000
        else:
            if self.check_slur(u,v):
                return 1 * self.h_reg / h_int + self.part_bias + self.slur_bias
            else:
                return 1 * self.h_reg / h_int + self.part_bias


    def check_slur(self, u,v):
        if u.element.getSpannerSites() == [] or v.element.getSpannerSites() == []:
            return True


    def annotate_score(self, v_partition, colors = ['green', 'blue']):
        # iterate over all the notes of the score
        # make sure the number of notes is the same as the number of nodes in the graph
        # for each note, find the node with the same m21_id and assign a color accordingly
        n_notes = 0
        for part in self.score[stream.Part]:
            for elem in part[note.Note]:
                node_ind = [x for x,y in self.G.nodes(data=True) if y['m21_id'] == elem.id]
                
                if len(node_ind) > 1:
                    print('more than one node with the same m21_id')
                    print(node_ind)
                    assert(False)
                split = v_partition[node_ind[0] - 1]
                if split == -1:
                    elem.style.color = colors[0]
                elif split == 1:
                    elem.style.color = colors[1]
                n_notes += 1
        assert(n_notes == len(self.G.nodes))
    
    def save_xml(self, filename):
        self.score.write('musicxml', filename)
        
