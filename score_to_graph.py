#%%
from music21 import converter, environment, stream, note
from utils import (normalized_min_cut, ScoreGraph)
import networkx as nx
import os
from pathlib import Path
os.putenv('DISPLAY', ':99.0')
#%%
us = environment.UserSettings()
us['musicxmlPath'] = '/usr/bin/mscore'
us['musescoreDirectPNGPath'] = '/usr/bin/mscore'

#%%
## Load score 
path = Path('scores')
mxl_file = path / 'MozartCut.musicxml'
score_whole = converter.parse(mxl_file, format='musicxml')
score = score_whole.measures(1, 13)

#%%
# tunable parameters
h_reg = 2.0
concur_bias = 0.4
part_bias = 0.2
slur_bias = 0.4

scoreGraph = ScoreGraph(score, h_reg, concur_bias, part_bias, slur_bias)
v_partition = normalized_min_cut(scoreGraph.G)
scoreGraph.annotate_score(v_partition, colors = ['green', 'blue'])
scoreGraph.save_xml('annotated_score.musicxml')
#%%
