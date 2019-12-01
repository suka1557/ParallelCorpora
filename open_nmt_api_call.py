import os, sys
import numpy as np
import pandas as pd
import subprocess

input_file = 'input.txt'
output_file = 'output.txt'

if os.path.exists(input_file):
	os.remove(input_file)
if os.path.exists(output_file):
	os.remove(output_file)
	
english_sentence = Input('Enter text: ')

f = open(input_file, 'w+')
f.write(english_sentence)
f.close()

subprocess.call(["onmt_translate ", "-model " , "open_nmt_default_model_step_100000.pt " , "-src " , "input.txt " , "-output " , "output.txt ", "-replace_unk ", "-verbose " ])

of = open(output_file, 'r+')
spanish_sentence = f.read()
print('Translated Sentence: ', spanish_sentence)

if os.path.exists(input_file):
	os.remove(input_file)
if os.path.exists(output_file):
	os.remove(output_file)