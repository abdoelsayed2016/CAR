#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, glob, codecs
import json
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz


# In[ ]:





# In[ ]:





# In[ ]:


def symbols_calc_Rasul(all_files): 
    symbols = {}
    words = {}
    for file_id, filename in enumerate(all_files):
        #if file_id < 149:
        #    continue
        with codecs.open(filename,'r', encoding="utf-8") as df:
            data = json.load(df)
            description = data['description']
            predicted = data['moderation']['predicted']
        #print(file_id, filename, description, '-', predicted)

        for ch in description:
            if not ch in symbols:
                symbols[ch] = {'freq': 1, 'pred': 0, 'acc': 0}
            else:
                symbols[ch]['freq'] += 1
        correct_chars = []
        correct_chars_sum = 0
        for outerindex in range(len(description)):
            try:
                match_cnt = 0
                for i1 in range(outerindex, len(description)):
                    ind2 = max(0, i1-2)
                    ind2plus3 = ind2 + 4
                    while ind2 < min(ind2plus3, len(predicted)):
                        if (description[i1] == predicted[ind2]):
                            match_cnt += 1
                            ind2 += 1
                            ind2plus3 = ind2 + 4
                            if match_cnt > correct_chars_sum:
                                correct_chars_sum = match_cnt
                                #print(description[i1], correct_chars_sum)
                                symbols[description[i1]]['pred'] += 1
                                correct_chars.append(description[i1])
                            break
                        else:
                            ind2 += 1
            except Exception as ex:
                print(ex)
                pass

        incorrect_predicted_chars = []                                                         
        for p_ch in predicted:
            if not p_ch in correct_chars:
                incorrect_predicted_chars.append(p_ch)
                                                                 
        for ex_ch in incorrect_predicted_chars:
            if not ex_ch in symbols:
                symbols[ex_ch] = {'freq': 1, 'pred': 0, 'acc': 0}
            else:
                symbols[ex_ch]['freq'] += 1
            
        words[filename] = {'desc': description, 'pred': predicted, 
                       'Lev': fuzz.ratio(description, predicted),
                       'Our': round((correct_chars_sum/(len(description) + len(incorrect_predicted_chars)))*100),
                       'delta': abs(fuzz.ratio(description, predicted) - round((correct_chars_sum/(len(description) + len(incorrect_predicted_chars)))*100))}
        #if file_id > 40:
        #    break
    l = 0
    for key, val in symbols.items():
        val['acc'] = round((val['pred']/val['freq'])*100)
        l += val['freq']
    
    symbols = OrderedDict(sorted(symbols.items(), key = lambda x : x[1]['freq'], reverse=False))
    return symbols, words


# In[ ]:


# Word accuracy rate
def words_calc(all_files, model_name):   
   correct_words = 0
   car_sum = 0
   for file_id, filename in enumerate(all_files):
       with codecs.open(filename,'r', encoding="utf-8") as df:
           data = json.load(df)
           description = data['description']
           predicted = data['moderation']['predicted']
       # equal words
       if description == predicted:
           correct_words +=1
       # Levenshtein distance
       car_sum += fuzz.ratio(description, predicted)
   print(model_name + " WAR:" , round(correct_words/len(all_files) * 10000)/100, "%")
   print(model_name + " CAR: ", round(car_sum/len(all_files) * 100)/100, "%")


# In[ ]:


model_name = 'Flor Model (modified)'
TEST1_DIR = model_name + "/test1/ann_pred"
TEST2_DIR = model_name + "/test2/ann_pred"

#test1_symbols, test1_words = symbols_calc(glob.glob(os.path.join(TEST1_DIR,"*.json")))
#test2_symbols, test2_words = symbols_calc(glob.glob(os.path.join(TEST2_DIR,"*.json")))
#words_calc(glob.glob(os.path.join(TEST1_DIR,"*.json")), model_name)
test1_symbols, test1_words = symbols_calc_Rasul(glob.glob(os.path.join(TEST1_DIR,"*.json")))
test2_symbols, test2_words = symbols_calc_Rasul(glob.glob(os.path.join(TEST2_DIR,"*.json")))
words_calc(glob.glob(os.path.join(TEST1_DIR,"*.json")), model_name)


# In[ ]:


print(model_name)
sorted_words = OrderedDict(sorted(test1_words.items(), key = lambda x : x[1]['delta'], reverse=True))
i = 0
for name, value in sorted_words.items():
    print(value)
    i += 1
    if i > 10:
        break


# In[ ]:


test1_symbols_freq = [x['freq'] for x in test1_symbols.values()]
test1_symbols_pred = [x['pred'] for x in test1_symbols.values()]
test1_symbols_acc = [x['acc'] for x in test1_symbols.values()]
test1_avg_acc = round(sum(test1_symbols_acc)/len(test1_symbols_acc)*100)/100
print(test1_avg_acc)


# In[ ]:


test2_symbols_freq = [x['freq'] for x in test2_symbols.values()]
test2_symbols_pred = [x['pred'] for x in test2_symbols.values()]
test2_symbols_acc = [x['acc'] for x in test2_symbols.values()]
test2_avg_acc = round(sum(test2_symbols_acc)/len(test2_symbols_acc)*100)/100
print(test2_avg_acc)


# In[ ]:


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,20), sharey=False)
ax1.text(2750, 85, "Character Accuracy Rates (CAR), " + model_name, fontsize=24)

#AX1 axis
ax1.text(2000, 80, "on TEST1", fontsize=24)
ax1.text(2000, 40, "Average CAR: " + str(test1_avg_acc) + "%", fontsize=24)
width = 0.7
plt.rcParams['font.size'] = 12
ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
ax1.set_xticks(np.linspace(0, 5500, 5))
ax1.set_xlabel('Number of characters', fontsize = 20)
ax1.set_ylabel('Characters', fontsize = 20)
rects = ax1.barh(list(test1_symbols.keys()), test1_symbols_freq, width, color = 'r', log = False, label='number of annotation characters')
rects_acc = ax1.barh(list(test1_symbols.keys()), test1_symbols_pred, width, color = 'g', log = False, label='number of predicted characters')
ax1.legend(loc="lower right", fontsize = 20)
for i, rect in enumerate(rects):
    yloc = rect.get_y() + rect.get_height() / 2
    xloc = -5
    width = int(rect.get_width())
    # The bars aren't wide enough to print the ranking inside
    if width < 999999:
        # Shift the text to the right side of the right edge
        xloc = 2
        # Black against white background
        clr = 'black'
        align = 'left'
    else:
        # Shift the text to the left side of the right edge
        xloc = -5
        # White on magenta
        clr = 'white'
        align = 'right'
    label = ax1.annotate(str(test1_symbols_acc[i])+'%', xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, clip_on=True, fontsize=12)
    for i in range(len(test1_symbols_freq)):
        if list(test1_symbols.keys())[i] in ['а','я', 'А', 'Ж','щ','э','Қ','ғ']:
            ax1.text(-550 , i-0.25, str(list(test1_symbols.keys())[i]), color='black', fontsize='24')
  

# AX2 axis
ax2.text(2000, 80, "on TEST2", fontsize=24)
ax2.text(2000, 40, "Average CAR: " + str(test2_avg_acc) + "%", fontsize=24)
width = 0.7
plt.rcParams['font.size'] = 13
ax2.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
ax2.set_xticks(np.linspace(0, 5500, 5))
ax2.set_xlabel('Number of characters', fontsize = 20)
ax2.set_ylabel('Characters', fontsize = 20)
rects = ax2.barh(list(test2_symbols.keys()), test2_symbols_freq, width, color = 'r', log = False, label='number of annotation characters')
rects_acc = ax2.barh(list(test2_symbols.keys()), test2_symbols_pred, width, color = 'g', log = False, label='number of predicted characters')
ax2.legend(loc="lower right", fontsize = 20)
for i, rect in enumerate(rects):
    yloc = rect.get_y() + rect.get_height() / 2
    xloc = -5
    width = int(rect.get_width())
    # The bars aren't wide enough to print the ranking inside
    if width < 999999:
        # Shift the text to the right side of the right edge
        xloc = 2
        # Black against white background
        clr = 'black'
        align = 'left'
    else:
        # Shift the text to the left side of the right edge
        xloc = -5
        # White on magenta
        clr = 'white'
        align = 'right'
    label = ax2.annotate(str(test2_symbols_acc[i])+'%', xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, clip_on=True, fontsize=12)
    for i in range(len(test2_symbols_freq)):
        if list(test2_symbols.keys())[i] in ['а','я', 'А', 'Ж','щ','э','Қ','ғ']:
            ax2.text(-700 , i, str(list(test2_symbols.keys())[i]), color='black', fontsize='24')
    
#plt.show()
plt.savefig('CAR2_'+model_name+'.png', size=6, transparent=False, bbox_inches='tight', pad_inches=0, dpi=300)


# In[ ]:




