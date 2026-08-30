'''
用figure4計算論文的指標
輸出: 
without BPM: 
    IoU =  0.4884393063583815
    Dice =  0.6563106796116505
    Precision =  0.66015625
    Recall =  0.6525096525096525

with BPM: 
    IoU =  0.8717948717948718
    Dice =  0.9315068493150684
    Precision =  0.9066666666666666
    Recall =  0.9577464788732394
'''
import numpy as np

print("without BPM: ")
TP=169
TN=678
FP=87
FN=90
print("IoU = ", TP/(TP+FP+FN))
print("Dice = ", 2*TP/(2*TP+FP+FN))
print("Precision = ", TP / (TP + FP))
print("Recall = ", TP / (TP + FN))

print("with BPM: ")
TP=68
TN=946
FP=7
FN=3
print("IoU = ", TP/(TP+FP+FN))
print("Dice = ", 2*TP/(2*TP+FP+FN))
print("Precision = ", TP / (TP + FP))
print("Recall = ", TP / (TP + FN))