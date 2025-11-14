
📈 Detailed Classification Report:
================================================================================
              precision    recall  f1-score   support

    airplane     0.9470    0.9290    0.9379      1000
  automobile     0.9644    0.9760    0.9702      1000
        bird     0.9110    0.9010    0.9060      1000
         cat     0.8526    0.8330    0.8427      1000
        deer     0.9347    0.9300    0.9323      1000
         dog     0.8550    0.8670    0.8610      1000
        frog     0.9374    0.9590    0.9481      1000
       horse     0.9429    0.9570    0.9499      1000
        ship     0.9661    0.9690    0.9675      1000
       truck     0.9637    0.9550    0.9593      1000

    accuracy                         0.9276     10000
   macro avg     0.9275    0.9276    0.9275     10000
weighted avg     0.9275    0.9276    0.9275     10000



🎯 Per-class Accuracy Analysis (Ranked):
================================================================================
Rank Class        Accuracy Count        Status    
------------------------------------------------------------
1    automobile   0.9760    976/1000     EXCELLENT 🏆
2    ship         0.9690    969/1000     EXCELLENT 🏆
3    frog         0.9590    959/1000     EXCELLENT 🏆
4    horse        0.9570    957/1000     EXCELLENT 🏆
5    truck        0.9550    955/1000     EXCELLENT 🏆
6    deer         0.9300    930/1000     VERY GOOD ✅
7    airplane     0.9290    929/1000     VERY GOOD ✅
8    bird         0.9010    901/1000     VERY GOOD ✅
9    dog          0.8670    867/1000     GOOD 👍
10   cat          0.8330    833/1000     FAIR ⚠️



📊 Comprehensive Model Statistics:
================================================================================
Best Class:    automobile   0.9760
Worst Class:   cat          0.8330
Average Class:              0.9276
Std Deviation:              0.0447
Accuracy Range:             0.1430



🔍 Detailed Prediction Analysis:
============================================================
Correct predictions analyzed: 12
Incorrect predictions analyzed: 12
High-confidence errors: 8
Low-confidence correct: 6
Avg confidence on correct: 0.812 (±0.026)
Avg confidence on incorrect: 0.559 (±0.154)



📊 VAN-B0 Advanced Confidence Analysis:
==================================================
Total predictions: 10000
Correct predictions: 9276
Incorrect predictions: 724
Mean confidence (correct): 0.7960 (±0.0765)
Mean confidence (incorrect): 0.5895 (±0.1546)
Confidence gap: 0.2065

🎉 VAN-B0 Evaluation Complete!
📈 Final Test Accuracy: 0.9276 (92.76%)
