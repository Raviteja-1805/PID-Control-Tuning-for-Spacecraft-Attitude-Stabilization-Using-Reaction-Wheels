# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:17:45 2026

@author: Teja3
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(10,3))

plt.text(0.1, 0.5, 'qd\n(Desired)', bbox=dict(facecolor='lightblue'))
plt.text(0.3, 0.5, 'Error\n(qe)', bbox=dict(facecolor='lightgreen'))
plt.text(0.5, 0.5, 'PID\nController', bbox=dict(facecolor='orange'))
plt.text(0.7, 0.5, 'Dynamics\n(I, ω)', bbox=dict(facecolor='pink'))
plt.text(0.9, 0.5, 'q\n(Output)', bbox=dict(facecolor='lightblue'))

# Arrows
plt.arrow(0.15, 0.5, 0.1, 0, head_width=0.03)
plt.arrow(0.35, 0.5, 0.1, 0, head_width=0.03)
plt.arrow(0.55, 0.5, 0.1, 0, head_width=0.03)
plt.arrow(0.75, 0.5, 0.1, 0, head_width=0.03)

# Feedback loop
plt.arrow(0.9, 0.4, -0.7, 0, head_width=0.02)
plt.text(0.45, 0.35, 'Feedback (q)', ha='center')

plt.axis('off')
plt.title("Spacecraft Attitude Control Block Diagram")

plt.show()