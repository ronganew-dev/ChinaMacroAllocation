"""
Backtest All Cycle Stratage program
By Hujun Tang, 2026.04.12
"""

import models
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# macro allocation strategy
print('Load macro allocation parameter')
all_cycle_model = models.allCycle.AllCycleModel()

print('Load input data')
all_cycle_model.load_input()

print('Calculate macro cycle signal')
all_cycle_model.calc_signal()

print('Calculate VCV')
all_cycle_model.calc_vcv()

print('Calculate model weight')
all_cycle_model.calc_model_wt()

print('Calculate portfolio weight')
all_cycle_model.calc_port_wt()

print('Calculate portfolio performance')
all_cycle_model.calc_performance()

print('Output result')
all_cycle_model.output_result()


