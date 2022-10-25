
c = 2.998e8  # speed of light in m/s
fr_c = c/0.02998



swarm4a = {
    "name": "DJI_Mavic_Air_2",
    "N": 2,
    "L_1": 0.005,
    "L_2": 0.07,
    # https://mavicpilots.com/threads/mavic-average-rpm.4982/
    "f_rot": 91.66,
    "swarm_amount" : 4,
    "RD" : [0,0,0,10],
    "Thet": [0.0,0.01,0.02,0.015]
}



djimatrice300rtk = {
    # https://www.bhphotovideo.com/c/product/1565975-REG/dji_cp_en_00000270_01_matrice_300_series_propeller.html
    "name": "DJI_Matrice_300_RTK",
    "N": 2,  # two blades per rotor
    "L_1": 0.05,
    "L_2": 0.2665,
    "f_rot": 70,  # a guess
    "swarm_amount" : 1,
    "RD" : 0.0,
    "Thet": 0.0
}



djiphantom4 = {
    # https://store.dji.com/ca/product/phantom-4-series-low-noise-propellers
    "name": "DJI_Phantom_4",
    "N": 2,  # two blades per rotor
    "L_1": 0.006,
    "L_2": 0.05,
    # https://phantompilots.com/threads/motor-rpm.16886/
    "f_rot": 116,
    "swarm_amount" : 1,
    "RD" : 0.0,
    "Thet": 0.0
}



noise = {
    "name": "noise",
    "N": 2,  # two blades per rotor
    "L_1": 0.0,
    "L_2": 0.0,
    "f_rot": 0.0,
    "swarm_amount" : 1,
    "RD" : 0.0,
    "Thet": 0.0
}



swarm4 = {
    # https://store.dji.com/ca/product/phantom-4-series-low-noise-propellers
    "name": "DJI_Phantom_4",
    "N": 2,  # two blades per rotor
    "L_1": 0.006,
    "L_2": 0.05,
    # https://phantompilots.com/threads/motor-rpm.16886/
    "f_rot": 116,
    "swarm_amount" : 4,
    "RD" : [0,0,0,10],
    "Thet": [0.0,0.01,0.02,0.015]
}



swarm8 = {
    # https://store.dji.com/ca/product/phantom-4-series-low-noise-propellers
    "name": "DJI_Phantom_4",
    "N": 2,  # two blades per rotor
    "L_1": 0.006,
    "L_2": 0.05,
    # https://phantompilots.com/threads/motor-rpm.16886/
    "f_rot": 116,
    "swarm_amount" :8,
    "RD" : [0,0,0,0,10,10,10,10],
    "Thet": [0.0, 0.01, 0.02, 0.03, 0.05, 0.015, 0.025, 0.035]
}



drones1 = [swarm4, swarm8, djiphantom4, djimatrice300rtk, swarm4a, noise]

class_map1 = ["swarm4","swarm8","djiphantom4","djimatrice300rtk","swarm4a","noise"]









