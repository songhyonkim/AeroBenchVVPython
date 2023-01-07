import numpy as np

def sphere(center_pos, threat_radius):

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    x = threat_radius*np.outer(np.cos(u), np.sin(v)) + center_pos[0]
    y = threat_radius*np.outer(np.sin(u), np.sin(v)) + center_pos[1]
    z = threat_radius*np.outer(np.ones(np.size(u)), np.cos(v)) + center_pos[2]
    
    return [x, y, z]