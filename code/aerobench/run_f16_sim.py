'''
Stanley Bak
run_f16_sim python version
'''

import time

import numpy as np
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler


def run_f16_sim(initial_state, tmax, ap, step=1/30, extended_states=False, model_str='morelli',
                integrator_str='rk45', v2_integrators=False):
    '''Simulates and analyzes autonomous F-16 maneuvers

    if multiple aircraft are to be simulated at the same time,
    initial_state should be the concatenated full (including integrators) initial state.

    returns a dict with the following keys:

    'status': integration status, should be 'finished' if no errors, or 'autopilot finished'
    'times': time history
    'states': state history at each time step
    'modes': mode history at each time step

    if extended_states was True, result also includes:
    'xd_list' - derivative at each time step
    'ps_list' - ps at each time step
    'Nz_list' - Nz at each time step
    'Ny_r_list' - Ny_r at each time step
    'u_list' - input at each time step, input is 7-tuple: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    These are tuples if multiple aircraft are used
    '''

    start = time.perf_counter()

    initial_state = np.array(initial_state, dtype=float)
    llc = ap.llc

    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if initial_state.size < num_vars:
        # append integral error states to state vector
        x0 = np.zeros(num_vars)
        x0[:initial_state.shape[0]] = initial_state
    else:
        x0 = initial_state

    assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

    # run the numerical simulation
    times = [0]
    states = [x0]

    # mode can change at time 0
    ap.advance_discrete_mode(times[-1], states[-1])

    modes = [ap.mode]

    cof_alpha = -0.006225
    cof = -0.06667
    rho0 = 2.377e-3

    S_aileron = 0.3193
    cbar_aileron = 0.31

    S_elevator = 1.08
    cbar_elevator = 0.6

    S_rudder = 0.9405
    cbar_rudder = 0.73
    

    if extended_states:
        xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

        xd_list = [xd]

        u_list = [u]
        throttle_list = [u[0]]
        ele_list = [u[1]]
        ali_list = [u[2]]
        rud_list = [u[3]]

        Nz_list = [Nz]
        ps_list = [ps]
        Ny_r_list = [Ny_r]

        # 计算舵机功率
        v = states[-1][0]
        alpha = states[-1][1]
        alt = states[-1][11]
        rho_alt = rho0*(((1 - 0.703e-5*alt))**4.14)
        qbar = 0.5*47.88026247*(v**2)*rho_alt

        # 副翼
        moment_ail = (alpha*cof_alpha + ali_list[-1]*21.5*cof)*S_aileron*cbar_aileron*qbar

        # 升降舵
        moment_ele = (alpha*cof_alpha + ele_list[-1]*25*cof)*S_elevator*cbar_elevator*qbar

        # 方向舵
        moment_rud = (alpha*cof_alpha + rud_list[-1]*30*cof)*S_rudder*cbar_rudder*qbar

        moment_aileron = [moment_ail]
        moment_elevator = [moment_ele]
        moment_rudder = [moment_rud]

        power_aileron = [0]
        power_elevator = [0]
        power_rudder = [0]

    der_func = make_der_func(ap, model_str, v2_integrators)

    if integrator_str == 'rk45':
        integrator_class = RK45
        kwargs = {}
    else:
        assert integrator_str == 'euler'
        integrator_class = Euler
        kwargs = {'step': step}

    # note: fixed_step argument is unused by rk45, used with euler
    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)

    while integrator.status == 'running':
        integrator.step()

        if integrator.t >= times[-1] + step:
            dense_output = integrator.dense_output()

            while integrator.t >= times[-1] + step:
                t = times[-1] + step
                #print(f"{round(t, 2)} / {tmax}")

                aileron_prev = ali_list[-1]
                elevator_prev = ele_list[-1]
                rudder_prev = rud_list[-1]

                times.append(t)
                states.append(dense_output(t))

                updated = ap.advance_discrete_mode(times[-1], states[-1])
                modes.append(ap.mode)

                # re-run dynamics function at current state to get non-state variables
                if extended_states:
                    xd, u, Nz, ps, Ny_r = get_extended_states(ap, times[-1], states[-1], model_str, v2_integrators)

                    xd_list.append(xd)
                    u_list.append(u)
                    throttle_list.append(u[0])
                    ele_list.append(u[1])
                    ali_list.append(u[2])
                    rud_list.append(u[3])

                    Nz_list.append(Nz)
                    ps_list.append(ps)
                    Ny_r_list.append(Ny_r)

                    # 计算舵机功率
                    v = states[-1][0]
                    alpha = states[-1][1]
                    alt = states[-1][11]
                    rho_alt = rho0*(((1 - 0.703e-5*alt))**4.14)
                    qbar = 0.5*47.88026247*(v**2)*rho_alt

                    # 副翼
                    moment_ail = (alpha*cof_alpha + ali_list[-1]*21.5*cof)*S_aileron*cbar_aileron*qbar
                    d_aileron = np.pi/180*21.5*(ali_list[-1] - aileron_prev)/step
                    power_ail = moment_ail*d_aileron 

                    moment_aileron.append(moment_ail)
                    power_aileron.append(power_ail)

                    # 升降舵
                    moment_ele = (alpha*cof_alpha + ele_list[-1]*25*cof)*S_elevator*cbar_elevator*qbar
                    d_elevator = np.pi/180*21.5*(ele_list[-1] - elevator_prev)/step
                    power_ele = moment_ele*d_elevator

                    moment_elevator.append(moment_ele)
                    power_elevator.append(power_ele)

                    # 方向舵
                    moment_rud = (alpha*cof_alpha + rud_list[-1]*30*cof)*S_rudder*cbar_rudder*qbar
                    d_rudder = np.pi/180*21.5*(rud_list[-1] - rudder_prev)/step
                    power_rud = moment_rud*d_rudder

                    moment_rudder.append(moment_rud)
                    power_rudder.append(power_rud)


                if ap.is_finished(times[-1], states[-1]):
                    # this both causes the outer loop to exit and sets res['status'] appropriately
                    integrator.status = 'autopilot finished'
                    break

                if updated:
                    # re-initialize the integration class on discrete mode switches
                    integrator = integrator_class(der_func, times[-1], states[-1], tmax, **kwargs)
                    break

    assert 'finished' in integrator.status

    res = {}
    res['status'] = integrator.status
    res['times'] = times
    res['states'] = np.array(states, dtype=float)
    res['modes'] = modes

    if extended_states:
        res['xd_list'] = xd_list
        res['ps_list'] = ps_list
        res['Nz_list'] = Nz_list
        res['Ny_r_list'] = Ny_r_list
        
        res['u_list'] = u_list
        res['throttle_list'] = throttle_list
        res['ele_list'] = ele_list
        res['ali_list'] = ali_list
        res['rud_list'] = rud_list

        res['moment_aileron'] = moment_aileron
        res['power_aileron'] = power_aileron
        res['moment_elevator'] = moment_elevator
        res['power_elevator'] = power_elevator
        res['moment_rudder'] = moment_rudder
        res['power_rudder'] = power_rudder

    res['runtime'] = time.perf_counter() - start

    return res

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv
    
    return der_func

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r
