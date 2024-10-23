"""
This is the checkpoint 5 code comparing the normal implementation with an RK4 implementation
The code is the same as normal but the comparisons are made in main() and a new function is added called RK4_step_forward
Check the console outputs for comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import time

g = 9.81 #m/s**2

# Sets initial values for horizontal/vertical components
def set_initial(v_initial, theta):
    rad = np.radians(theta)
    return 0,0,v_initial*np.cos(rad),v_initial*np.sin(rad)
   
# The acceleration for the current time step is calculated and returned
def acceleration(vx, vy, beta):
    v = np.sqrt(vx**2+vy**2)
    ax = -1*beta*v*vx
    ay = -1*beta*v*vy-g
    return ax, ay

# Gets acceleration for current time step, changes velocity, then changes position
def step_forward(x, y, vx, vy, beta, delta_t):
    ax, ay = acceleration(vx,vy,beta)
    vx += ax * delta_t
    vy += ay * delta_t
    x += vx * delta_t
    y += vy * delta_t
    return x, y, vx, vy

# This will look excessive with too many variable names, but it's generally how it's implemented
# Normally, variables such as vx1, vx2, vx3, vx4 would be k1, k2, k3, k4, but it's easier to keep 
# track of vertical/horizontal components this way
def RK4_step_forward(x, y, vx, vy, beta, delta_t): 
    ax, ay = acceleration(vx,vy,beta)
    vx1 = vx + ax * delta_t/2
    vy1 = vy + ay * delta_t/2
    ax1, ay1 = acceleration(vx1,vy1,beta)
    vx2 = vx + ax1 * delta_t/2
    vy2 = vy + ay1 * delta_t/2
    ax2, ay2 = acceleration(vx2,vy2,beta)
    vx3 = vx + ax2 * delta_t
    vy3 = vy + ay2 * delta_t
    ax3, ay3 = acceleration(vx3,vy3,beta)   # magic
    x += delta_t*(vx+2*vx1+2*vx2+vx3)/6
    y += delta_t*(vy+2*vy1+2*vy2+vy3)/6
    vx += delta_t*(ax+2*ax1+2*ax2+ax3)/6
    vy += delta_t*(ay+2*ay1+2*ay2+ay3)/6
    return x, y, vx, vy

def main(v_initial, theta, beta, delta_t):
    print("Trajectory for (v0, theta, beta, delta_t) = ({0}, {1}, {2}, {3})".format(v_initial, theta, beta, delta_t))

    # lists/arrays for x and y values for true values (estimate with very small time step), normal estimates, and RK4 estimates  
    x_true, y_true, x_est, y_est, x_est_rk, y_est_rk  = [0.0],[0.0],[0.0],[0.0],[0.0],[0.0]

    # Accurate estimate 
    t = 0.0
    ACC_MULT = 1000 # 1000 times more accurate than normal prediction
    x, y, vx, vy = set_initial(v_initial, theta)
    while(y>=0.0): # Iterates forward steps until the particle drops below the x-axis
        x, y, vx, vy = step_forward(x, y, vx, vy, beta, delta_t/ACC_MULT)    # <---- small time difference
        x_true.append(x) # Save data
        y_true.append(y)
        t+=delta_t/ACC_MULT # Increment time step
        
    # Normal estimate
    def normal(save=True):
        t=0.0
        x, y, vx, vy = set_initial(v_initial, theta)
        while(y>=0.0): 
            x, y, vx, vy = step_forward(x, y, vx, vy, beta, delta_t)
            if save:
                x_est.append(x)
                y_est.append(y)
            t+=delta_t 

    # RK4 estimate
    def rk4(save=True):
        t=0.0
        x, y, vx, vy = set_initial(v_initial, theta)
        while(y>=0.0): 
            x, y, vx, vy = RK4_step_forward(x, y, vx, vy, beta, delta_t)
            if save:
                x_est_rk.append(x) # Save data
                y_est_rk.append(y)
            t+=delta_t 

    normal()
    rk4()

    # Calculate average time taken for normal method
    start_time = time.perf_counter_ns()
    for _ in range(1000):
        normal(False)
    el_time = time.perf_counter_ns()-start_time
    
    
    # Calculate average time taken for rk4 method
    start_time = time.perf_counter_ns()
    for _ in range(1000):
        rk4(False)
    el_time_rk4 = time.perf_counter_ns()-start_time


    # Calculations for average error from the true values (average Euclidian distance between true and estimate points)
    av_err, av_err_rk = 0,0 
    n_points = len(x_est) 
    for i in range(n_points): 
        try: # Lazy solution to deal with slightly different amounts of points as the number of points is dependent on the accuracy
            av_err += np.sqrt((y_true[i*ACC_MULT]-y_est[i])**2 + (x_true[i*ACC_MULT]-x_est[i])**2)
            av_err_rk += np.sqrt((y_true[i*ACC_MULT]-y_est_rk[i])**2 + (x_true[i*ACC_MULT]-x_est_rk[i])**2)
        except IndexError:
            break
    av_err /= n_points    # Average distance per point from true value
    av_err_rk /= n_points

    accuracy_factor = av_err/av_err_rk              # Ratio of how much more accurate it's on average
    time_factor = el_time_rk4/el_time          # Ratio of how much longer it took on average 
    acc_time_ratio = accuracy_factor/time_factor    # Accuracy/time comparison
    print("Average accuracy:", accuracy_factor, "times more accurate.")
    print("Average time:", time_factor, "times slower.")
    print("Average accuracy/time ratio:", acc_time_ratio, "times higher.")

    plt.plot(x_true,y_true,"g",label="True plot",zorder=3,linewidth=3)                          # Plot true values in green
    plt.plot(x_est,y_est,"b",label="Normal estimate",linestyle="dashed",marker="o")             # Plot normal estimate in blue
    plt.plot(x_est_rk,y_est_rk,"r",label="RK4 estimate",linestyle="dashed",marker="o")          # Plot RK4 estimate in red
    plt.axhline(y=0.0, color='grey', linestyle='dashed')            
    plt.legend(loc="upper right")
    plt.title("Trajectory of particle")     # Add title/lables
    plt.xlabel("x-distance/m")
    plt.ylabel("y-height/m")
    plt.grid()
    plt.show()  
    
# Get initial conditions
if __name__=="__main__":
    # v_initial = float(input("v0 (initial velocity) : "))
    # theta = float(input("theta (angle from the horizontal) : "))
    # beta = float(input("beta (drag coefficient) : "))
    # delta_t = float(input("timestep (delta_t) : "))
    # main(v_initial, theta, beta, delta_t) # Uncomment to input values. 

    # Don't use delta_t values that are too large, things get messed up on a small n. of points
    main(20, 60, 0.2, 0.05) 

