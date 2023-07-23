from turtle import title
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams['font.size'] = '13'

p0_inf = 0.1

T = 100
alpha = 0.1
beta = 0.5

t = np.arange(T)

def przelicz_sir(p0_inf, T, alpha, beta):
    
    s_list_model = np.zeros(T)
    i_list_model = np.zeros(T)
    r_list_model = np.zeros(T)

    s_list_model[0] = (1-p0_inf)
    i_list_model[0] = p0_inf

    R0 = beta * s_list_model[0] / alpha

    print(f'{R0 = }')

    for i in range(1, T):
        
        s_next = s_list_model[i-1] - beta * s_list_model[i-1] * i_list_model[i-1]
        
        if 0 <= s_next <= 1:
            s_list_model[i] = s_next
        elif s_next > 1:
            s_list_model[i] = 1
        else:
            s_list_model[i] = 0
            
        i_next = i_list_model[i-1] + beta * s_list_model[i-1] * i_list_model[i-1] - alpha * i_list_model[i-1]
        
        if 0 <= i_next <= 1:
            i_list_model[i] = i_next
        elif i_next > 1:
            i_list_model[i] = 1
        else:
            i_list_model[i] = 0
            
        r_next = r_list_model[i-1] + alpha * i_list_model[i-1]
        
        if 0 <= r_next <= 1:
            r_list_model[i] = r_next
        elif r_next > 1:
            r_list_model[i] = 1
        else:
            r_list_model[i] = 0
        
    return R0, s_list_model, i_list_model, r_list_model
 

R0, s_list_model, i_list_model, r_list_model = przelicz_sir(p0_inf, T, alpha, beta) 
     
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(t, s_list_model, label = 'S - model', color='blue', linestyle='dashed')
ax.plot(t, i_list_model, label = 'I - model', color='red', linestyle='dashed')
ax.plot(t, r_list_model, label = 'R - model', color='green', linestyle='dashed')
ax.set_yticks(np.linspace(0, 1, 11))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.legend()
plt.title(f'Względna ilość osób w każdej grupie modelu SIR dla {R0 = }')
plt.xlabel('t [krok]')
plt.show()