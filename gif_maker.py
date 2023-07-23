from cProfile import label
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from numba import njit,jit
from tqdm import tqdm
import matplotlib.ticker as mtick
import imageio
import os
import sys
import copy

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams['font.size'] = '13'

def check_coevolution(node_list, state, g, coev_prob):
    
    infected_nodes = node_list[state == 'I']
    i_count = infected_nodes.shape[0]
    
    non_infected_nodes = node_list[state != 'I']
    ni_count = non_infected_nodes.shape[0]
    
    if (i_count > 0) & (ni_count > 1):
        
        for i_node in infected_nodes:
            
            neighbours = np.array(g[i_node])
            
            if neighbours.shape[0] > 0:
                
                ni_neighbours = neighbours[state[neighbours] != 'I']
            
                if ni_neighbours.shape[0] > 0:
                    
                    coevolution_prob = np.random.random_sample(size = ni_neighbours.shape[0])
                    rewiring_nodes = ni_neighbours[coevolution_prob <= coev_prob]
                    
                    if rewiring_nodes.shape[0] > 0:
                        
                        edges_to_remove = [(r_node, i_node) for r_node in rewiring_nodes]
                        g.remove_edges_from(edges_to_remove)
                    
                        for rew_node in rewiring_nodes:
                            
                            new_ni_node = np.random.choice(non_infected_nodes, 1)[0]
                            
                            while(new_ni_node == rew_node):
                                new_ni_node = np.random.choice(non_infected_nodes, 1)[0]
                            
                            g.add_edge(rew_node, new_ni_node)

def check_recovery_to_susceptible(node_list, state, gamma):
    
    recovery_nodes = node_list[state == 'R']
    r_count = recovery_nodes.shape[0]
    
    if r_count > 0:
    
        rs_prob = np.random.random_sample(size = r_count)
        rs_nodes = recovery_nodes[rs_prob <= gamma]
        state[rs_nodes] = 'S'

def check_recovery(node_list, state, alpha):
    
    infected_nodes = node_list[state == 'I']
    i_count = infected_nodes.shape[0]
    
    if i_count > 0:
        
        recovery_prob = np.random.random_sample(size = i_count)
        recovered_nodes = infected_nodes[recovery_prob <= alpha]
        state[recovered_nodes] = 'R'

def check_infections(node_list, g, state, beta):
    
    infected_nodes = node_list[state == 'I']
    susceptible_nodes = node_list[state == 'S']
    
    i_count = infected_nodes.shape[0]
    s_count = susceptible_nodes.shape[0]
    
    if (i_count > 0) and (s_count != 0):
        
        for i_node in infected_nodes:
            
            neighbours = np.array(g[i_node])
            
            if neighbours.shape[0] > 0:
                
                s_neighbours = neighbours[state[neighbours] == 'S']
                
                s_count = s_neighbours.shape[0]
                
                if s_count > 0:
                    
                    infection_prob = np.random.random_sample(size = s_count)
                    new_infected_nodes = s_neighbours[infection_prob <= beta]
                    state[new_infected_nodes] = 'I'

def symulacja(node_list, g, state, alpha, beta, gamma, coev_prob, T):
    
    for i in range(1, T):
        
        check_recovery_to_susceptible(node_list, state, gamma)
        check_recovery(node_list, state, alpha)
        check_infections(node_list, g, state, beta)
        check_coevolution(node_list, state, g, coev_prob)
        

N = 100
p = 0.08
m = 4

p0_infect = 0.1

T = 60

alpha = 0.1 # prawd. wyzdrowienia
beta = 0.1 # prawd. zarażenia
gamma = 0.05 # prawd. R -> S
coev_prob = 0.02 # prawd. koewolucji

t = np.arange(T)

states = np.array(['S', 'I', 'R'])
color_state_map = {'S': 'blue', 'I': 'red', 'R': 'green'}

g = nx.barabasi_albert_graph(N, m)
#g = nx.erdos_renyi_graph(N, p)

g_degrees = np.array(list(dict(g.degree).values()))

node_list = np.array(g.nodes())

#state = np.random.choice(states[:-1], N, p=[1- p0_infect, p0_infect])

state = np.full(N, 'S')

starter_nodes = node_list[g_degrees <= g_degrees.mean()]
state[starter_nodes] = np.random.choice(states[:-1], starter_nodes.shape[0], p=[1- p0_infect, p0_infect])

# R0 = beta * (Counter(state)['S']/N) / alpha

# print(f'{R0 = }')

state_dict = dict(zip(node_list, state))

k_sorted_before = sorted(Counter(g_degrees).items())
k_before = [i[0] for i in k_sorted_before]
p_k_before = [i[1]/N for i in k_sorted_before]

pos = nx.spring_layout(g)

fname = f'{0}.png'
dirname = 'semestr2/MPS/projekt1/shots/'

nx.draw(g, pos=pos, with_labels=True, font_weight='bold', font_color='white', node_color=[color_state_map[s] 
                    for s in state], nodelist=node_list, node_size=[d * 75 for d in g_degrees], alpha=0.8, edge_color='grey', labels = state_dict)

plt.savefig(dirname + fname)
plt.close()

filenames = []

filenames.append(fname)

s_list = np.zeros(T)
i_list = np.zeros(T)
r_list = np.zeros(T)

s_list[0] = Counter(state)['S']
i_list[0] = Counter(state)['I']
r_list[0] = Counter(state)['R']

for i in tqdm(range(1, T)):
    
    check_recovery_to_susceptible(node_list, state, gamma)
    check_recovery(node_list, state, alpha)
    check_infections(node_list, g, state, beta)
    check_coevolution(node_list, state, g, coev_prob)
    
    s_list[i] = Counter(state)['S']
    i_list[i] = Counter(state)['I']
    r_list[i] = Counter(state)['R']
    
    state_dict = dict(zip(node_list, state))
    
    fname = f'{i}.png'
    
    g_degrees = dict(g.degree).values()
    
    nx.draw(g, pos=pos, with_labels=True, font_weight='bold', font_color='white', node_color=[color_state_map[s] 
                    for s in state], nodelist=node_list, node_size=[d * 75 for d in g_degrees], alpha=0.8, edge_color='grey', labels = state_dict)
    
    plt.savefig(dirname + fname)
    plt.close()
    
    filenames.append(fname)


k_sorted_after = sorted(Counter(g_degrees).items())
k_after = [i[0] for i in k_sorted_after]
p_k_after = [i[1]/N for i in k_sorted_after]

plt.plot(k_before, p_k_before, label = 'przed koewolucją')
plt.plot(k_after, p_k_after, label = 'po koewolucji')
plt.title('Rozkład stopni wierzchołków sieci')
plt.xlabel(r'$k$')
plt.ylabel(r"$P(k)$")
plt.legend()
plt.savefig('semestr2/MPS/projekt1/p_k.png')
plt.close()

print('Preparing .gif file ...')

with imageio.get_writer('semestr2/MPS/projekt1/sir.gif', mode='I', fps = 2) as writer:
    for filename in filenames:
        image = imageio.imread(dirname + filename)
        writer.append_data(image)

print('Deleting .png files ...')

for filename in set(filenames):
    os.remove(dirname + filename)

print('Preparing graph ...')

s_list_total = s_list/N
i_list_total = i_list/N
r_list_total = r_list/N
   
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(t, s_list_total, label = 'S', color='blue')
ax.plot(t, i_list_total, label = 'I', color='red')
ax.plot(t, r_list_total, label = 'R', color='green')

ax.set_yticks(np.linspace(0, 1, 11))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.legend()
plt.title(f'Względna ilość osób w każdej grupie modelu SIR')
plt.xlabel('t [krok]')
#plt.show()

plt.savefig('semestr2/MPS/projekt1/sir.png')
plt.close()