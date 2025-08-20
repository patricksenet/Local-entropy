# +
# 
# Program to compute the graph properties
# of an ensemble of structures of a polymer in the .xyz format
#
# Add a local analysis of a graph around a given node July 2024
#
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import pyplot as plt, cm, colors, ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
import numpy as np
import networkx as nx
import pickle
from numpy import linalg as LinA
import math 
from scipy import stats
from scipy.stats import gaussian_kde
from collections import defaultdict
import hashlib
#
pi=4.0*math.atan(1.0)
#
#  Generate a unique hash for a graph G based on its adjacency matrix.
#
def graph_hash_sha256(G):
    # Obtenir la matrice d'adjacence en tant que numpy array
    adj_matrix = nx.adjacency_matrix(G).toarray()
    # Convertir la matrice en bytes
    adj_bytes = adj_matrix.tobytes()
    # Calculer le hash SHA-256 des bytes de la matrice d'adjacence
    return hashlib.sha256(adj_bytes).hexdigest()
#
# Generate a subgraph starting from a node to a certain distance
#
#
# Generate a subgraph starting from a node to a distance 1
#
def generate_subgraph1(graph, start_node):
    G1 = nx.Graph()
    adj_matrix = nx.adjacency_matrix(graph)
    adj_matrix_dense = adj_matrix.todense()
    shape = adj_matrix_dense.shape[0]
    noeuds_1 = []
    # Première boucle pour trouver les voisins directs
    for k in range(shape):
        if adj_matrix_dense[start_node, k] == 1:
            G1.add_edge(start_node, k)
            noeuds_1.append(k)
    # Deuxième boucle pour trouver les arêtes entre les voisins directs
    for node1 in noeuds_1:
        for node2 in noeuds_1:
            if node1 != node2 and adj_matrix_dense[node1, node2] == 1:
                G1.add_edge(node1, node2)
                hello=1
    return G1, noeuds_1
#
# Generate a subgraph starting from a node to a distance 1 & 2
#
def generate_subgraph12(graph, start_node):
    G1 = nx.Graph()
    G2 = nx.Graph()
    adj_matrix = nx.adjacency_matrix(graph)
    adj_matrix_dense = adj_matrix.todense()
    shape = adj_matrix_dense.shape[0]
    noeuds_1 = []
    noeuds_2 = []
    # Première boucle pour trouver les voisins directs
    for k in range(shape):
        if adj_matrix_dense[start_node, k] == 1:
            G1.add_edge(start_node, k)
            noeuds_1.append(k)
    # Deuxième boucle pour trouver les arêtes entre les voisins directs
    for node1 in noeuds_1:
        for node2 in noeuds_1:
            if node1 != node2 and adj_matrix_dense[node1, node2] == 1:
                G1.add_edge(node1, node2)
                hello=1
    # rechercher les voisins des voisins
    for k in range(shape):
        for nodes in noeuds_1:
            if adj_matrix_dense[nodes, k] == 1:
                G2.add_edge(nodes, k)
                noeuds_2.append(k)
    # Deuxième boucle pour trouver les arêtes entre les seconds voisins 
    for node1 in noeuds_2:
        for node2 in noeuds_2:
            if node1 != node2 and adj_matrix_dense[node1, node2] == 1:
                G2.add_edge(node1, node2)
                hello=2
    return G2, noeuds_1, noeuds_2
#
# Couleur des arêtes - chaine (noir) ou contact (rouge)
#
def edge_attributes(G):
    edge_colors = []
    edge_styles = []
    edge_widths = []
    nblack = 0
    ndotted = 0
    nred =0
    for u, v in G.edges():
        if (u == v - 1) or (u == v + 1):
            edge_colors.append('black')
            edge_styles.append('solid')
            edge_widths.append(2.5)  # Plus épais
            nblack +=1
        elif (u == v - 2) or (u == v + 2):
            edge_colors.append('red')
            edge_styles.append('dotted')
            edge_widths.append(1.0)  # Epaisseur intermédiaire
            ndotted +=1
        else:
            edge_colors.append('red')
            edge_styles.append('solid')
            edge_widths.append(2.0)  # Moins épais
            nred +=1
    return edge_colors, edge_styles, edge_widths, nblack,ndotted,nred
#
# Calculate the Kirchhoff index (total resistance) and K of a graph G.
#
def kirchhoff_index(G):
    L = nx.laplacian_matrix(G).todense()
    n = L.shape[0]
    if n == 0:
        return 0
    L_pinv = np.linalg.pinv(L)
    kirchhoff_idx = n * np.trace(L_pinv)
    constant_K = n/kirchhoff_idx
    constant_Kref = 6.0/(n**2-1)
    constant_Krel = constant_K/constant_Kref
    return kirchhoff_idx, constant_K, constant_Krel
#
# FUNCTION compute the graph of the structures
#
def graph(Y_raw,rcontact):
# Compute the contact matrix 
    A=((Y_raw < rcontact) & (Y_raw > 0.0).astype(int))
# Graph corresponding to A
#      G=nx.from_numpy_matrix(A)
#      G=nx.Digraph(A)
    G=nx.from_numpy_array(A)
# Average shortest path length
    xhi=nx.average_shortest_path_length(G)
# Laplacian
    AA=np.sum(A,axis=1)  
    AA=np.diag(AA)
    LA=AA-A
    contact1=np.trace(LA)
    eigenvalues, eigenvectors = LinA.eigh(LA)
    eigenvalues=np.sort(np.array(eigenvalues))
    modes=np.real(eigenvalues[1:])
    contact2=np.sum(modes)
    ctest=contact2-contact1
    ctest=np.abs(ctest)
    if ctest > 0.001:
        print('Trace error for contact number')
    modes=1/modes
    K=np.sum(modes)
    K=1/K
    return K,xhi,G
#
# FUNCTION read the XYZ coordinates, compute the distances
#
def read_xyz_file(file_path,rcontact):
    structures = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
# Number of atoms
    num_atoms = int(lines[0].strip())
# Initialisation of atom coordinates of a frame
    structure_coordinates = []
# Initialisation of the graph properties
    graph_properties = []
# Initialisation of the graph list
    graphs = []
# Initialisation for the model number
    nmodel=0
#
# Loop on all frames/models
#
    for line in lines[2:]:
        if line.strip():  # Verrifie si la ligne n'est pas vide
            atom_data = line.split()
            label = str(atom_data[0])
            if len(atom_data) == 4 and label == 'CA':  
             x, y, z = map(float, atom_data[1:4])
# Coordinates of each frame/model
             structure_coordinates.append([x, y, z])
             if len(structure_coordinates) == num_atoms:
                 structures.append(structure_coordinates)
# Compute the distance matrix between the atoms of the model
                 Y = distance.cdist(structure_coordinates, structure_coordinates, 'euclidean')
                 Y = np.array(Y)
# CALL THE FUNCTION  - Compute the graph 
                 K, xhi, G = graph(Y,rcontact) 
                 data_graph=(nmodel,K,xhi)
# Save the graph properties in a list graph_properties
                 graph_properties.append(data_graph)
# Save the graph itself
                 graphs.append(G)
# Reinitialize the list of coordinates of the frame and the model number
                 structure_coordinates = []
                 nmodel=nmodel+1
#                 print('test',nmodel,num_atoms)
    return graphs, graph_properties,num_atoms,nmodel
#
# MAIN PROGRAM
#
# Smallest distance to define the polymer as a SAW
dcut = 4.0 # for information only
# Maximum distance to define a link in the graph 
rcontact=2*dcut
#rcontact=6.0
# Cut-off for the selection of hardest structures (K >=Khard)
Khard=0.001
print('dcut=',dcut,' rcontact=',rcontact,' Khard=',Khard)
# INPUT XYZ FILE
xyz_file_path = 'input.xyz'
xyz_file_path= '/Users/psenet/Desktop/Graph-related/ACS-central-science/ChemSci-version/all-files/Local-entropy-code/input.xyz'
file_path='./'
#
# CALL THE FUNCTION
#
graphs_full,graph_list,num_atoms, nmodel= read_xyz_file(xyz_file_path,rcontact)
print('File read ',xyz_file_path,' Number of frames=',nmodel,' Number of atoms=',num_atoms)
#
# GRAPH PROPERTIES FOR EACH FRAME
#
with open(file_path+'graph_properties_each_frame', 'w') as graph_file:
    for item in graph_list:
      graph_file.write(f"{item[0]} {item[1]} {item[2]}\n")
#

print('Graph analysis ans statistics starts')
print('Size of the complete graph', num_atoms)
dir_path='./'
# Paramètres pour l'extraction des graphes
nplot_max = 20  # Nombre maximum de graphes à représenter
#
with open(dir_path+'graph_data_hash256.txt', mode='w') as file, \
     open(dir_path+'graph_data_analysis_hash256.txt', mode='w') as file_analysis, \
     open(dir_path+'graph_data_analysis_all_hash256.txt', mode='w') as file_analysis_all:

# Calculer les variables en utilisant tous les graphes
 for residue in range(1, num_atoms + 1):  # le numéro du résidu dans la chaîne
#
    Amino=residue-1
    print(residue) # selected amino = residue -1 , nodes starts at 0
    D = 1 # distance from A
#
# Graph analysis
    graph_counts = defaultdict(int)
    graph_examples = {}
    graph_voisins1 = {}
    with open(dir_path+'every_frame_data_hash256_'+str(residue)+'.txt', mode='w') as file_hash: 
     frame =0
     for G in graphs_full:
         frame +=1
         if D == 2: graph, premiers_voisins, seconds_voisins = generate_subgraph12(G, Amino)
         if D == 1: graph, premiers_voisins = generate_subgraph1(G, Amino)
         graph_id = graph_hash_sha256(graph)
         graph_counts[graph_id] += 1
         if graph_id not in graph_examples:
             graph_examples[graph_id] = graph
             graph_voisins1[graph_id] = premiers_voisins
         file_hash.write(f'{residue} {frame} {graph_id} \n')
# Trier les graphes par ordre décroissant d'occurrences
    sorted_graph_items = sorted(graph_counts.items(), key=lambda item: item[1], reverse=True)
    with open(dir_path+'basis_set_data_hash256_'+str(residue)+'.txt', mode='w') as file_basis_set: 
     for item in sorted_graph_items:
        file_basis_set.write(f'{residue} {item} \n')  
# Store the data
    Entropy = 0
    Krel_average = 0
    dG = 0
    for graph_id, count in sorted_graph_items:
            graph = graph_examples[graph_id]
            kirchhoff_idx, constant_K, constant_Krel = kirchhoff_index(graph)
            dG += 0.5 * count * math.log(constant_Krel)
            proba = float(count / len(graphs_full))
            Krel_average += proba * constant_Krel
            Entropy -= proba * math.log(proba)
            edge_colors, edge_styles, edge_widths, nblack, ndotted, nred = edge_attributes(graph)
            file_analysis_all.write(f'{residue} {proba} {kirchhoff_idx:.3f} {constant_K:.3f} {constant_Krel:.3f} {nblack} {ndotted} {nred} \n')
    Entropymax = math.log(len(graphs_full))
    print('Entropy=', Entropy, ' Max=', Entropymax, 'DeltaG=', dG)
# Écrire les données dans le fichier texte
    file.write(f'{residue} {D} {Entropy} {Entropymax} {dG} {len(graphs_full)} {Krel_average}\n')
# Representation
# Limiter le nombre de graphes à représenter
    if len(sorted_graph_items) > nplot_max:
        sorted_graph_items = sorted_graph_items[:nplot_max]
# Déterminer le nombre de colonnes et de lignes
    num_columns = 8
    num_graphs = len(sorted_graph_items)
    num_rows = (num_graphs + num_columns - 1) // num_columns  # Calcul du nombre de lignes
# Créer une figure pour regrouper tous les graphes
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 5))
# Aplatir les axes pour une itération facile
    axes = axes.flatten()
# Dessiner chaque graphe dans un subplot
    for ax, (graph_id, count) in zip(axes, sorted_graph_items):
                graph = graph_examples[graph_id]
                noeuds_premiers_voisins = graph_voisins1[graph_id]
                pos = nx.spring_layout(graph)
                edge_colors, edge_styles, edge_widths, nblack, ndotted, nred = edge_attributes(graph)
            # Changer la numérotation des labels - démarre à 1 pour le premier noeud au lieu de 0
            # Créer des labels personnalisés numérotés à partir de 1
                labels = {i: i + 1 for i in graph.nodes()}
                for (u, v), color, style, width in zip(graph.edges(), edge_colors, edge_styles, edge_widths):
                    nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], edge_color=color, style=style, width=width, ax=ax)
                nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=500, ax=ax)
                nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, ax=ax)
                kirchhoff_idx, constant_K, constant_Krel = kirchhoff_index(graph)
                proba = float(count / len(graphs_full))
                file_analysis.write(f'{residue} {proba} {kirchhoff_idx:.3f} {constant_K:.3f} {constant_Krel:.3f} {nblack} {ndotted} {nred} \n')
                ax.set_title(f'Occurrences: {count}/{len(graphs_full)}  Proba={proba} \nKirchhoff index: {kirchhoff_idx:.3f} K: {constant_K:.3f} Krel: {constant_Krel:.3f}')
                ax.axis('off')
# Masquer les axes inutilisés
                for ax in axes[num_graphs:]:
                  ax.axis('off')
# Titre général
    fig.suptitle(f'{residue} {D} {Entropy} {Entropymax} {dG} {len(graphs_full)} {Krel_average}\n',fontsize=16)
# Ajuster l'espacement pour éviter le chevauchement
    plt.subplots_adjust(top=0.9)
# Sauvegarder la figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(dir_path+f'graph_occurrences_{residue}_hash256.png', facecolor='white')
    plt.close(fig)  # Fermer la figure pour libérer de la mémoire
print('FIN')







