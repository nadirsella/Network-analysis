from igraph import *
from scipy import linalg
import time
import numpy as np
from collections import defaultdict
import random

colors = ['r', 'b', 'g', 'y']

columnNames = ['Name', 'Activates', 'Inhibits', 'Activated', 'Inhibited', 'Out degree', 'In degree', 'Total degree', 'Eccentricity out', 'Eccentricity in', 'Entropy', 'Betweenness', 'Clustering coefficient', 'Vibrational centrality out', 'Vibrational centrality in',
                            'Information centrality', 'Closeness centrality out', 'Closeness centrality in', 'Subgraph centrality']

# print the matrix
def pri(mat):
    print('\n'.join(', '.join(str(round(y, 3)) for y in x) for x in mat))

def dupli(the_list):
    counts = defaultdict(int)
    for item in the_list:
        counts[item] += 1
    return counts.items()

# average of a list
def avg(list_):
    val = 0.0
    l = len(list_)
    for i in range(l):
        val += list_[i]
    val /= float(l)

    return val


# def degree(self):
#     t = int(round(time.time()))
#     for n in xrange(self.number_table):
#         if(self.graphList[n] != None):
#             self.graphList[n].degree = []
#             if(self.graphList[n].adj_matrix == None):
#                 adj = Graph.get_adjacency(self.graphList[n].graph)

#                 len_matrix, _ = adj.shape

#                 adj_matrix = []
#                 for i in xrange(len_matrix):
#                     adj_matrix.append(adj.__getitem__(i))
#             else:
#                 adj_matrix = self.graphList[n].adj_matrix
#                 len_matrix, _ = adj.shape

#             for i in xrange(len_matrix):
#                 sum = 0
#                 for j in xrange(len_matrix):
#                     if(adj_matrix[i][j] or adj_matrix[j][i]):
#                         sum += 1
#                 self.graphList[n].degree.append(sum)

#                 self.graphList[n].adj_matrix = adj_matrix
#     print "degree time: " + str(int(round(time.time())) -t)
#     return True

#fin d the position of the centrality index in the list
def findPos(str_):
    for idx, val in enumerate(columnNames):
        if(val == str_):
            return idx - 1


# find the out degree
def out_degree(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.outdegree(self.graphList[i].graph)
                l_l.append(l)
        else:
            l_l.append([])
    print("out degree evaluated")
    return l_l

#find the in degree
def in_degree(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.indegree(self.graphList[i].graph)
                l_l.append(l)
        else:
            l_l.append([])
    print("in degree evaluated")
    return l_l


#find the total degree
def total_degree(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.degree(self.graphList[i].graph)
                l_l.append(l)
        else:
            l_l.append([])
    print("total degree evaluated")
    return l_l


# find the degree for the vibrational centrality
def degree(self):
    t = int(round(time.time()))
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            self.graphList[n].degree = []
            if(self.graphList[n].adj_matrix == None):
                adj = Graph.get_adjacency(self.graphList[n].graph)

                len_matrix, _ = adj.shape

                adj_matrix = []
                for i in xrange(len_matrix):
                    adj_matrix.append(adj.__getitem__(i))
            else:
                adj_matrix = self.graphList[n].adj_matrix
                len_matrix, _ = adj.shape

            for i in xrange(len_matrix):
                sum = 0
                for j in xrange(len_matrix):
                    if(adj_matrix[i][j] or adj_matrix[j][i]):
                        sum += 1
                self.graphList[n].degree.append(sum)

                self.graphList[n].adj_matrix = adj_matrix
    print("degree time: " + str(int(round(time.time())) -t))
    return True

#find the diameter
def diameter(self):
    l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                if(self.graphList[i].directed):
                    val = Graph.diameter(self.graphList[i].graph, directed=True)
                else:
                    val = Graph.diameter(self.graphList[i].graph, directed=False)
                l.append(val)
    print("diameter evaluated")
    return l

# find the assortativity
def assortativity(self):
    l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                if(self.graphList[i].directed):
                    val = Graph.assortativity_degree(self.graphList[i].graph, directed=True)
                else:
                    val = Graph.assortativity_degree(self.graphList[i].graph, directed=False)
                l.append(val)
    print("assortativity evaluated")
    return l


# find the betweenness
def betweenness((self)):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                graph = self.graphList[i].graph

                weights = graph.es["weight"]
                weights = [1 / w for w in weights]


                if(self.graphList[i].directed):
                    l =  Graph.betweenness(self.graphList[i].graph, vertices=None, 
                        directed=True, cutoff =None,  weights=weights, nobigint=False)
                else:
                    l =  Graph.betweenness(self.graphList[i].graph, vertices=None, 
                        directed=False, cutoff =None,  weights=weights, nobigint=False)
                l_l.append(l)
        else:
            l_l.append([])
    print("betweenness evaluated")
    return l_l
                # self.graphList[i].centrality[pos] = Graph.betweenness(self.graphList[i].graph, vertices=None, 
                #    directed=True, cutoff =None,  weights=self.graphList[i].graph.es["weight"], nobigint=True)


# find the clustering coefficient
def clustering_coefficient(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:

                weights = self.graphList[i].graph.es["weight"]
                weights = [1 / w for w in weights]

                l = Graph.transitivity_local_undirected(self.graphList[i].graph,
                    vertices=None, mode="nan", weights=weights)
                l_l.append(l)
        else:
            l_l.append([])
    print("cc evaluated")
    return l_l
                # self.graphList[i].centrality[pos] = Graph.transitivity_local_undirected(self.graphList[i].graph,
                #     vertices=None, mode="nan", weights=self.graphList[i].graph.es["weight"])

# find eccentricity out
def eccentricity_out(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.eccentricity(self.graphList[i].graph, vertices=None, mode=OUT)
                l_l.append(l)
        else:
            l_l.append([])
    print("eccentricity out evaluated")
    return l_l

# find the eccentricity in
def eccentricity_in(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.eccentricity(self.graphList[i].graph, vertices=None, mode=IN)
                l_l.append(l)
        else:
            l_l.append([])
    print("eccentricity in evaluated")
    return l_l

# find the entropy
def entropy(self):
    l_l = []
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                l = Graph.diversity(self.graphList[i].graph, weights=self.graphList[i].graph.es["weight"])
                l_l.append(l)
        else:
            l_l.append([])

    print("entropy evaluated")
    return l_l


#find the laplacian:pseudoinverse
def laplacian_psinv(self):
    l_l = []
    #laplacian matrix
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            if(self.graphList[n].laplacian_psinv == None):
                if(self.graphList[n].laplacian_psinv == None):
                    #get the graph laplacian
                    l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=False)
                    #pseudoinverse
                    self.graphList[n].laplacian_psinv = lpsinv = linalg.pinv2(l)

#find the laplacian:pseudoinverse normalized
def laplacian_psinv_norm(self):
    l_l = []
    #laplacian matrix
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            if(self.graphList[n].laplacian_psinv == None):
                if(self.graphList[n].laplacian_psinv == None):
                    #l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=True)
                    l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=False)
                    #pseudoinverse


                    len_matrix = len(self.graphList[n].graph.vs["name"])

                    graph = self.graphList[n].graph

                    adj_list = graph.get_adjlist(mode=IN)
                    # find the average beta for every node
                    avg_weight_vect = [0 for i in xrange(len_matrix)]
                    for i in range(len_matrix):
                        s = 0
                        for j in set(adj_list[i]):
                            s += graph[i, j]

                        avg_weight_vect[i] = s


                    deg = Graph.degree(self.graphList[n].graph)
                    graph = self.graphList[n].graph

                    len_matrix = len(self.graphList[n].graph.vs["name"])

                    # find matrix b
                    #b = [[deg[r] + 1 if r == c else 1 for c in xrange(len_matrix)] for r in xrange(len_matrix)]
                    out = Graph.outdegree(self.graphList[n].graph)
                    b = [[1.0 if(r == c and out[r] != 0) else 0 for c in xrange(len_matrix)] for r in xrange(len_matrix)]

                    adj_list = graph.get_adjlist(mode=OUT)

                    for i in xrange(len_matrix):
                        for j in adj_list[i]:
                            b[i][j] = (-1*((graph[i, j]/(1+(out[i]*out[j])))))

                    lpsinv = linalg.pinv2(b)

                    self.graphList[n].laplacian_psinv = lpsinv


# def laplacian_centrality(self, q, vs=None):
#     for n in xrange(self.number_table):
#         if vs is None:
#             vs = xrange(self.graphList[n].graph.vcount())
#         result = []
#         for v in vs:
#             neis = self.graphList[n].neighbors(v, mode="all")
#             result.append(self.graphList[n].centrality[4][v]**2 + self.graphList[n].centrality[4][v] + 2 * sum(self.graphList[i].centrality[4][i] for i in neis))
#         return result


# find the vibrational centrality out
def vibrational_centrality_out(self):
    l_l = []
    laplacian_psinv(self)
    t = int(round(time.time()))
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            len_matrix = len(self.graphList[n].graph.vs["name"])

            lpsinv = self.graphList[n].laplacian_psinv

            #node centrality list
            lpsinv = [[0 if lpsinv[r][c] < 1e-10 else lpsinv[r][c] for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            try:
                l1=[math.sqrt(lpsinv[i][i]) if (lpsinv[i][i] != 0) else 0 for i in xrange(len_matrix)]
            except:
                print(lpsinv[i][i])
            l_l.append(l1)
        else:
            l_l.append([])
    print("vibrational evaluated")
    return l_l


# find the vibrational centrality out normalized
def vibrational_centrality_out_normalized(self):
    l_l = []
    laplacian_psinv_norm(self)
    t = int(round(time.time()))
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            len_matrix = len(self.graphList[n].graph.vs["name"])
            lpsinv = self.graphList[n].laplacian_psinv

            #node centrality list
            lpsinv = [[0 if lpsinv[r][c] < 1e-10 else lpsinv[r][c] for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            try:
                l1=[math.sqrt(lpsinv[i][i]) if (lpsinv[i][i] != 0) else 0 for i in xrange(len_matrix)]
            except:
                print(lpsinv[i][i])
            l_l.append(l1)
        else:
            l_l.append([])
    print("vibrational evaluated")
    return l_l

# find the vibrational centrality in
def vibrational_centrality_in(self):
    l_l = []
    #laplacian_psinv(self)
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            len_matrix = len(self.graphList[n].graph.vs["name"])

            graph = self.graphList[n].graph

            adj_list = graph.get_adjlist(mode=IN)

            # find the average beta for every node
            avg_weight_vect = [0 for i in xrange(len_matrix)]
            for i in range(len_matrix):
                s = 0
                for j in (adj_list[i]):
                    s += graph[j, i]

                avg_weight_vect[i] = s


            deg = Graph.degree(self.graphList[n].graph)
            graph = self.graphList[n].graph

            len_matrix = len(self.graphList[n].graph.vs["name"])

            # find matrix b
            b = [[avg_weight_vect[r] if r == c else 0 for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            adj_list = graph.get_adjlist(mode=OUT)

            for i in xrange(len_matrix):
                for j in adj_list[i]:
                    b[i][j] = ((-1*(graph[i, j])))

            lpsinv = linalg.pinv2(b)

            #node centrality list
            lpsinv = [[0 if lpsinv[r][c] < 1e-10 else lpsinv[r][c] for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            try:
                l1=[math.sqrt(lpsinv[i][i]) if (lpsinv[i][i] != 0) else 0 for i in xrange(len_matrix)]
            except:
                print(lpsinv[i][i])
            l_l.append(l1)
        else:
            l_l.append([])
    return l_l

# find the vibrational centrality in normalized
def vibrational_centrality_in_normalized(self):
    l_l = []
    #laplacian_psinv(self)
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            len_matrix = len(self.graphList[n].graph.vs["name"])

            graph = self.graphList[n].graph

            adj_list = graph.get_adjlist(mode=IN)
            # find the average beta for every node
            avg_weight_vect = [0 for i in xrange(len_matrix)]
            for i in range(len_matrix):
                s = 0
                for j in set(adj_list[i]):
                    s += graph[j, i]

                avg_weight_vect[i] = s


            deg = Graph.degree(self.graphList[n].graph)
            graph = self.graphList[n].graph

            len_matrix = len(self.graphList[n].graph.vs["name"])

            in_ = Graph.indegree(self.graphList[n].graph)
            b = [[1.0 if(r == c and in_[r] != 0) else 0 for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            adj_list = graph.get_adjlist(mode=OUT)

            for i in xrange(len_matrix):
                for j in adj_list[i]:
                    b[i][j] = (-1*((graph[i, j]/(1+(in_[i]*in_[j])))))

            lpsinv = linalg.pinv2(b)

            #node centrality list

            lpsinv = [[0 if lpsinv[r][c] < 1e-10 else lpsinv[r][c] for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            try:
                l1=[math.sqrt(lpsinv[i][i]) if (lpsinv[i][i] != 0) else 0 for i in xrange(len_matrix)]
            except:
                print(lpsinv[i][i])
            l_l.append(l1)
        else:
            l_l.append([])
    return l_l

# find information centrality, only for undirected networks
def information_centrality(self):
    l_l = []
    #laplacian_psinv(self)
    for n in xrange(self.number_table):
        if(self.graphList[n] != None and self.graphList[n].directed == False):
            len_matrix = len(self.graphList[n].graph.vs["name"])

            graph = self.graphList[n].graph

            adj_list = graph.get_adjlist(mode=OUT)
            # find the average beta for every node
            avg_weight_vect = [0 for i in xrange(len_matrix)]
            for i in range(len_matrix):
                s = 0
                for j in set(adj_list[i]):
                    s += graph[i, j]

                avg_weight_vect[i] = s


            deg = Graph.degree(self.graphList[n].graph)
            graph = self.graphList[n].graph

            len_matrix = len(self.graphList[n].graph.vs["name"])

            # find matrix b
            b = [[avg_weight_vect[r] + 1 if r == c else 1 for c in xrange(len_matrix)] for r in xrange(len_matrix)]

            adj_list = graph.get_adjlist(mode=OUT)

            for i in xrange(len_matrix):
                for j in adj_list[i]:
                    b[i][j] = ((-1*(graph[i, j])) + 1)


            #find matrix inverse
            b_inv = linalg.inv(b)

            i_mat = [[1 / (b_inv[r][r] + b_inv[c][c] - 2*b_inv[r][c]) for c in xrange(len_matrix)] for r in xrange(len_matrix)]


            #node centrality list
            l1=[]
            for i in xrange(len_matrix):
                sum = 0
                for j in xrange(len_matrix):
                    sum += 1 / i_mat[i][j]
                l1.append(1/(sum/len_matrix))

            l_l.append(l1)
        else:
            l_l.append([])
    return l_l
            
#find the closeness 
def closeness_out(self):
    l_l = []
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
             if self.graphList[n].graph != None:

                weights = self.graphList[n].graph.es["weight"]
                weights = [1 / w for w in weights]

                l = Graph.closeness(self.graphList[n].graph, mode=OUT,
                    vertices=None, weights=weights)
                l_l.append(l)
        else:
            l_l.append([])
    print("closeness evaluated")
    return l_l

def closeness_in(self):
    l_l = []
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
             if self.graphList[n].graph != None:

                weights = self.graphList[n].graph.es["weight"]
                weights = [1 / w for w in weights]

                l = Graph.closeness(self.graphList[n].graph, mode=IN,
                    vertices=None, weights=weights)
                l_l.append(l)
        else:
            l_l.append([])
    print("closeness evaluated")
    return l_l

def subgraph_centrality(self):
    l_l = []
    for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            if self.graphList[n].graph != None:
                graph = self.graphList[n].graph
                len_matrix = len(graph.vs["name"])

                adj_mat = [[0 for c in xrange(len_matrix)] for r in xrange(len_matrix)]

                adj_list = graph.get_adjlist(mode=OUT)

                weights = graph.oldes
                graph.esCopy = graph.es["weight"]
                graph.es["weight"] = graph.oldes

                for i in xrange(len_matrix):
                    for j in adj_list[i]:
                        adj_mat[i][j] = (graph[i, j])

                graph.es["weight"] = graph.esCopy

                adj_mat = np.array(adj_mat)

                expA = linalg.expm(adj_mat)

                l1=[expA[i][i] for i in xrange(len_matrix)]

                l_l.append(l1)
        else:
            l_l.append([])
            
    print("subgraph evaluated")
    return l_l

def pseudocolor(val, minval, maxval):
    import colorsys
    # convert val in range minval..maxval to the range 0..120 degrees which
    # coorespond to the colors red..green in the HSV colorspace
    h = (float(val-minval) / (maxval-minval)) * 330
    # convert hsv color (h,1,1) to its rgb equivalent
    # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
    return (r,g,b)
    #return "rgba(" + str(r) + "," + str(g) +"," + str(b) + ",50)"

def plot_coefficient(self, i, j):
    import cairo
    from igraph.drawing.text import TextDrawer
    random.seed(1)
    g = self.graphList[i].graph
    if(g != None and len(g.vs["name"]) < 1000):
        #layout = g.layout("kamada_kawai")
        layout = g.layout("fr")

        g.es["curved"] = False
        visual_style = {}
        visual_style["vertex_size"] = 20
        #visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
        visual_style["vertex_label"] = g.vs["name"]
        l = self.graphList[i].centrality[j]
        m = max(l)
        l1 = [(float(l[i]) / float(m)) * 100 for i in range(len(l))]

        col = []
        pal = RainbowPalette(n=120)
        for n, val in enumerate(l1):
            col.append(pseudocolor(val,0,100))

        visual_style["vertex_color"] = col

        visual_style["edge_width"] = [float(weight) for weight in g.es["weight"]]
        visual_style["edge_color"] = ["red" if (w < 0) else "black" for w in g.es["weight"]]
        visual_style["edge_color"] = ["red" if (w < 0) else "black" for w in g.oldes]

        visual_style["layout"] = layout
        #visual_style["bbox"] = (800, 800)
        visual_style["margin"] = 20
        visual_style["edge-curved"] = False


        plot = Plot("plot.png", bbox=(720, 760), background="white")

        # # Create the graph and add it to the plot
        plot.add(g, bbox=(20, 80, 580, 630), **visual_style)

        # save_img = PhotoImage(file="save.gif")
        #plot.add(save_img, bbox=(20, 70, 580, 630))

        # Make the plot draw itself on the Cairo surface
        plot.redraw()

        # Grab the surface, construct a drawing context and a TextDrawer
        ctx = cairo.Context(plot.surface)
        ctx.set_font_size(36)
        drawer = TextDrawer(ctx, columnNames[j+1], halign=TextDrawer.CENTER)
        drawer.draw_at(0, 40, width=600)

        image_surface = cairo.ImageSurface.create_from_png("rainbow.png")
        ctx.set_source_surface(image_surface, 55, 680)
        ctx.paint()
        
        #ctx.plot(image_surface, target=None, bbox=(0, 0, 600, 600))
        #plot.add(image_surface, bbox=(20, 20, 580, 580))

        print type(plot)

        return plot


def plot_graphs(self, folder):
    import matplotlib.pyplot as plt


    if(self.number_table > 0):
         for tab in xrange(self.number_table):
            for n in range(len(self.graphList[0].centrality)):
                if(len(self.graphList[tab].centrality[n]) > 0):
                    networkFile = self.graphList[tab].name


                    try:
                        if(os.name == "nt"):
                            pos = networkFile.index("\\")
                        else:
                            pos = networkFile.index("/")
                    except:
                        pos = -1

                    if(pos != -1):
                        networkFile = networkFile[pos+1:]

                    try:
                        pos = networkFile.index(".txt")
                    except:
                        pos = -1
                    if(pos != -1):
                        networkFile = networkFile[:pos]
                    

                    if(os.name == "nt"):
                        path = folder +"\\"+ networkFile + "\\"
                    else:
                        path = folder +"/"+ networkFile + "/"

                    if not os.path.exists(path): 
                        os.makedirs(path)

                    if(os.name == "nt"):
                        path = folder +"\\"+ networkFile + "\\" + columnNames[n+1]
                    else:
                        path = folder +"/"+ networkFile + "/" + columnNames[n+1]


                    #plt.scatter(self.graphList[tab].centrality[2],self.graphList[tab].centrality[6])

                    #plt.show()


                    plt = plot_coefficient(self, tab, n)
                    print(type(plt))
                    # Save the plot
                    plt.save(path + ".png")
