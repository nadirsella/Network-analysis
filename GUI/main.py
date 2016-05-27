#!/usr/bin/python

__author__ = 'Nadir Sella'
__version__ = '0'
__date__ = ''
import time
import logging
from sys import argv
from os.path import isfile
import itertools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tkFileDialog import askopenfilename
from tkintertable.Tables import *
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel
import colorsys
from thread_func import *
from igraph import *
from read_csv import plot_curve


colors = ['r', 'b', 'g', 'y']
columnNames = ['Name', 'Activates', 'Inhibits', 'Activated', 'Inhibited', 'Out degree', 'In degree', 'Total degree', 'Eccentricity out', 'Eccentricity in', 'Entropy', 'Betweenness', 'Clustering coefficient', 'Vibrational centrality out', 'Vibrational centrality in',
                            'Information centrality', 'Closeness centrality out', 'Closeness centrality in', 'Subgraph centrality']

def pri(mat):
	print '\n'.join(', '.join(str(round(y, 3)) for y in x) for x in mat)

#center the window on the screen
def center_window(root, w=300, h=200):
    # get screen width and height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    # calculate position x, y
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))


class Set_value(object):
    def __init__(self, frame, title, label):
        self.finestra = Tk()
        self.finestra.title(title)
        center_window(self.finestra, 250, 120)
        label = Label(self.finestra, text="\n" + label)
        label.pack()
        txt_box = Entry(self.finestra, text = "", width=60)
        txt_box.pack()
        button = Button(master=self.finestra, text='Ok', command=lambda: self.insert_value(frame, txt_box))
        button.pack()
        self.finestra.mainloop()

    def insert_value(self, frame,  txt_box):
        frame.value=txt_box.get()
        self.finestra.destroy()
        self.finestra.quit()

def close(self):
    self.destroy()

#show a window with No valid network loaded message
def no_valid_message(msg):
    finestra = Tk()
    finestra.title("Error")
    center_window(finestra, 250, 80)
    label = Label(finestra, text="\n" + msg)
    button = Button(master=finestra, text='Ok',command=lambda: close(finestra), width=8)

    label.pack()
    button.pack()

    finestra.mainloop()

# let to choose a file and load the corresponding network
def choose_file(self, usable_width):
    t = int(round(time.time()))
    filename = askopenfilename()
    if(filename is None):
        return
    if not (isfile(filename)):
        no_valid_message("No valid network selected")
    else:
        #if(self.number_table > 0):
        self.txt_box.delete(0, END)
        self.txt_box.insert(END, filename)

        Set_value(self, "Direction?", "Is network directed? y/n")
        v = self.value
        if(v == "y"):
            v = True
        elif(v == "n"):
            v = False

        try:
            if(v):
                graph = Graph.Read_Ncol(filename, names=True, directed=True, weights="if_present")
            else:
                graph = Graph.Read_Ncol(filename, names=True, directed=False, weights="if_present")
            #print graph.is_connected(mode=WEAK)

            #l_g = Graph.decompose(graph, mode=1, maxcompno=30000, minelements=1)
            #print(len(l_g))

            g = my_graph(self, usable_width, v)
            self.graphList.append(g)
            self.graphList[self.number_table].graph = graph
            self.graphList[self.number_table].assortativity = round(Graph.assortativity_degree(self.graphList[self.number_table].graph, directed=True), 5)
            addGraph(self, filename, t)


            # if(graph.is_connected()):
            #     g = my_graph(self, usable_width)
            #     self.graphList.append(g)
            #     self.graphList[self.number_table].graph = graph
            #     self.graphList[self.number_table].assortativity = round(Graph.assortativity_degree(self.graphList[self.number_table].graph, directed=True), 5)
            #     self.graphList[self.number_table].name= filename[filename.rfind("/")+1:]
            #     addGraph(self, filename, t)
            # else:
            #     l_g = Graph.decompose(graph, mode=WEAK, maxcompno=30, minelements=1)
            #     print(len(l_g))
            #     for idx, l in enumerate(l_g):
            #         g = my_graph(self, usable_width)
            #         print self.number_table
            #         self.graphList.append(g)
            #         self.graphList[self.number_table].graph = l
            #         self.graphList[self.number_table].assortativity = round(Graph.assortativity_degree(self.graphList[self.number_table].graph, directed=True), 5)
            #         self.graphList[self.number_table].name= filename[filename.rfind("/")+1:]
            #         addGraph(self, filename + str(idx), t)
        except Exception as e:
            print e
            no_valid_message("No valid network selected")


#add the graph and the table to the list
def addGraph(self, filename,t ):
    n = self.number_table
    self.graphList[n].name= filename[filename.rfind("/")+1:]

    self.graphList[n].graph.vs["label"] = self.graphList[n].graph.vs["name"]
    self.graphList[n].file_name.set("File loaded: " + filename[filename.rfind("/")+1:])
    nodes = self.graphList[n].graph.vcount()
    arcs = self.graphList[n].graph.ecount()
    self.graphList[n].number_nodes_str.set(nodes)
    self.graphList[n].number_arcs_str.set(arcs)
    self.graphList[n].assortativity_str.set(round(Graph.assortativity_degree(self.graphList[n].graph, directed=True), 5))
    #create rows for the table
    nodes = self.graphList[n].graph.vcount()
    table = self.graphList[n].table
    for i in xrange(nodes):
        table.addRow(self.graphList[n].graph.vs["label"][i])
        table.model.data[self.graphList[n].graph.vs["name"][i]]["Name"] = self.graphList[n].graph.vs["label"][i]
    self.graphList[n].model.deleteRow(0)
    self.graphList[n].table.adjustColumnWidths()
    self.graphList[n].table.redrawTable()
    #self.graphList[self.number_table].adj_matrix = Graph.get_adjacency(self.graphList[self.number_table].graph)

    graph = self.graphList[n].graph

    #count the activations and inhibitions if the graph is oriented
    countInhibitions = [0] * graph.vcount()
    countActivations = [0] * graph.vcount()
    countActivated = [0] * graph.vcount()
    countInhibited = [0] * graph.vcount()

    for idx, node in enumerate(graph.vs["name"]):
        nodeneigh = graph.neighbors(node)
        for n in nodeneigh:
            if(graph[node,n] > 0):
                countActivations[idx] += 1
            elif(graph[node,n] < 0):
                countInhibitions[idx] +=1

            if(graph[n,node] > 0):
                countActivated[idx] += 1
            elif(graph[n,node] < 0):
                countInhibited[idx] +=1

    pos = findPos("Activates")
    assoc_name(self, countActivations, graph.vs["name"], self.graphList[self.number_table].columnNames[pos+1], self.number_table)
    pos = findPos("Inhibits")
    assoc_name(self, countInhibitions, graph.vs["name"], self.graphList[self.number_table].columnNames[pos+1], self.number_table)
    pos = findPos("Activated")
    assoc_name(self, countActivated, graph.vs["name"], self.graphList[self.number_table].columnNames[pos+1], self.number_table)
    pos = findPos("Inhibited")
    assoc_name(self, countInhibited, graph.vs["name"], self.graphList[self.number_table].columnNames[pos+1], self.number_table)

    graph.oldes = graph.es["weight"]
    graph.es["weight"] = [abs(x) for x in (graph.es["weight"])]


    self.number_table += 1



# associate the value of the centrality index to the correct node and plot in the table
def assoc_name(self, value_list, node_list, type, i):
    for idx, degree in enumerate(value_list):
        if(self.graphList[i].directed):
            self.graphList[i].table.model.data[node_list[idx]][type] = round(degree, 5)
        else:
            print type
            self.graphList[i].table.model.data[node_list[idx]][type.replace(" out", "").replace(" in", "").replace("Out degree", "Degree").replace("In degree", "Degree")] = round(degree, 5)

    self.graphList[i].table.redrawTable()


#find the position of the centrality measure
def findPos(str):
    for idx, val in enumerate(columnNames):
        if(val == str):
            return idx - 1

#fill the table with the centrality measure passed as parameter
def fill(self, pos, l_l):
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            self.graphList[i].centrality[pos] = l_l[i]
            if(self.graphList[i] != []):
                if self.graphList[i].graph != None:
                    assoc_name(self, self.graphList[i].centrality[pos], self.graphList[i].graph.vs["name"], self.graphList[i].columnNames[pos+1], i)


#evaluate centrality measures
def out_degreeTT(self, str_):
    l_l = out_degree(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def in_degreeTT(self, str_):
    l_l = in_degree(self)
    pos = findPos(str_)
    fill(self, pos, l_l)


def total_degreeTT(self, str_):
    l_l = total_degree(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def diameter(self):
    for i in xrange(self.number_table):
        if(self.graphList[i] != None):
            if self.graphList[i].graph != None:
                self.graphList[i].diameter_str.set(Graph.diameter(self.graphList[i].graph, directed=True))

def betweennessTT(self, str_):
    l_l = betweenness(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def clustering_coefficientTT(self, str_):
    l_l = clustering_coefficient(self)
    pos = findPos(str_)
    fill(self, pos, l_l)


def eccentricity_out_TT(self, str_):
    l_l = eccentricity_out(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def eccentricity_in_TT(self, str_):
    l_l = eccentricity_in(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def entropyTT(self, str_):
    l_l = entropy(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def laplacian_inv(self):
    #laplacian matrix
    for n in xrange(self.number_table):
        if(self.graphList[n].laplacian_inv == None):
            if(self.graphList[n].laplacian_inv == None):
                l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=False)
                #pri(l)
                #pseudoinverse
                lpsinv=linalg.pinv2(l)
                self.graphList[n].laplacian_psinv = lpsinv


def laplacian_centrality(self, vs=None):
    for n in xrange(self.number_table):
        if vs is None:
            vs = xrange(self.graphList[n].graph.vcount())
        result = []
        for v in vs:
            neis = self.graphList[n].neighbors(v, mode="all")
            result.append(self.graphList[n].centrality[4][v]**2 + self.graphList[n].centrality[4][v] + 2 * sum(self.graphList[i].centrality[4][i] for i in neis))
        return result

def vibrational_centrality_out_TT(self, str_):
    l_l = vibrational_centrality_out(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def vibrational_centrality_in_TT(self, str_):
    l_l = vibrational_centrality_in(self)
    pos = findPos(str_)
    fill(self, pos, l_l)


def information_centralityTT(self, str_):
    l_l = information_centrality(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def closeness_centrality_out_TT(self, str_):
    l_l = closeness_out(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def closeness_centrality_in_TT(self, str_):
    l_l = closeness_in(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

def subgraph_centralityTT(self, str_):
    l_l = subgraph_centrality(self)
    pos = findPos(str_)
    fill(self, pos, l_l)

# def vibr2(self):
#      for n in xrange(self.number_table):
#         if(self.graphList[n] != None):
#             len_matrix = len(self.graphList[n].graph.vs["name"])
#             #laplacian matrix
#             l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=False)
#             a = np.array(l)
#             res = []
#             for i in xrange(len_matrix):
#                 res.append([])
#                 for j in xrange(len_matrix):
#                     res[i][j] = linalg.det(a) / linalg.det(a)


# find the indiced subgraph
def induced_subgraph(self , i, usable_width):
    vertex = []
    try:
        vertex = self.graphList[i].table.get_selectedRecordNames()
    except:
        print "Error"

    if(len(vertex) > 0):
        self.graph_in_memory = self.graphList[i].graph.subgraph(vertex)

        Set_value(self, "Insert name", "Insert graph name")
        if(isinstance(self.value, str)):
            g = my_graph(self, usable_width, self.graphList[i].directed)
            self.graphList.append(g)
            self.graphList[self.number_table].graph = self.graph_in_memory
            addGraph(self, self.value,1)
        else:
            print "Error"
    else:
        no_valid_message("No gene selected")

# find the subgraph at a certain depth
def subgraph_levels(self, i, usable_width):
    vertex = []
    try:
        vertex = self.graphList[i].table.get_selectedRecordNames()
    except:
        print "Error"

    if(len(vertex) > 0):
        try:
            Set_value(self, "Insert depth", "Insert search depth")

            copy =  self.graphList[i].graph.es["weight"]
            self.graphList[i].graph.es["weight"] = self.graphList[i].graph.oldes
            #list_list = self.graphList[i].graph.neighborhood(vertices=vertex, order=int(self.value), mode="out")
            list_list = self.graphList[i].graph.neighborhood(vertices=vertex, order=int(self.value))
            merged = list(itertools.chain.from_iterable(list_list))
            self.graph_in_memory = self.graphList[i].graph.subgraph(merged)
            self.graphList[i].graph.es["weight"] = copy
        except:
            print "error"

        Set_value(self, "Insert name", "Insert graph name")
        if(isinstance(self.value, str)):
            g = my_graph(self, usable_width, self.graphList[i].directed)
            self.graphList.append(g)
            self.graphList[self.number_table].graph = self.graph_in_memory
            addGraph(self, self.value,1)
        else:
            print "Error"
    else:
        no_valid_message("No gene selected")

#save the network in a file
def save_network(self, num):
    f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".txt")
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    copy =  self.graphList[num].graph.es["weight"]
    self.graphList[num].graph.es["weight"] = self.graphList[num].graph.oldes
    Graph.write(self.graphList[num].graph, f, "ncol")
    self.graphList[num].graph.es["weight"] = copy

# def plot_histogram(self, pos, plot_name):
#     for i in xrange(self.number_table):
#         map = Counter(self.graphList[i].centrality[pos])

#         plt.bar(map.keys(), map.values(), color=colors[i])
#         plt.axis([-0.5, int(max(map.keys())+2), -5,int(max(map.values())+2)])

#         #plt.xticks(range(len(map)), map.keys())

#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.title(plot_name)
#     plt.show()


# def plot_chart(self,pos, plot_name):
#     lx=[]
#     ly=[]
#     for i in xrange(self.number_table):
#         map = Counter(self.graphList[i].centrality[pos])

#         plt.plot(map.keys(), map.values(), 'o', color=colors[i])
#         plt.axis([-0.5, int(max(map.keys())+2), -5,int(max(map.values())+2)])


#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.title(plot_name)
#     plt.show()

# plot the graph with no colors
def plot_graph(self, i):
   #random.seed(1)
    g = self.graphList[i].graph
    if g != None:
        layout = g.layout("kamada_kawai")
        g.es["curved"] = False
        visual_style = {}
        visual_style["vertex_size"] = 70
        #visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
        visual_style["vertex_label"] = g.vs["name"]
        visual_style["vertex_label_size"] = 28
        visual_style["edge_width"] = [float(weight) for weight in g.es["weight"]]
        visual_style["vertex_color"] = ["red" if(n =="SPOP" or n =="AR") else "grey" for n in g.vs["name"]]
        visual_style["edge_color"] = ["red" if (w < 0) else "black" for w in g.oldes]
        visual_style["layout"] = layout
        visual_style["bbox"] = (1080, 1080)
        visual_style["margin"] = 60

        if(os.name == "nt"):
            path = "temp\\"+ self.graphList[i].name + ".png"
        else:
            path = "temp/"+ self.graphList[i].name + ".png"

        plot(g, path, **visual_style)
        plot(g, **visual_style)

# def pseudocolor(val, minval, maxval):
#     # convert val in range minval..maxval to the range 0..120 degrees which
#     # coorespond to the colors red..green in the HSV colorspace
#     h = (float(val-minval) / (maxval-minval)) * 330
#     # convert hsv color (h,1,1) to its rgb equivalent
#     # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
#     r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
#     return (r,g,b)
#     #return "rgba(" + str(r) + "," + str(g) +"," + str(b) + ",50)"

# remove the network from the list of networks, it is not deleted, a boolean is put as deleted and the memory is freed
def delete_network(self, i):
    self.graphList[i].deleted = True
    self.graphList[i].ftable.pack_forget()
    self.graphList[i] = None

# class for each network
class my_graph(object):
    def __init__(self, frame, usable_width, direction):

        self.deleted = False

        self.directed = direction

        self.ftable = Frame(frame.ftables)
        self.ft = Frame(self.ftable)
        self.fnp = Frame(self.ftable)

        #frame network properties
        if(usable_width > 1500)       :
            button_width=17
            button_width_plot=7
            font=('Verdana',9)
            image_button = 23
            list_fonts = list( tkFont.families() )
        elif(usable_width > 1200):
            button_width=15
            button_width_plot=6
            image_button = 28
            font=('Verdana',10)
        else:
            button_width=14
            button_width_plot=5
            image_button = 28
            font=('Verdana',8)


        if(usable_width > 1500):
            self.save_img = PhotoImage(file="save.gif")
            self.load_img = PhotoImage(file="load.gif")
            self.delete_img = PhotoImage(file="delete.gif")
        elif(usable_width > 1200):
            self.save_img = PhotoImage(file="save22.gif")
            self.load_img = PhotoImage(file="load22.gif")
            self.delete_img = PhotoImage(file="delete.gif")
        else:
            self.save_img = PhotoImage(file="save20.gif")
            self.load_img = PhotoImage(file="load20.gif")
            self.delete_img = PhotoImage(file="delete.gif")

        labelfnp_width = 10
        pos = 0
        # buttons area for the right panel of each graph
        self.buttonSave = Button(master=self.fnp, image=self.save_img, command=lambda num=frame.number_table: save_network(frame, num), relief=FLAT, width=image_button)
        self.buttonSave.image = self.save_img
        self.buttonSave.grid(row=0, column=0, pady=1, padx=1)

        self.buttonDelete = Button(master=self.fnp, image=self.delete_img, command=lambda num=frame.number_table: delete_network(frame, num), relief=FLAT, width=image_button)
        self.buttonDelete.image = self.delete_img
        self.buttonDelete.grid(row=pos, column=1, pady=1, padx=1)
        pos += 1

        self.plot_button = Button(master=self.fnp, text='Plot', command=lambda num=frame.number_table: plot_graph(frame, num), width=button_width)
        self.plot_button.grid(row=pos, column=0, columnspan=2, pady=2, padx=5)

        pos += 1

        self.netPro_label = Label(self.fnp, text="Network properties:", width=3*labelfnp_width, anchor=CENTER)
        self.netPro_label.grid(row=pos, column=0, columnspan=2, pady=2, padx=5)
        pos += 1
        self.number_nodesLabel = Label(self.fnp, text="Nodes: ", width=labelfnp_width, anchor=W)
        self.number_nodesLabel.grid(row=pos, column=0, pady=2, padx=5)
        self.number_nodes_str = StringVar()
        self.number_nodes_l = Label(self.fnp, textvariable=self.number_nodes_str, width=labelfnp_width, anchor=W)
        self.number_nodes_l.grid(row=pos, column=1, pady=2, padx=5)
        pos += 1

        self.number_arcsLabel = Label(self.fnp, text="Arcs: ", width=labelfnp_width, anchor=W)
        self.number_arcsLabel.grid(row=pos, column=0, pady=2, padx=5)
        self.number_arcs_str = StringVar()
        self.number_arcs_l = Label(self.fnp, textvariable=self.number_arcs_str, width=labelfnp_width, anchor=W)
        self.number_arcs_l.grid(row=pos, column=1, pady=2, padx=5)
        pos += 1

        self.number_diameterLabel = Label(self.fnp, text="Diameter: ", width=labelfnp_width, anchor=W)
        self.number_diameterLabel.grid(row=pos, column=0, pady=2, padx=5)
        self.diameter_str = StringVar()
        self.diameter_l = Label(self.fnp, textvariable=self.diameter_str, width=labelfnp_width, anchor=W)
        self.diameter_l.grid(row=pos, column=1, pady=2, padx=5)
        pos += 1

        self.assortativityLabel = Label(self.fnp, text="Assortativity: ", width=labelfnp_width, anchor=W)
        self.assortativityLabel.grid(row=pos, column=0, pady=2, padx=5)
        self.assortativity_str = StringVar()
        self.assortativity_l = Label(self.fnp, textvariable=self.assortativity_str, width=labelfnp_width, anchor=W)
        self.assortativity_l.grid(row=pos, column=1, pady=2, padx=5)
        pos += 1

        self.entropyLabel = Label(self.fnp, text="Entropy: ", width=labelfnp_width, anchor=W)
        self.entropyLabel.grid(row=pos, column=0, pady=2, padx=5)
        self.entropy_str = StringVar()
        self.entropy_l = Label(self.fnp, textvariable=self.entropy_str, width=labelfnp_width, anchor=W)
        self.entropy_l.grid(row=pos, column=1, pady=2, padx=5)
        pos += 1

        self.subgraph_ind_sub = Button(master=self.fnp, text='Induced subgraph', font=font, command=lambda num=frame.number_table: induced_subgraph(frame, num, usable_width), width=button_width)
        self.subgraph_ind_sub.grid(row=pos, column=0, pady=2, padx=5)

        pos+=1
        self.subgraph_ind_sub_lev = Button(master=self.fnp, text='Subgraph level', font=font, command=lambda num=frame.number_table: subgraph_levels(frame, num, usable_width), width=button_width)
        self.subgraph_ind_sub_lev.grid(row=pos, column=0, pady=2, padx=5)
        self.frame_but = Frame(self.fnp)



        #table part, create and fill columns
        self.columnNames = columnNames

        self.model = TableModel()
        self.model.data = {}
        for i in xrange(len(self.columnNames)):
            if(direction):
                if(columnNames[i] != "Information centrality"):
                    self.model.addColumn(self.columnNames[i])
            else:
                if(columnNames[i] != 'Eccentricity in'):
                    if(columnNames[i] != 'Vibrational centrality in'):
                        if(columnNames[i] != 'In degree'):
                            if(columnNames[i] != 'Total degree'):
                                self.model.addColumn(self.columnNames[i].replace(" out", "").replace("Out degree", "Degree"))

        if(usable_width > 1500):
            thefont=30
            rowheight=22

        else:
            thefont=5
            rowheight=20

        self.table = TableCanvas(self.ft, model=self.model, rows=0, cols=0, rowheaderwidth=0, rowheight=rowheight,
                                 cellbackgr='#ffffff', thefont=thefont, editable=False, showkeynamesinheader=True,
                                  autoresizecols=True, fill=X)

        self.model.addRow()
        self.table.createTableFrame()

        self.file_name = StringVar()
        self.file_name_label = Label(self.ftable, textvariable=self.file_name)

        self.file_name_label.pack(side=TOP)
        self.ft.pack(side=LEFT, expand=True, fill=X)
        self.fnp.pack(side=RIGHT)

        self.ftable.pack(expand=True, fill=X)

        #create the centrality list
        self.centrality = [[] for x in xrange(len(self.columnNames))]

        self.table.fontsize=thefont
        self.table.setFontSize()
        self.adj_matrix = None
        self.laplacian_psinv = None


        frame.canvas.config(scrollregion=frame.canvas.bbox("all"))

#plot the ditribution of the centrality index
def plot_curveMain(self, tit):
    pos = findPos(tit)

    if(os.name == "nt"):
        path = "temp\\"+ tit + "_hist"
    else:
        path = "temp/"+ tit + "_hist"
    

    plt = plot_curve(self, pos, tit, path)
    plt.show()

# plot the coloured graph with each node coloured with its normalized value of the choosen centrality index
def plot_coeffMain(self, tit):
    import matplotlib.pyplot as plt
    import cairo
    from igraph.drawing.text import TextDrawer

    pos = findPos(tit)
    for i in range(self.number_table):
        plt = plot_coefficient(self, i, pos)
        # Save the plot
        plt.save("temp\\" + tit + "_graph.png")
        plt.show()


class Application(Frame):
    # constructor that creates graphical objects
    def __init__(self, networkFile, master, usable_width):
        #maximum number of tables
        self.table_max_num = 5

        #scrollbars
        self.vsbv = Scrollbar(master, orient="vertical")
        self.vsbh = Scrollbar(master, orient="horizontal")
        #frame and canvas
        self.fl = Frame(master)
        self.ft = Frame(master)
        self.canvas = Canvas(master,  highlightthickness=0,  yscrollcommand= self.vsbv.set)
        self.ftables = Frame(self.canvas)
        self.fl = Frame(master)
        self.ft = Frame(master)

        #number of tables
        self.number_table = 0

        #file chooser part
        self.choose_label = Label(self.ft, text="Network file: ")
        self.choose_label.grid(row=0, column=0, pady=2, padx=5)
        self.txt_box = Entry(self.ft, text = "", width=60)
        self.txt_box.grid(row=0, column=1, pady=2, padx=5)
        self.choose_file_button = Button(master=self.ft, text='Choose', command=lambda: choose_file(self, usable_width), width=10)
        self.choose_file_button.grid(row=0, column=2, pady=2, padx=5)

        #area
        # self.txt_area = ScrolledText(fr, undo=True, width=50, height=20)
        # self.txt_area['font'] = ('consolas', '12')
        # self.txt_area.config(state=DISABLED)
        # self.txt_area.pack(expand=True, fill='both')

        # buttons
        if(usable_width > 1500)       :
            button_width=27
            button_width_plot=7
            font=('Verdana',9)
        elif(usable_width > 1200):
            button_width=25
            button_width_plot=6
            font=('Verdana',10)
        else:
            button_width=18
            button_width_plot=5
            font=('Verdana',8)

        #buttons for centrality measures
        pos=0
        self.evaluate_label = Label(self.fl, text="Evaluate coefficients", font=font)
        self.evaluate_label.grid(row=(pos), column=0, columnspan=2, pady=2, padx=5)
        pos+=1
        self.out_degree_button = Button(master=self.fl, text='Out degree', font=font, command=lambda:  out_degreeTT(self, 'Out degree'), width=button_width)
        self.out_degree_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button1 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Out degree"), width=button_width_plot)
        self.plot_button1.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button1 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Out degree"), width=button_width_plot)
        self.plot_button1.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.in_degree_button = Button(master=self.fl, text='In degree', font=font, command=lambda: in_degreeTT(self, 'In degree'), width=button_width)
        self.in_degree_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button2 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "In degree"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button2 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "In degree"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.in_degree_button = Button(master=self.fl, text='Total degree', font=font, command=lambda: total_degreeTT(self, 'Total degree'), width=button_width)
        self.in_degree_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button2 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Total degree"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Total degree"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.eccentricity_button = Button(master=self.fl, text='Eccentricity out', font=font, command=lambda: eccentricity_out_TT(self, "Eccentricity out"), width=button_width)
        self.eccentricity_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button2 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Eccentricity out"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Eccentricity out"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.eccentricity_in_button = Button(master=self.fl, text='Eccentricity in', font=font, command=lambda: eccentricity_in_TT(self, "Eccentricity in"), width=button_width)
        self.eccentricity_in_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button2 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Eccentricity in"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Eccentricity in"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.eccentricity_in_button = Button(master=self.fl, text='Entropy', font=font, command=lambda: entropyTT(self, "Entropy"), width=button_width)
        self.eccentricity_in_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button2 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Entropy"), width=button_width_plot)
        self.plot_button2.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Entropy"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.betweenness_button = Button(master=self.fl, text='Betweenness', font=font, command=lambda: betweennessTT(self, 'Betweenness'), width=button_width)
        self.betweenness_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button3 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Betweenness"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Betweenness"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.clustering_coefficient_button = Button(master=self.fl, text='Clustering coefficient', font=font, command=lambda: clustering_coefficientTT(self, 'Clustering coefficient'), width=button_width)
        self.clustering_coefficient_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button4 = Button(master=self.fl, text='Plot', font=font, command=lambda: plot_curveMain(self, "Clustering coefficient"), width=button_width_plot)
        self.plot_button4.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Clustering coefficient"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.vibrational_centrality_button = Button(master=self.fl, text='Vibrational centrality out', font=font, command=lambda: vibrational_centrality_out_TT(self, 'Vibrational centrality out'), width=button_width)
        self.vibrational_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button5 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Vibrational centrality out"), width=button_width_plot)
        self.plot_button5.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Vibrational centrality out"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.vibrational_centrality_in_button = Button(master=self.fl, text='Vibrational centrality in', font=font, command=lambda: vibrational_centrality_in_TT(self, 'Vibrational centrality in'), width=button_width)
        self.vibrational_centrality_in_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button5 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Vibrational centrality in"), width=button_width_plot)
        self.plot_button5.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Vibrational centrality in"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self. informational_centrality_button = Button(master=self.fl, text='Informational centrality', font=font, command=lambda: information_centralityTT(self, "Information centrality"), width=button_width)
        self.informational_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button6 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Information centrality"), width=button_width_plot)
        self.plot_button6.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button3 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Information centrality"), width=button_width_plot)
        self.plot_button3.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.closeness_centrality_button = Button(master=self.fl, text='Closeness centrality out', font=font, command=lambda: closeness_centrality_out_TT(self, "Closeness centrality out"), width=button_width)
        self.closeness_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button7 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Closeness centrality"), width=button_width_plot)
        self.plot_button7.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button7 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Closeness centrality"), width=button_width_plot)
        self.plot_button7.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.closeness_centrality_button = Button(master=self.fl, text='Closeness centrality in', font=font, command=lambda: closeness_centrality_in_TT(self, "Closeness centrality in"), width=button_width)
        self.closeness_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button7 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Closeness centrality"), width=button_width_plot)
        self.plot_button7.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button7 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Closeness centrality"), width=button_width_plot)
        self.plot_button7.grid(row=pos, column=2, pady=1, padx=1)
        pos+=1
        self.subgraph_centrality_button = Button(master=self.fl, text='Subgraph centrality', font=font, command=lambda: subgraph_centralityTT(self, "Subgraph centrality"), width=button_width)
        self.subgraph_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        self.plot_button8 = Button(master=self.fl, text='Hist', font=font, command=lambda: plot_curveMain(self, "Subgraph centrality"), width=button_width_plot)
        self.plot_button8.grid(row=pos, column=1, pady=1, padx=1)
        self.plot_button7 = Button(master=self.fl, text='Graph', font=font, command=lambda: plot_coeffMain(self, "Subgraph centrality"), width=button_width_plot)
        self.plot_button7.grid(row=pos, column=2, pady=1, padx=1)
        pos +=1
        self.diameter_button = Button(master=self.fl, text='Diameter', font=font, command=lambda: diameter(self), width=button_width)
        self.diameter_button.grid(row=pos, column=0, pady=2, padx=5)
        pos+=1
        self.subgraph_centrality_button = Button(master=self.fl, text='Evaluate all', font=font, command=lambda: betweenness(self), width=button_width)
        self.subgraph_centrality_button.grid(row=pos, column=0, pady=2, padx=5)
        pos+=1

        #pack the frames
        self.ft.pack(side=TOP)
        self.fl.pack(side=LEFT)
        self.vsbv.pack(side="right", fill="y")
        self.vsbh.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.vsbv.config(command=self.canvas.yview)
        self.window = self.canvas.create_window((0,0), window=self.ftables, anchor="nw")

        #scroll bar control
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (self.ftables.winfo_reqwidth(), self.ftables.winfo_reqheight())
            self.canvas.config(scrollregion="0 0 %s %s" % size)
            if self.ftables.winfo_reqwidth() != self.canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                self.canvas.config(width=self.ftables.winfo_reqwidth())
        self.ftables.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if self.ftables.winfo_reqwidth() != self.canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                self.canvas.itemconfigure(self.window, width=self.canvas.winfo_width())
        self.canvas.bind('<Configure>', _configure_canvas)

        #list of objects of type graph
        self.graphList = []
        #g = my_graph(self, usable_width)
        #self.graphList.append(g)

        #load a network if passed by parameters
        if networkFile != "":
            self.txt_box.delete(0, END)
            self.txt_box.insert(END, networkFile)
            self.graphList[self.number_table].graph = Graph.Read_Ncol(networkFile, names=True, directed=True, weights="if_present")
            self.graphList[self.number_table].graph.vs["label"] = self.graphList[self.number_table].graph.vs["name"]
            self.file_name.set("File loaded: " + networkFile[networkFile.rfind("/")+1:])

            self.nodes = self.graphList[self.number_table].graph.vcount()
            self.arcs = self.graphList[self.number_table].graph.ecount()

            self.graphList[self.number_table].number_nodes.set(self.nodes)
            self.graphList[self.number_table].number_arcs.set(self.arcs)

            #create rows for the table
            nodes = self.graphList[self.number_table].graph.vcount()
            for i in xrange(nodes):
                self.graphList[self.number_table].table.addRow(self.graphList[self.number_table].graph.vs["label"][i])

            self.graphList[self.number_table].model.deleteRow(0)
            self.graphList[self.number_table].table.redrawTable()

            self.number_table += 1

        #frame.graphList[0].centrality[pos] = Graph.eccentricity(frame.graphList[0].graph)

# the main of the program
if __name__ == '__main__':

    argv=argv[1:]
    # set-up a format for the logging prints in BOINC style
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)

    network_file = ""
    # check input params
    if len(argv) > 0:
        try:
            network_file = str(argv[0])
        except:
            logging.error("Invalid input file")

        if not (isfile(network_file)):
            logging.critical("The file \"" + network_file + "\" is not valid")
            exit(1)

    # check the debug level
    #	0 : print only critical
    #	1 : print error and critical
    #	2 : print warning, error and critical
    #	3 : print info, warning, error and critical
    #	4 : print debug, info, warning, error and critical
    if len(argv) > 1:
        try:
            debug_level = int(argv[1])
        except:
            logging.error("Invalid debug level \"" + argv[1] + "\", set default value: 2 (warning)")
            debug_level = 2
    else:
        #logging.warning("No debug level found, set default value: 2 (warning)")
        debug_level = 2

    if (debug_level < 0) or (debug_level > 4):
        logging.warning("Wrong debug level \"" + str(debug_level) + "\", set default value: 2 (warning)")
        debug_level = 2

    # set the logging level accordingly
    if debug_level == 0:
        # set the logging level to CRTICAL
        logging.getLogger().setLevel(logging.CRTICAL)

    # set the logging level accordingly
    if debug_level == 1:
        # set the logging level to ERROR
        logging.getLogger().setLevel(logging.ERROR)

    if debug_level == 2:
        # set the logging level to WARNING
        logging.getLogger().setLevel(logging.WARNING)

    if debug_level == 3:
        # set the logging level to INFO
        logging.getLogger().setLevel(logging.INFO)

    if debug_level == 4:
        # set the logging level to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)

    # GUI class call
    finestra = Tk()
    finestra.title("Network analyzer")

    finestra.state('zoomed')
    finestra.update()
    usable_width  = finestra.winfo_screenwidth()
    usable_height = finestra.winfo_screenheight()


    center_window(finestra, usable_width, usable_height)
    app = Application(network_file, finestra, usable_width)
    finestra.mainloop()


