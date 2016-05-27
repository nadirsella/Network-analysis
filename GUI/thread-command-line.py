#!/usr/bin/python

__author__ = 'Nadir Sella'
__version__ = '0'
__date__ = ''
import logging
import getopt
import shutil
import sys
from os.path import isfile
import numpy as np
from  threading import Thread
from thread_func import *
from multiprocessing import Pool
import time

columnNames = ['Name', 'Activates', 'Inhibits', 'Activated', 'Inhibited', 'Out degree', 'In degree', 'Total degree', 'Eccentricity out', 'Eccentricity in', 'Entropy', 'Betweenness', 'Clustering coefficient', 'Vibrational centrality out', 'Vibrational centrality in',
                            'Information centrality', 'Closeness centrality out', 'Closeness centrality in', 'Subgraph centrality']

single_params=["Diameter", "Assortativity"]

#print the matrix
def pri(mat):
    print '\n'.join(', '.join(str(round(y, 3)) for y in x) for x in mat)


def vibr2(self):
     for n in xrange(self.number_table):
        if(self.graphList[n] != None):
            len_matrix = len(self.graphList[n].graph.vs["name"])
            #laplacian matrix
            l = self.graphList[n].graph.laplacian(weights=self.graphList[n].graph.es["weight"], normalized=False)
            a = np.array(l)
            res = []
            for i in xrange(len_matrix):
                res.append([])
                for j in xrange(len_matrix):
                    res[i][j] = linalg.det(a) / linalg.det(a)




# find the indiced subgraph
def induced_subgraph(self , i):
    vertex = self.graphList[i].table.get_selectedRecordNames()
    print vertex
    self.graph_in_memory = self.graphList[i].graph.subgraph(vertex)
    plot_graph(self, self.subgraph)


# find the subgraph at a certain depth
def subgraph_levels(self, i):

    Set_value(self, "Insert depth", "Insert search depth")
    print "value: " + str(i)

    vertex = self.graphList[i].table.get_selectedRecordNames()
    try:
        val = int(self.value)
        list_list = self.graphList[i].graph.neighborhood(vertices=vertex, order=int(self.value), mode="out")
        merged = list(itertools.chain.from_iterable(list_list))
        self.graph_in_memory = self.graphList[i].graph.subgraph(merged)
    except:
        print "error"



def save_network(network):
    print "ciao"


def plot_histogram(self,pos, plot_name):
    for i in xrange(self.number_table):
        map = Counter(self.graphList[i].centrality[pos])

        plt.bar(map.keys(), map.values(), color=colors[i])
        plt.axis([-0.5, int(max(map.keys())+2), -5,int(max(map.values())+2)])

        #plt.xticks(range(len(map)), map.keys())

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(plot_name)
    fig = plt.figure()
    #plt.show()
    fig.savefig(plot_name, dpi=fig.dpi)


def plot_chart(self,pos, plot_name):
    lx=[]
    ly=[]
    for i in xrange(self.number_table):
        map = Counter(self.graphList[i].centrality[pos])

        plt.plot(map.keys(), map.values(), 'o', color=colors[i])
        plt.axis([-0.5, int(max(map.keys())+2), -5,int(max(map.values())+2)])


    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(plot_name)
    plt.show()

# plot the graph
def plot_graph(self, i):
    g = self.graphList[i].graph
    if g == None:
        no_valid_network_message()
    else:
        layout = g.layout("kamada_kawai")
        visual_style = {}
        visual_style["vertex_size"] = 20
        #visual_style["vertex_color"] = [color_dict[gender] for gender in g.vs["gender"]]
        visual_style["vertex_label"] = g.vs["name"]
        visual_style["edge_width"] = [float(weight) for weight in g.es["weight"]]
        visual_style["layout"] = layout
        visual_style["bbox"] = (800, 800)
        visual_style["margin"] = 20
        plot(g, **visual_style)

# remove the network from the list of networks, it is not deleted, a boolean is put as deleted and the memory is freed
def delete_network(self, i):
    print i
    self.graphList[i].deleted = True
    self.graphList[i].ftable.pack_forget()
    self.graphList[i] = None


def load_network(self, usable_width):
    Set_value(self, "Insert name", "Insert graph name")
    if(isinstance(self.value, str)):
        g = my_graph(self, usable_width)
        self.graphList.append(g)
        self.graphList[self.number_table].graph = self.graph_in_memory
        addGraph(self, self.value,1)
    else:
        print "Error"

#create the graph object
class my_graph(object):
    def __init__(self, frame, direction, pos):
        #create the centrality list
        self.centrality = [[] for x in xrange(len(columnNames))]
        self.laplacian_psinv = None
        self.values = [0 for x in xrange(len(single_params))]
        self.directed = direction[pos]


#evaluates the centrality measures
class Application(object):
    # metodo costruttore che crea gli oggetti grafici
    def __init__(self, argv, direction, out_folder, number_thread, plot):
        self.graphList = []
        #maximum number of tables
        self.table_max_num = 5
        #number of tables
        self.number_table = 0

        #load the networks passed by parameters
        for networkFile in argv:
            if networkFile != "":
                if isfile(networkFile):


                    if(os.name == "nt"):
                        name= networkFile[networkFile.rfind("\\")+1:]
                    else:
                        name= networkFile[networkFile.rfind("/")+1:]

                    find = networkFile.rfind(".")

                    if(find > 0):
                        name = name[:find]


                    if(os.stat(networkFile)[6]==0):
                        if(os.name == "nt"):
                            path = out_folder +"\\"+ name + ".csv"
                        else:
                            path = out_folder +"/"+ name + ".csv"
                        out_file = open(path, 'w')
                        out_file.close()
                        if(os.name == "nt"):
                            path = out_folder +"\\"+ name + "_single_values.csv"
                        else:
                            path = out_folder +"/"+ name + "_single_values.csv"
                        out_file = open(path, 'w')
                        out_file.close()
                    else:

                        g = my_graph(self, direction, self.number_table)
                        self.graphList.append(g)
                        if(direction):
                            self.graphList[self.number_table].graph = Graph.Read_Ncol(networkFile, names=True, directed=True, weights="if_present")
                        else:
                            self.graphList[self.number_table].graph = Graph.Read_Ncol(networkFile, names=True, directed=False, weights="if_present")

                        if(os.name == "nt"):
                            self.graphList[self.number_table].name= networkFile[networkFile.rfind("\\")+1:]
                        else:
                            self.graphList[self.number_table].name= networkFile[networkFile.rfind("/")+1:]

                        find = self.graphList[self.number_table].name.rfind(".")

                        if(find > 0):
                            self.graphList[self.number_table].name = self.graphList[self.number_table].name[:find]
                        
                        self.number_table += 1

        logname=""
        for i in xrange(self.number_table):
            logname += "_" + self.graphList[i].name

        #name of the log file
        if(os.name == "nt"):
            filename = out_folder + "\\file_log" + logname + ".txt"
        else:
            filename = out_folder + "/file_log" + logname + ".txt"

        #log to file
        handler = logging.FileHandler(os.path.join(filename),"w", encoding=None, delay="true")
        formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)


        #evaluates out/in/total/activators/inhibitors, they are not evaluated as threads
        for i in range(self.number_table):
            graph = self.graphList[i].graph
            
            
            if(self.graphList[i].directed == True):
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
                self.graphList[i].centrality[pos] = countActivations
                print("Activates evaluated")
                pos = findPos("Inhibits")
                self.graphList[i].centrality[pos] = countInhibitions
                print("Inhibits evaluated")
                pos = findPos("Activated")
                self.graphList[i].centrality[pos] = countActivated
                print("Activated evaluated")
                pos = findPos("Inhibited")
                self.graphList[i].centrality[pos] = countInhibited
                print("Inhibited evaluated")

            if(self.number_table > 0):
                pos = findPos("Out degree")
                self.graphList[i].centrality[pos] = Graph.outdegree(graph)
                print("Out degree evaluated")
                pos = findPos("In degree")
                self.graphList[i].centrality[pos] = Graph.indegree(graph)
                print("In degree evaluated")
                pos = findPos("Total degree")
                self.graphList[i].centrality[pos] = Graph.degree(graph)
                print("Total degree evaluated")

            graph.oldes = graph.es["weight"]
            graph.es["weight"] = [abs(x) for x in (graph.es["weight"])]

            #create the thread pool
            threads=[]
            threads_single_param=[]

            logging.info("Start evaluating indexes\n")        

            #create thread pool
            pool = Pool(processes=number_thread)

            tititi = time.time()

            p3 = pool.map_async(func = eccentricity_out, iterable = [(self)])
            threads.append(p3)

            p3_in = pool.map_async(func = eccentricity_in, iterable = [(self)])
            threads.append(p3_in)

            p_en = pool.map_async(func = entropy, iterable = [(self)])
            threads.append(p_en)

            p4 = pool.map_async(func = betweenness, iterable = [(self)])
            threads.append(p4)

            p5 = pool.map_async(func = clustering_coefficient, iterable = [(self)])
            threads.append(p5)

            logging.info("Time up to vibration:" + str(time.time() - tititi))

            p6 = pool.map_async(func = vibrational_centrality_out_normalized, iterable = [(self)])
            threads.append(p6)

            p7 = pool.map_async(func = vibrational_centrality_in_normalized, iterable = [(self)])
            threads.append(p7)

            p8 = pool.map_async(func = information_centrality, iterable = [(self)])
            threads.append(p8)

            p9 = pool.map_async(func = closeness_out, iterable = [(self)])
            threads.append(p9)

            p10 = pool.map_async(func = closeness_in, iterable = [(self)])
            threads.append(p10)

            p11 = pool.map_async(func = subgraph_centrality, iterable = [(self)])
            threads.append(p11)


            #create threads that return params
            p12 = pool.map_async(func = diameter, iterable = [(self)])
            threads_single_param.append(p12)

            p13 = pool.map_async(func = assortativity, iterable = [(self)])
            threads_single_param.append(p13)

            # close and wait all the threads
            pool.close()
            pool.join()

            # fill the centrality vectors with the values returned by the threads
            for i in range(len(threads)):
                try:
                    l_l_l = threads[i].get()
                    for l_l in l_l_l:
                        n=0
                        for l in l_l:
                            # +3 because 5 centralities are not launched as threads
                            self.graphList[n].centrality[i+7] = l
                            n+=1
                except Exception as e:
                    logging.error("Error on " + str(columnNames[i+8]) +": " + str(e));
                    for n in range(self.number_table):
                        self.graphList[n].centrality[i] = []

            # fill the single parameters
            for i in range(len(threads_single_param)):
                try:
                    l_l = threads_single_param[i].get()
                    for l in l_l:
                        n = 0
                        for val in l:
                            self.graphList[n].values[i] = val
                            n +=1
                except Exception as e:
                    logging.error("Error on " + str(single_params[i]) +": " + str(e));
                    for n in range(self.number_table):
                        self.graphList[n].values[i] = ""

            ####################################################################
            ###  print data to file
            ####################################################################
            i=0
            try:
                for i in xrange(self.number_table):

                    if(os.name == "nt"):
                        path = out_folder +"\\"+ self.graphList[i].name + ".csv"
                    else:
                        path = out_folder +"/"+ self.graphList[i].name + ".csv"
                    out_file = open(path, 'w')

                    out_file.write(columnNames[0])

                    for j in xrange(1,len(columnNames)):
                        if(direction[i] == True):
                            if(columnNames[j] != "Information centrality"):
                                out_file.write("," + columnNames[j])
                        else:
                                if(columnNames[j] != "Eccentricity in"):
                                    if(columnNames[j] != "Vibrational centrality in"):
                                        if(columnNames[i+1] != 'In degree'):
                                            if(columnNames[i+1] != 'Total degree'):
                                                if(len(self.graphList[i].centrality[n]) > 0):
                                                    out_file.write("," + columnNames[j].replace(" out", "").replace("Out degree", "Degree"))

                    if(len(self.graphList[i].centrality[len(columnNames)-1]) > 0):
                        out_file.write(columnNames[len(columnNames)-1])
                    out_file.write("\n")

                    for r in xrange(len(self.graphList[i].centrality[0])):
                        out_file.write(self.graphList[i].graph.vs["name"][r])
                        # to remove -2
                        for n in xrange(len(columnNames) -1):
                            if(direction[i] == True):
                                if(columnNames[n+1] != "Information centrality"):
                                    if(len(self.graphList[i].centrality[n]) > 0):
                                        out_file.write("," + str(self.graphList[i].centrality[n][r]))

                            else:
                                if(columnNames[n+1] != 'Eccentricity in'):
                                    if(columnNames[n+1] != 'Vibrational centrality in'):
                                        if(columnNames[n+1] != 'In degree'):
                                            if(columnNames[n+1] != 'Total degree'):
                                                if(len(self.graphList[i].centrality[n]) > 0):
                                                    out_file.write(str("," + self.graphList[i].centrality[n][r]))

                        n = len(columnNames) - 1
                        if(len(self.graphList[i].centrality[n]) > 0):
                            out_file.write("," + str(self.graphList[i].centrality[n][r]))
                        out_file.write("\n")

                    out_file.close()
                    
                    if(os.name == "nt"):
                        path = out_folder +"\\"+ self.graphList[i].name + "_single_values.csv"
                    else:
                        path = out_folder +"/"+ self.graphList[i].name + "_single_values.csv"
                    out_file = open(path, 'w')

                    out_file.write("Number of nodes: " + str(len(self.graphList[i].graph.vs))+ "\n")
                    out_file.write("Number of edges: " + str(len(self.graphList[i].graph.es))+ "\n")
                    
                    #da sistemare
                    for n in xrange(len(columnNames) - 1):
                        if(direction[i] == True):
                            if(columnNames[n+1] != "Information centrality"):
                                if(len(self.graphList[i].centrality[n]) > 0):
                                    out_file.write(columnNames[n+1] + " average: " + str(avg(self.graphList[i].centrality[n]))+ "\n")
                        else:
                            if(columnNames[n+1] != 'Vibrational centrality in'):
                                if(columnNames[n+1] != 'In degree'):
                                    if(columnNames[n+1] != 'Total degree'):
                                        if(len(self.graphList[i].centrality[n]) > 0):
                                            out_file.write(columnNames[n+1].replace(" out", "").replace("Out degree", "Degree") + " average: " + str(avg(self.graphList[i].centrality[n]))+ "\n")


                    out_file.write("\n")
                    for idx, val in enumerate(self.graphList[i].values):
                        out_file.write(str(single_params[idx]) + ": " + str(val)+ "\n")

                    out_file.close()               

            except Exception as e:
                print e
                logging.error("Error on print phase: " + str(e))

            if(plot):
                plot_graphs(self, out_folder)

#usage of the progrtam
def usage():
    print "SYNOPSIS:"
    print "\command-line.py -n network -o result_directory -t threads_number -v verbosity -h help\n"
    print "PARAMS:"
    print "\t-n\tThe path of the network file.\n"
    print "\t-o\tThe path to the directory in which the results will be stored.\n"
    print "\t-d\tIf the network is directed: y/n.\n"
    print "\t-t\tThe number of threads to create.\n"
    print "\t-v\tThe verbosity of the log (optional)\n"
    print "\t-p\tPrint the charts of the distributions (optional/do not work on servers)\n"
    print "\t-h\tThis tutorial\n"

if __name__ == '__main__':

    tim = time.time()
    argv = sys.argv[1:]
    

    out_folder = ""
    verbosity = 0
    number_thread = 0
    plot=False

    # getopt part, take the parameters passed in command-line
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:d:o:t:hsp")
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    # lists for networks to evaluate
    out = []
    direction = []

    for o, v in opts:
        if o == "-n":
            out.append(v)
        if o == "-d":
            if(v == "y"):
                direction.append(True)
            if(v == "n"):
                direction.append(False)
        elif o == "-o":
            out_folder = v
        elif o == "-t":
            number_thread = v
        elif o == "-p":
            plot = True
        elif o == "-v":
            verbosity = v
        elif o == "-h":
            usage()
            sys.exit(2)

    # data check
    if(out_folder == ""):
        print "[ERR] The options -n -o must be specified\n"
        usage()
        sys.exit(2)

    if(len(out) == 0):
        print "[ERR] The options -n -o must be specified\n"
        usage()
        sys.exit(2)

    if(len(out) != len(direction)):
        print "[ERR] The options -d must be specified\n"
        usage()
        sys.exit(2)

    if(number_thread == 0):
        print "[ERR] Number of threads must be different from 0\n"
        usage()
        sys.exit(2)


    # check the debug level
    #   0 : print only critical
    #   1 : print error and critical
    #   2 : print warning, error and critical
    #   3 : print info, warning, error and critical
    #   4 : print debug, info, warning, error and critical

    if len(argv) > 2:
        try:
            debug_level = int(verbosity)
        except:
            logging.error("Invalid debug level \"" + argv[1] + "\", set default value: 2 (warning)")
            debug_level = 2
    else:
        #logging.warning("No debug level found, set default value: 2 (warning)")
        debug_level = 0

    try:
        number_thread = int(number_thread)
    except:
        number_thread = 1

    if (debug_level < 0) or (debug_level > 4):
        logging.warning("Wrong debug level \"" + str(debug_level) + "\", set default value: 2 (warning)")
        debug_level = 2


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


    #create new folder
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder)
    # else:
    #     if not raw_input("The working folder already exists, do you want to override the content? (Y/N)\n").lower() in ['y', 'yes']:
    #             sys.exit(0)
        # else:
        #     shutil.rmtree(out_folder)
        #     time.sleep(0.2)
        #     os.makedirs(out_folder)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)



    app = Application(out, direction, out_folder, number_thread, plot)

    logging.info("Total time: " + str(time.time()-tim))