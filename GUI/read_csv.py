import sys
import getopt
import time
import logging
import shutil
import matplotlib
import pylab
import os
from igraph.drawing.text import TextDrawer
import cairo
import colorsys
from tkintertable.Tables import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import random

colors = ['#e50000', '#393939', '#393939', '#393939', '#393939', '#393939', '#393939', '#393939']
colors = ['r', 'b', 'g', 'k', '#393939', '#393939', '#393939', '#393939']


def power_law(x, m, b):
    return b * (x**m)

def exponential(x, a, b):
    return a * np.exp(-b * x)

def gauss_function(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

# def format_e(n):
# 	print n
# 	a = '%E' % n
# 	return str(round(a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]), 4)

def format_e(a):
	s = ""
	if(a > 0):
		s = "+"

	num = round(a, 4)
	if(num == 0.0 or num > 100):	
		return s + str('{:.4e}'.format(float(a)))
	else:
		return s + str(num)

def func_print(method, coefs):
    p = ""
    if(method == 1):
        if(len(coefs) == 1):
            p = '$y = ' + format_e(coefs[0])+ '$'
        elif(len(coefs) == 2):
            p = '$y = ' + format_e(coefs[0]) + ' x ' + format_e(coefs[1]) + '$'
        elif(len(coefs) == 3):  
            p = '$y = ' + format_e(coefs[0]) + ' x ^ {2} ' + format_e(coefs[1]) + 'x ' + format_e(coefs[2]) + '$'
        elif(len(coefs) == 4):
            p = '$y = ' + format_e(coefs[0]) + ' x ^ {3} ' + format_e(coefs[1]) + ' x ^ {2} $' + '\n$' +format_e(coefs[2]) + ' x '+ format_e(coefs[3]) + '$'
        elif(len(coefs) == 5):
      	    p = '$y = ' + format_e(coefs[0]) + ' x ^ {4} ' + format_e(coefs[1]) + ' x ^ {3} $' + '\n$' + format_e(coefs[2]) + ' x ^ {2} '+ format_e(coefs[3]) + 'x $' + '\n$' + format_e(coefs[4]) + '$'
    elif(method == 2):
        p = '$y = ' + format_e(coefs[1]) + ' x ^{' + format_e(coefs[0]) + '}$'
    elif(method == 3):
           p = '$y = ' + format_e(coefs[0]) + ' e ^{' + format_e(-1*coefs[1]) + 'x} $'
    elif(method == 4):
            p = r'$y = ' + format_e(coefs[0]) + '$' + r'$ e ^ {\frac{-(x -' + (format_e(coefs[1])) + ') ^ {2} }{2' + format_e(coefs[2]) + ' ^ 2 } + format_e(coefs[3])}$'


    return p+"\n"

def plot_curves(self, folder):
    if(self.number_table > 0):
        for n in range(len(columnNames)-1):
            print columnNames[n+1]
            if(len(self.graphList[0].centrality[n]) > 0):
				tit = columnNames[n+1]
				path = ""
				if(os.name == "nt"):
					path = folder +"\\"+ columnNames[n+1]
				else:
					path = folder +"/"+ columnNames[n+1]
				plot_curve(self, n, tit, path)
				
			   
			   	#fig = plt.gcf()
				#fig.set_size_inches(15,9)
				

from scipy import asarray as ar,exp

def plot_curve(self, pos, tit, path):
	max_degree = 3

	fig = plt.figure()
	ax = plt.subplot(111)
	#ax =  fig.add_axes([0.1, 0.2, 0.4, 0.4])
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

	maxX = 0
	minX = 100000000

	for n in xrange(self.number_table):
		if(self.graphList[n] != None):
			l = self.graphList[n].centrality[pos]
			
			if(min(l) < minX):
				minX = min(l)

			if(max(l) > maxX):
				maxX = max(l)

	nbin = 50
	vec = []
	vec.append(minX)
	val = minX
	step = float(maxX-minX)/float(nbin)

	for i in xrange(nbin-1):
		val += step
		vec.append(val)

	vec.append(maxX)

	for n in xrange(self.number_table):
		if(self.graphList[n] != None):
			l = self.graphList[n].centrality[pos]

			#ax.hist(l, bins=np.arange(minX, maxX, 10))
			hist, bins = np.histogram(l, bins=vec)
			width = 1 * (bins[1] - bins[0])
			center = (bins[:-1] + bins[1:]) / 2
			ax.bar(center, hist, align='center', alpha=0.4, width=width, color=colors[n])

			x = np.array(center)
			y = np.array(hist)

			#print np.array(x)
			#print np.array(y)

			# m = dict((i,l.count(i)) for i in l)

			# y = np.array(m.values())
			# x = np.array(m.keys())

			# x = ar(range(10)) #gauss points
			# y = ar([0,1,2,3,4,5,4,3,2,1])     

			x_new = np.linspace(x[0], x[-1], num=len(x)*10)

			#ploynomial fit
			best = 0
			min_ = float('inf')
			i = 0
			while(i <= max_degree):
				coefs, val = poly.polyfit(x, y, i, full=True)
				#print str(i) + " *" +  str(val[0])+"*"

				if(val[0] < 1e-15 and min_ != 0):
					best=i
					method = 1
					min_ = 0

				if(len(val[0]) == 0 and min_ != 0):
					best=i
					method = 1
					min_ = 0
				elif(val[0] < min_ and coefs[0] != 0):
					best=i
					method = 1
					min_ = val[0]
				i+=1

			#print "rect" + str(min_)

			# power law
			try:
				popt, pcov = curve_fit(power_law, x, y, maxfev=1000000)
				residuals = y - power_law(x, *popt)
				fres = sum((residuals)**2)
				if(fres < min_):
					method = 2
					min_ = fres
			except Exception as e:
				print "No power-law: " + str(e)


			#print "pl " + str(fres)
			try:
			# exponential
				popt1, pcov = curve_fit(exponential, x, y, maxfev=1000000)
				residuals = y - exponential(x, *popt1)
				fres = sum((residuals)**2)
				if(fres < min_):
					method = 3

					min_ = fres
			except:
				print "no expopential"

			#print "exp " + str(fres)

			#gaussian
			try:
				len_ = len(x)                          #the number of data
				mean = sum(x*y)/len_                  #note this correction
				sigma = sum(y*(x-mean)**2)/len_
				popt2, pcov = curve_fit(gauss_function, x, y, maxfev=1000000)
				residuals = y - gauss_function(x, *popt2)
				fres = sum((residuals)**2)
				if(fres < min_):
					method = 4
				#print fres
				#print min_
				#print(fres < min_)
			except:
				print "no gaussian"

			#print "gaus " + str(fres)

			#hist, bins = ax.hist(x)


			#ax.plot(x, y, colors[n]+ 'o', label= self.graphList[n].name +" data")

			#print "meth: " + str(method)

			if(method == 1):
				coefs = poly.polyfit(x, y, best)
				coefs = coefs[::-1]    
				f = np.poly1d(coefs)
				ax.plot(x_new, f(x_new), colors[n], label= self.graphList[n].name + "\n" + func_print(method, coefs))           
			elif(method == 2):
				ax.plot(x_new, power_law(x_new, *popt), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt))
			elif(method == 3):
				ax.plot(x_new, exponential(x_new, *popt1), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt1))
			elif(method == 4):
				ax.plot(x_new, gauss_function(x_new, *popt2), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt2))

			print method
	ax.set_xlabel('Value')
	ax.set_ylabel('Frequency')

	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# ax.legend(loc='center left', bbox_to_anchor=(1, 1))
	#ax.axis([-0.05, 1.05, -0.05, 1.05])
	#ax.legend(loc='center left', bbox_to_anchor=(0.78, 0.95), prop={'size':9})
	ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.60), prop={'size':10})
	#ax.legend()

	plt.title(tit)
	plt.savefig(path + ".png", format='png', dpi=400)

	return plt


	# ####################################
	# #plot grafici singoli
	# xlim = ax.get_xlim()
	# ylim = ax.get_ylim()


	# if(self.number_table > 1):
	# 	for n in xrange(self.number_table):
	# 		if(self.graphList[n] != None):

	# 			fig = plt.figure()
	# 			ax = plt.subplot(111)
	# 			#ax =  fig.add_axes([0.1, 0.2, 0.4, 0.4])
	# 			box = ax.get_position()
	# 			ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

	# 			l = self.graphList[n].centrality[pos]

	# 			#ax.hist(l, bins=np.arange(minX, maxX, 10))
	# 			hist, bins = np.histogram(l, bins=vec)
	# 			width = 1 * (bins[1] - bins[0])
	# 			center = (bins[:-1] + bins[1:]) / 2
	# 			ax.bar(center, hist, align='center', alpha=0.4, width=width, color=colors[n])

	# 			x = np.array(center)
	# 			y = np.array(hist)

	# 			#print np.array(x)
	# 			#print np.array(y)

	# 			# m = dict((i,l.count(i)) for i in l)

	# 			# y = np.array(m.values())
	# 			# x = np.array(m.keys())

	# 			# x = ar(range(10)) #gauss points
	# 			# y = ar([0,1,2,3,4,5,4,3,2,1])     

	# 			x_new = np.linspace(x[0], x[-1], num=len(x)*10)
	# 			#x_new = np.linspace(x[0], x[-1], num=len(x)*100)
	# 			print(x[0], x[-1], max(l))
				
	# 			#ploynomial fit
	# 			best = 0
	# 			min_ = float('inf')
	# 			i = 0
	# 			while(i <= max_degree):
	# 				coefs, val = poly.polyfit(x, y, i, full=True)
	# 				#print str(i) + " *" +  str(val[0])+"*"

	# 				if(val[0] < 1e-15 and min_ != 0):
	# 					best=i
	# 					method = 1
	# 					min_ = 0

	# 				if(len(val[0]) == 0 and min_ != 0):
	# 					best=i
	# 					method = 1
	# 					min_ = 0
	# 				elif(val[0] < min_ and coefs[0] != 0):
	# 					best=i
	# 					method = 1
	# 					min_ = val[0]
	# 				i+=1

	# 			#print "rect" + str(min_)

	# 			# power law
	# 			try:
	# 				popt, pcov = curve_fit(power_law, x, y, maxfev=1000000)
	# 				residuals = y - power_law(x, *popt)
	# 				fres = sum((residuals)**2)
	# 				if(fres < min_):
	# 					method = 2
	# 					min_ = fres
	# 			except Exception as e:
	# 				print "No power-law: " + str(e)


	# 			#print "pl " + str(fres)
	# 			try:
	# 			# exponential
	# 				popt1, pcov = curve_fit(exponential, x, y, maxfev=1000000)
	# 				residuals = y - exponential(x, *popt1)
	# 				fres = sum((residuals)**2)
	# 				if(fres < min_):
	# 					method = 3

	# 					min_ = fres
	# 			except:
	# 				print "no expopential"

	# 			#print "exp " + str(fres)

	# 			#gaussian
	# 			try:
	# 				len_ = len(x)                          #the number of data
	# 				mean = sum(x*y)/len_                  #note this correction
	# 				sigma = sum(y*(x-mean)**2)/len_
	# 				popt2, pcov = curve_fit(gauss_function, x, y, maxfev=1000000)
	# 				residuals = y - gauss_function(x, *popt2)
	# 				fres = sum((residuals)**2)
	# 				if(fres < min_):
	# 					method = 4
	# 				#print fres
	# 				#print min_
	# 				#print(fres < min_)
	# 			except:
	# 				print "no gaussian"

	# 			#print "gaus " + str(fres)

	# 			#hist, bins = ax.hist(x)


	# 			#ax.plot(x, y, colors[n]+ 'o', label= self.graphList[n].name +" data")

	# 			#print "meth: " + str(method)

	# 			if(method == 1):
	# 				coefs = poly.polyfit(x, y, best)
	# 				coefs = coefs[::-1]    
	# 				f = np.poly1d(coefs)
	# 				ax.plot(x_new, f(x_new), colors[n], label= self.graphList[n].name + "\n" + func_print(method, coefs))           
	# 			elif(method == 2):
	# 				ax.plot(x_new, power_law(x_new, *popt), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt))
	# 			elif(method == 3):
	# 				ax.plot(x_new, exponential(x_new, *popt1), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt1))
	# 			elif(method == 4):
	# 				ax.plot(x_new, gauss_function(x_new, *popt2), colors[n], label= self.graphList[n].name + "\n" + func_print(method, popt2))


	# 			ax.set_xlim(xlim)
	# 			ax.set_ylim(ylim)


	# 			ax.set_xlabel('Value')
	# 			ax.set_ylabel('Frequency')

	# 			# box = ax.get_position()
	# 			# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# 			# ax.legend(loc='center left', bbox_to_anchor=(1, 1))
	# 			#ax.axis([-0.05, 1.05, -0.05, 1.05])
	# 			ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':9})

	# 			plt.title(tit)
	# 			plt.savefig(path + "_" + self.graphList[n].name + ".png", format='png', dpi=300)




def plot_graphs(self, folder):
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


                    plt.scatter(self.graphList[tab].centrality[2],self.graphList[tab].centrality[6])

                    plt.show()


                    plot_coefficient(self, tab, n , path)




columnNames = ['Name', 'Activates', 'Inhibits', 'Activated', 'Inhibited', 'Out degree', 'In degree', 'Total degree', 'Eccentricity out', 'Eccentricity in', 'Entropy', 'Betweenness', 'Clustering coefficient', 'Vibrational centrality out', 'Vibrational centrality in',
                            'Information centrality', 'Closeness centrality', 'Subgraph centrality']

class my_graph(object):
    def __init__(self):
        #create the centrality list
        self.centrality = [[] for x in xrange(len(columnNames))]


class Application(object):
	def __init__(self, files, out_folder):
		self.number_table = len(files);

		self.graphList = []
		self.number_table = 0
		for fil in files:
			g = my_graph()
			self.graphList.append(g)

			if(os.name == "nt"):
				self.graphList[self.number_table].name= fil[fil.rfind("\\")+1:]
				pos = self.graphList[self.number_table].name.rfind(".")
				if(pos > 0):
					self.graphList[self.number_table].name = self.graphList[self.number_table].name[:pos]
			else:
				self.graphList[self.number_table].name= fil[fil.rfind("/")+1:]
				pos = self.graphList[self.number_table].name.rfind(".")
				if(pos > 0):
					self.graphList[self.number_table].name = self.graphList[self.number_table].name[:pos]				
			with open(fil) as f:
				for row in f.readlines()[1:]:
					l = (row.replace('\n', '').split(','))[1:]
					#print l
					i = 0
					for val in l:
						self.graphList[self.number_table].centrality[i].append(float(val))
						i += 1
				
			#print self.graphList[self.number_table].centrality[0]

			self.number_table += 1

			print len(self.graphList)

		#try:
		plot_curves(self, out_folder)
		#plot_graphs(self, out_folder)




			# matplotlib.pyplot.scatter(self.graphList[self.number_table].centrality[2], self.graphList[self.number_table].centrality[6])
			# matplotlib.pyplot.show()

			


def usage():
    print "SYNOPSIS:"
    print "\command-line.py -n network -o result_directory -h help\n"
    print "PARAMS:"
    print "\t-n\tThe path of the network file.\n"
    print "\t-o\tThe path to the directory in which the results will be stored.\n"
    print "\t-h\tThis tutorial\n"


if __name__ == '__main__':

	tim = time.time()
	argv = sys.argv[1:]
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	handler = logging.FileHandler(os.path.join("file.log"),"w", encoding=None, delay="true")
	formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	files = []

	try:
		opts, args = getopt.getopt(sys.argv[1:], "n:o:hs")
	except getopt.GetoptError as err:		
		# print help information and exit:
		print str(err)  # will print something like "option -a not recognized"
		usage()
		sys.exit(2)

	for o, v in opts:
		if o == "-n":
			files.append(v)
		elif o == "-o":
			out_folder = v
		elif o == "-h":
		    usage()
		    sys.exit(2)

	#create new folder
	if not os.path.exists(out_folder): 
		os.makedirs(out_folder)
	else:
		if not raw_input("The working folder already exists, do you want to override the content? (Y/N)\n").lower() in ['y', 'yes']:
			sys.exit(0)
		else:
			shutil.rmtree(out_folder)
			time.sleep(0.2)
			os.makedirs(out_folder)

	Application(files, out_folder)
