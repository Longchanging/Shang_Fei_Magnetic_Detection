# a slight modification of your code using multiprocessing
import matplotlib
import matplotlib.pyplot as plt 
# import threading
# let's try using multiprocessing instead of threading module:
import multiprocessing
import time

# we'll keep the same plotting function, except for one change--I'm going to use the multiprocessing module to report the plotting of the graph from the child process (on another core):
def plot_a_graph():
    f, a = plt.subplots(1)
    line = plt.plot(range(10))
    print (multiprocessing.current_process().name, "starting plot show process")  # print statement preceded by true process name
    plt.show()  # I think the code in the child will stop here until the graph is closed
    print (multiprocessing.current_process().name, "plotted graph")  # print statement preceded by true process name
    time.sleep(4)

# use the multiprocessing module to perform the plotting activity in another process (i.e., on another core):
job_for_another_core = multiprocessing.Process(target=plot_a_graph, args=())
job_for_another_core.start()

# the follow print statement will also be modified to demonstrate that it comes from the parent process, and should happen without substantial delay as another process performs the plotting operation:
print (multiprocessing.current_process().name, "The main process is continuing while another process deals with plotting.")
