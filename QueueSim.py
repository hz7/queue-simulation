#!/usr/bin/env python

"""
usage: QueueSim input_file output_file

input_file is path for json formated file specifying these simulation parameters:
nservers - number of servers
capacity - line capacity, customer leaves if line size exceed capacity, can be infinite
arrival_rate - inter-arrivals times are exponential with rate arrival_rate
service_rate - service times are exponential with rate service_rate
t_lim - how long to run simulation, observe arrivals until t_lim
max_record - stats will be given for first max_record states (stationary distribution)
Ex.
{
    "nservers" : 1,
    "capacity" : 5,
    "arrival_rate" : 1,
    "service_rate" : 1.5,
    "t_lim" : 500,
    "max_record" : 10
}

output_file is path where simulation results are saved.

Wrote this script to get practice with Python.
Probably a more efficient way here:
http://www.scipy.org/Cookbook/Solving_Large_Markov_Chains
"""

import json, numpy as np, sys
from collections import deque

class QueueSim:
    """
    M/M/m/K Queue simulation.
    """

    def __init__(self, params):
        """params is a dict containing the simulation params, see usage for explanation"""
        self.params = params
        self.nservers = int(params["nservers"])
        self.capacity = int(params["capacity"])
        self.arrival_rate = float(params["arrival_rate"])
        self.service_rate = float(params["service_rate"])
        self.t_lim = float(params["t_lim"])
        self.max_record = int(params["max_record"])

    def pregen(self):
        """
        Pre-generate arrival times up to t_lim.
        Note waiting times between arrivals is exponential -- scale parameter is inverse of rate.
        
        Also pre-generate service times for each arrival
        """
        t = 0
        arrival_times = []
        while True:
            t += np.random.exponential(1 / self.arrival_rate)
            if t < self.t_lim:
                arrival_times.append(t)
            else:
                break
        self.arrival_times = np.hstack(arrival_times) #note this is an increasing array of times
        self.narrivals = len(self.arrival_times)
        self.service_times = np.random.exponential(1 / self.service_rate, self.narrivals)

    def sim(self):
        """
        Simulate a M/M/s queuing process from pre-computed arrival and service times
        for each customer.
        """
        self.pregen()

        self.departure_times = np.zeros(self.narrivals) #when a customer finishes (either balking/finished service)
        self.t = np.zeros(2 * self.narrivals) #times of each arrival or departure event
        self.states = np.zeros(2 * self.narrivals) #number of customers in system at each event
        self.balk = np.zeros(self.narrivals, dtype=np.bool) #indicates if a customer leaves the line because of full capacity
        self.balkEventMask = np.zeros(2 * self.narrivals) #for balk events
        t_current = 0 #clock
        idx_next_arrival = 0 #index of next arrival
        idx_being_served = [] #indices of current customers being served
        idx_waiting = deque() #indices of customers waiting in line, service is first come first served

        #simulation looks at the embedded markov chain over arrival/departure events
        i = 0
        while i < 2 * self.narrivals:
            #Compute next arrival
            if idx_next_arrival < self.narrivals:
                t_next_arrival = self.arrival_times[idx_next_arrival]
            else:
                t_next_arrival = float("inf")

            #Compute next departure
            t_next_departure = float("inf")
            for customer_idx in idx_being_served:
                if self.departure_times[customer_idx] < t_next_departure:
                    t_next_departure = self.departure_times[customer_idx]
                    idx_next_departure = customer_idx

            #Current event is the arrival or departure that comes first
            #arrival takes precedent if t_next_arrival == t_next_departure
            is_arrival = t_next_arrival < t_next_departure
            if is_arrival:
                t_current = t_next_arrival
            else:
                t_current = t_next_departure
            if t_current == float("inf"):
                break #no more events

            #Move customers around
            if is_arrival:
                isBalk = len(idx_waiting) >= self.capacity
                if isBalk:
                    self.balk[idx_next_arrival] = True
                    self.departure_times[idx_next_arrival] = t_current
                else: #add arrival to line only if below capacity
                    idx_waiting.append(idx_next_arrival) 
                idx_next_arrival += 1
                if isBalk:
                    for j in range(2):
                        self.balkEventMask[i] = True
                        self.t[i] = t_current
                        self.states[i] = self.capacity
                        i += 1
                    continue
            else:
                idx_being_served.remove(idx_next_departure)

            #Finally, if possible, move a waiting customer from line into service
            spots_open = self.nservers - len(idx_being_served)
            if spots_open > 0 and len(idx_waiting) > 0:
                idx_first_customer_in_line = idx_waiting.popleft() #first come first served
                idx_being_served.append(idx_first_customer_in_line)
                self.departure_times[idx_first_customer_in_line] = t_current + self.service_times[idx_first_customer_in_line] #set departure time (now known)

            self.t[i] = t_current
            self.states[i] = len(idx_being_served) + len(idx_waiting) #number of customers in system from t[i] to t[i+1]
            i += 1


    def write_stats(self, filename):
        """ Write statistics to path filename. """
        output = open(filename, 'w')
        output.write("Simulation results for params {0}\n\n".format(self.params))
        #Stats will ignore customers who balk

        #Stationary distribution
        customer_times = self.t[np.logical_not(self.balkEventMask)]
        customer_states = self.states[np.logical_not(self.balkEventMask)]
        interval_lengths = np.diff(customer_times)
        cum_state = np.zeros(self.max_record) #keeps track of time in a particular state
        for i in range(self.max_record):
            cum_state[i] = np.sum(interval_lengths[customer_states[:-1] == i]) #last state is zero, exclude
        cum_state[0] += customer_times[0] #include the time up to first arrival for state 0
        stationary = cum_state / customer_times[-1]
        output.write("Stationary distribution on customer size 0 to {0}:\n{1}\n".format(self.max_record, stationary))

        #Wait time stats
        wait = (self.departure_times - self.arrival_times)[np.logical_not(self.balk)]
        mean_wait = np.mean(wait)
        sd_wait = np.std(wait)
        output.write("Mean total wait time: {0}\n".format(mean_wait))
        output.write("SD total wait time: {1}\n".format(mean_wait, sd_wait))

        #Customers who leave if line is too long
        output.write("Customers who balked: {0} of {1} \n".format(np.sum(self.balk), self.narrivals))
        output.close()


def main():
    fileInput = open(sys.argv[1], 'r')
    params = json.load(fileInput)
    fileInput.close()
    q1 = QueueSim(params)
    q1.sim()
    q1.write_stats(sys.argv[2])

if __name__ == '__main__':
    main()
