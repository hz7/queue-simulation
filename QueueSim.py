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

import heapq, json, numpy as np, sys

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
        self.arrival_times = np.hstack(arrival_times)
        self.narrivals = len(self.arrival_times)
        self.service_times = np.random.exponential(1 / self.service_rate, self.narrivals)

    def sim(self):
        """
        Simulate a M/M/s queuing process from pre-computed arrival and service times
        for each customer.
        """
        self.pregen()

        #variables for simulation statistics
        self.departure_times = np.zeros(self.narrivals) #when a customer finishes (either balking/finished service)
        self.t = np.zeros(2 * self.narrivals) #times of each arrival or departure event
        self.states = np.zeros(2 * self.narrivals) #number of customers in system at each event
        self.balk = np.zeros(self.narrivals, dtype=np.bool) #indicates if a customer leaves the line because of full capacity
        self.balkEventMask = np.zeros(2 * self.narrivals) #indicates balk events

        #simulation looks at the embedded markov chain over arrival/departure events
        #Variables for simulation state
        self.idx_next_arrival = 0 #index of next arrival
        self.service_pq = [] #priority queue of customers being served, top item is earliest event
        self.line_pq = [] #priority queue of customers in line, top item is earliest event
        self.event_no = 0 #current event number

        while self.event_no < 2 * self.narrivals:
            #Current event is the arrival or departure that comes first
            #arrival takes precedent if t_next_arrival == t_next_departure.
            #If no more events, end the simulation.
            if self.idx_next_arrival < self.narrivals and self.service_pq:
                t_next_arrival = self.arrival_times[self.idx_next_arrival]
                t_next_departure = self.service_pq[0].t
                if t_next_arrival <= t_next_departure:
                    event = Arrival(t_next_arrival, self.idx_next_arrival)
                else:
                    event = heapq.heappop(self.service_pq)
            elif self.idx_next_arrival < self.narrivals:
                event = Arrival(self.arrival_times[self.idx_next_arrival], self.idx_next_arrival)
            elif self.service_pq:
                event = heapq.heappop(self.service_pq)
            else:
                break

            #Allow the event to act. Skip rest of loop if action returns true.
            if event.act(self):
                continue

            #Finally, if possible, move a waiting customer from line into service
            spots_open = self.nservers - len(self.service_pq)
            if spots_open > 0 and self.line_pq:
                arrival = heapq.heappop(self.line_pq)
                t_departure = event.t + self.service_times[arrival.customer_idx]
                heapq.heappush(self.service_pq, Departure(t_departure, arrival.customer_idx))
                self.departure_times[arrival.customer_idx] = t_departure

            self.t[self.event_no] = event.t
            self.states[self.event_no] = len(self.service_pq) + len(self.line_pq)
            self.event_no += 1


    def write_stats(self, filename):
        """ Write statistics to path filename. """
        output = open(filename, 'w')
        output.write("Simulation results for params {0}\n\n".format(self.params))
        #Stats ignore customers who balk

        #Stationary distribution
        customer_times = self.t[np.logical_not(self.balkEventMask)]
        customer_states = self.states[np.logical_not(self.balkEventMask)]
        interval_lengths = np.diff(customer_times)
        cum_state = np.zeros(self.max_record) #keeps track of time in a particular state
        for i in range(self.max_record):
            cum_state[i] = np.sum(interval_lengths[customer_states[:-1] == i]) #last state is zero, exclude
        cum_state[0] += customer_times[0] #include the time up to first arrival for state 0
        stationary = cum_state / customer_times[-1]
        output.write("Stationary distribution on customer size 0 to {0}:\n{1}\n"
                                .format(self.max_record, stationary))

        #Wait time stats
        wait = (self.departure_times - self.arrival_times)[np.logical_not(self.balk)]
        mean_wait = np.mean(wait)
        sd_wait = np.std(wait)
        output.write("Mean total wait time: {0}\n".format(mean_wait))
        output.write("SD total wait time: {1}\n".format(mean_wait, sd_wait))

        #Customers who leave if line is too long
        output.write("Customers who balked: {0} of {1} \n".format(np.sum(self.balk), self.narrivals))
        output.close()

class Event:
    """ The events in a queue simulation. """
    def __init__(self, t, customer_idx):
        self.t = t #event time
        self.customer_idx = customer_idx #customer number associated with event

    def act(self, simulation):
        """ Event action on a simulation. """
        return False

    def __lt__(self, other):
        """ Events ordered by increasing time. """
        return self.t < other.t

class Arrival(Event):
    """ Arrival event. """
    def act(self, simulation):
        isBalk = len(simulation.line_pq) >= simulation.capacity
        if isBalk:
            simulation.balk[simulation.idx_next_arrival] = True
            simulation.departure_times[simulation.idx_next_arrival] = self.t
        else: #add arrival to line only if below capacity
            heapq.heappush(simulation.line_pq, self)
        simulation.idx_next_arrival += 1
        if isBalk:
            for j in range(2):
                simulation.balkEventMask[simulation.event_no] = True
                simulation.t[simulation.event_no] = self.t
                simulation.states[simulation.event_no] = simulation.capacity
                simulation.event_no += 1
            return True
        return False

class Departure(Event):
    """ Departure Event. """
    pass

def main():
    fileInput = open(sys.argv[1], 'r')
    params = json.load(fileInput)
    fileInput.close()
    q1 = QueueSim(params)
    q1.sim()
    q1.write_stats(sys.argv[2])

if __name__ == '__main__':
    main()
