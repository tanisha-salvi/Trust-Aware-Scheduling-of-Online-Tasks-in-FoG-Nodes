# import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import constants as constants
import random
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np


# ANSI color codes(color sholor)
class Color:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

# Define a Task class to represent tasks
class Task:
    def __init__(self, arrival_time, execution_time, user, reliability_requirement, expected_time=None, fog_node_scheduled=None):
        # Assign a unique ID to each task
        self.id = constants.num
        constants.num += 1
        # Initialize task attributes
        self.arrival_time = arrival_time
        self.release_time = 0  # Release time (initially set to 0)
        self.completion_time = 0  # Completion time (initially set to 0)
        self.execution_time = execution_time  # Execution time required for the task
        self.final_fog_node = None  # Final assigned fog node (initially set to None)
        self.user = user  # User ID associated with the task
        self.deadline = arrival_time + constants.K  # Deadline for task completion
        self.reliability_requirement = reliability_requirement  # Reliability requirement of the task
        self.num_of_failures = 0  # Number of failures encountered by the task
        self.expected_time = expected_time  # Expected time for task completion (optional)
        self.fog_node_scheduled = fog_node_scheduled  # Fog node where task is scheduled (optional)
        # List to track visited fog nodes (initially set to 0 for all nodes)
        self.fog_node_visited = [0] * constants.Num_of_nodes

# Define a FogNode class to represent Fog nodes
class FogNode:
    # Class variable to track the number of instances
    node_count = 0

    def __init__(self, failure_rate):
        # Increment the node count and assign it as the ID
        FogNode.node_count += 1
        self.id = FogNode.node_count
        # Initialize fog node attributes
        self.shared_trust = 0.5  # Shared trust value
        self.total_tasks = 2  # Total number of tasks assigned to the fog node
        self.total_successful_tasks = 1  # Total number of successfully completed tasks
        # Lists to store individual trust, tasks, and successful tasks for each user
        self.individual_trust = [0.5] * (constants.Num_of_users + 1)
        self.individual_tasks = [2] * (constants.Num_of_users + 1)
        self.individual_successful_tasks = [1] * (constants.Num_of_users + 1)
        self.failure_rate = failure_rate  # Failure rate of the fog node
        self.local_queue = []  # Local queue of tasks
        self.active = dummy_task  # Active task (initially set to a dummy task)
        self.busy_until = 0  # Time until the fog node is busy (initially set to 0)

    # Function to calculate the cost of execution on a node
    def calculate_cost(self):
        return constants.Base + constants.BR * self.shared_trust ** 2
    
    def get_score(self,user):
        return 0.6 * self.individual_trust[user] + 0.4 * self.shared_trust
    
    # Method to update individual trust value when a task is successful
    def update_individual_trust_on_success(self, user_index):
        # Increment total tasks
        self.total_tasks += 1
        self.total_successful_tasks += 1
        # Increment successful tasks for the corresponding user
        self.individual_tasks[user_index] += 1
        self.individual_successful_tasks[user_index] += 1
        # Update individual trust for the corresponding user
        self.individual_trust[user_index] = self.individual_successful_tasks[user_index] / self.individual_tasks[user_index]

        # Update shared trust based on the updated individual trust values
        self.shared_trust = self.total_successful_tasks / self.total_tasks
    
    def update_individual_trust_on_failure(self, user_index):
        # Increment total tasks
        self.total_tasks += 1
        # Increment successful tasks for the corresponding user
        self.individual_tasks[user_index] += 1
        # Update individual trust for the corresponding user
        self.individual_trust[user_index] = self.individual_successful_tasks[user_index] / self.individual_tasks[user_index]

        # Update shared trust based on the updated individual trust values
        self.shared_trust = self.total_successful_tasks / self.total_tasks

    def optimal_schedule(self, task):

        # Step 1: Insert the input task into the local queue
        self.local_queue.append(task)
        task.expected_time = task.execution_time * math.exp(self.failure_rate * task.execution_time) 
        # Step 2: Enumerate jobs in the local queue based on non-decreasing due dates
        sorted_jobs = sorted(self.local_queue, key=lambda x: x.deadline)
        
        # Step 3: Initialize variables
        S = []  # Set to store optimal schedule
        t = self.busy_until   # Current schedule time

        # Step 4-7: Construct the optimal set S
        for job in sorted_jobs:
            S.append(job)
            t += job.expected_time  # Update current time based on expected time
            if t > job.deadline:
                # Find job j in S with the largest execution time
                max_execution_job = max(S, key=lambda x: x.expected_time)
                S.remove(max_execution_job)
                t -= max_execution_job.expected_time
        unscheduled_tasks = [job for job in self.local_queue if job not in S]
       # Calculate the final time until which the machine is busy
        self.local_queue = S     
        return unscheduled_tasks

def initialize_trust_values(num_of_nodes):
    for i in range(num_of_nodes):
        failure_rate = random.uniform(0, 0.2)
        fog_node = FogNode(failure_rate)
        constants.fog_nodes.append(fog_node)
        
def change_failure_rate(node):
    node.failure_rate = random.uniform(0, 0.2)
        
def empty_local_queue(node):
    for task in node.local_queue:
        if task not in constants.global_queue:
            constants.global_queue.append(task)
        node.local_queue.remove(task)

def global_scheduler():
    if len(constants.global_queue) == 0:
        pass
    else:
        unscheduled_tasks = []
        # print("Starting global scheduling...")
        while len(constants.global_queue):
            if len(constants.global_queue) == 0:
                break
            constants.global_queue = sorted(constants.global_queue, key=lambda x: (x.deadline, x.execution_time,x.reliability_requirement))
            
            task = constants.global_queue[0]
            if task.deadline < constants.cnt + task.execution_time:           
                constants.dropped_tasks.append(task)
                constants.global_queue.remove(task)                
                continue
            
            constants.fog_nodes = sorted(constants.fog_nodes, key=lambda x: (x.get_score(task.user)))  
            for node in constants.fog_nodes:
                if node.get_score(task.user) >= task.reliability_requirement :
                    tasks_unscheduled = node.optimal_schedule(task)
                    for jobs in tasks_unscheduled:
                        if(jobs not in constants.global_queue):
                            constants.global_queue.append(jobs)
                            
                    if task not in tasks_unscheduled:
                        constants.global_queue.remove(task)
                        break
                    
            if task in constants.global_queue:
                constants.global_queue.remove(task)
                
                if(task not in unscheduled_tasks):
                    unscheduled_tasks.append(task)
             
        constants.global_queue = unscheduled_tasks
        
        constants.global_queue = sorted(constants.global_queue, key=lambda x: (x.deadline, x.execution_time))
   
        for task in constants.global_queue:
            max_probability = 0
            node_idx = -1
            for node in constants.fog_nodes:
                if node.get_score(task.user) < task.reliability_requirement:
                    continue
                time_of_completion = constants.cnt
                for local_task in node.local_queue:
                    time_of_completion += (local_task.expected_time)
                if time_of_completion + task.execution_time <= task.deadline:
                    No_of_trials = (task.deadline - (time_of_completion)) / task.execution_time
                    probability_of_success = 1 - (1 - math.exp(-node.failure_rate * task.execution_time)) ** No_of_trials
                    if probability_of_success > max_probability:
                        max_probability = probability_of_success
                        node_idx = constants.fog_nodes.index(node)
            if node_idx != -1:
                task.expected_time = task.execution_time * math.exp(constants.fog_nodes[node_idx].failure_rate * task.execution_time)
                constants.fog_nodes[node_idx].local_queue.append(task)
                constants.global_queue.remove(task)

        for task in constants.global_queue:
            max_latency = 0
            node_idx = -1
            for node in constants.fog_nodes:
                if node.get_score(task.user) < task.reliability_requirement:
                    continue
                time_of_completion = constants.cnt
                for local_task in node.local_queue:
                    time_of_completion += (local_task.execution_time)
                if time_of_completion + task.execution_time <= task.deadline:
                    latency = (task.deadline - time_of_completion - task.execution_time)
                    if latency > max_latency:
                        max_latency = latency
                        node_idx = constants.fog_nodes.index(node)
            if node_idx != -1:
                task.expected_time = task.execution_time * math.exp(constants.fog_nodes[node_idx].failure_rate * task.execution_time)
                constants.fog_nodes[node_idx].local_queue.append(task)
                constants.global_queue.remove(task)

def local_completion():
    for node in constants.fog_nodes:
        if node.active.user != -1:
            task = node.active
            # Check if execution is complete
            if task.release_time + task.execution_time <= constants.cnt:
                # Flip a coin with success probability
                success_probability = math.exp(-node.failure_rate * task.execution_time)
                # completed_tasks
                constants.failure_count += 1
                constants.Cost += (constants.Base + constants.BR * (node.shared_trust ** 2))*task.execution_time
                constants.Cost /= (constants.Base + constants.BR)*constants.Execution_time
                # constants.Cost += constants.Base + constants.BR * (node.shared_trust ** 2)
                # constants.Cost /= (constants.Base + constants.BR)
                constants.total_time += task.execution_time / success_probability
                if random.random() < success_probability:
                    node.update_individual_trust_on_success(task.user)
                    # Task successful, remove from local queue and add to completed tasks
                    constants.completed_tasks.append(task)
                    # Clear active task flag
                    node.active = dummy_task
                    task.completion_time = task.release_time + task.execution_time
                    task.final_fog_node = node.id
                    constants.total_successful_tasks += 1
                    if(node.total_successful_tasks%constants.tasks_window == 0):
                        change_failure_rate(node)
                        empty_local_queue(node)
                else:
                    node.update_individual_trust_on_failure(task.user)
                    if constants.cnt + task.execution_time > task.deadline:
                        constants.dropped_tasks.append(task)
                        node.active = dummy_task
                        node.busy_until = constants.cnt
                        break
                    # failure Logic
                    task.expected_time = max(task.expected_time-task.execution_time,task.execution_time)
                    task.release_time = constants.cnt
                    node.busy_until = constants.cnt + task.execution_time * math.exp(node.failure_rate * task.execution_time)
    
                
def local_scheduler():
  
    for node in constants.fog_nodes:
        if node.active.user == -1 and len(node.local_queue):
            task = node.local_queue[0]
            if constants.cnt + task.execution_time > task.deadline:
                constants.dropped_tasks.append(task)
                node.local_queue.remove(task)
                continue
            
            constants.active+=1
            task.expected_time = task.execution_time * math.exp(node.failure_rate * task.execution_time)
            task.release_time = constants.cnt
            node.busy_until = constants.cnt + task.expected_time
            node.local_queue.remove(task)
            node.active = task
            
def generate_random_jobs(cnt):
    num_jobs = random.randint(1, constants.Num_of_jobs_per_sec)
    constants.total_tasks_generated += num_jobs
    for _ in range(num_jobs):
        execution_time = random.uniform(1, constants.Execution_time)
        user = random.randint(1, constants.Num_of_users)
        reliability_requirement = random.uniform(0, 0.5)
        curr_job = Task(cnt, execution_time, user, reliability_requirement)
        if(curr_job not in constants.global_queue):
            constants.global_queue.append(curr_job)
            constants.jobs_data.append([constants.cnt, curr_job.execution_time, curr_job.user, curr_job.reliability_requirement])    

def print_jobs_in_queue(x):
    headers = ["Task Number", "Arrival Time", "Execution Time", "User", "Reliability Requirement"]
    table_data = []
    for job in x:
        table_data.append([job.id, job.arrival_time, job.execution_time, job.user, job.reliability_requirement])
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    
def print_jobs_after_completion(x):
    headers = ["Task Number", "Arrival Time", "Execution Time", "Completion Time", "Fog Node", "Reliability Requirement"]
    table_data = []
    sorted_x = sorted(x, key=lambda job: job.final_fog_node)
    for job in sorted_x:
        table_data.append([job.id, job.arrival_time, job.execution_time, job.completion_time, job.final_fog_node, job.reliability_requirement])

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    headers = ["Final Fog Node", "Count", "Shared Trust"]
    fog_node_counts = {}

    # Count the occurrences of each final fog node
    for row in table_data:
        fog_node = row[4]  # Assuming the "Fog Node" column is at index 4
        if fog_node in fog_node_counts:
            fog_node_counts[fog_node] += 1
        else:
            fog_node_counts[fog_node] = 1

    # Prepare table data for count table
    count_table_data = [[fog_node, count] for fog_node, count in fog_node_counts.items()]

    # Sort count table data based on final fog node
    count_table_data.sort(key=lambda x: x[0])

    # Print count table
    print(tabulate(count_table_data, headers=headers, tablefmt="pretty"))

def print_final_shared_trust():
    headers = ["Fog Node", "Shared Trust", "Failure Rate"]
    table_data = []
    # Assuming constants.fog_nodes is your list of fog nodes
    sorted_fog_nodes = sorted(constants.fog_nodes, key=lambda node: node.id)

    for node in sorted_fog_nodes:
        table_data.append([node.id, node.shared_trust, node.failure_rate])

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

def print_local_queues():
    for i, node in enumerate(constants.fog_nodes, start=1):
        print(f"Local Queue of Fog Node {i}:")
        headers = ["Task Number", "Arrival Time", "Execution Time", "User", "Reliability Requirement"]
        table_data = []
        for job in node.local_queue:
            table_data.append([job.id, job.arrival_time, job.execution_time, job.user, job.reliability_requirement])
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        print()

def print_local_queue(node):
    headers = ["Task Number", "Arrival Time", "Execution Time", "User", "Reliability Requirement"]
    table_data = []
    for job in node.local_queue:
        table_data.append([job.id, job.arrival_time, job.execution_time, job.user, job.reliability_requirement])
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print()
    
def plot_graph(timestamps, successful_jobs):
    plt.plot(timestamps, successful_jobs, marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Successful Jobs')
    plt.title('Number of Jobs Successfully Completed vs Time')
    plt.grid(True)
    plt.show()
    
def plot_success_ratio_vs_num_of_nodes(num_of_nodes, total_tasks, total_successful):
    success_ratio = [total_successful[i] / total_tasks[i] for i in range(len(num_of_nodes))]
    plt.plot(num_of_nodes, success_ratio, marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Success Ratio')
    plt.title('Success Ratio vs Number of Nodes')
    plt.grid(True)
    plt.show()


def plot1(arr, costs):
    fig, ax = plt.subplots()
    colors = ['#ADD8E6', '#FFA07A']
    labels = ["Sorting tasks", "Without sorting"]
    
    for i in range(len(arr)):
        ax.bar(arr[i], costs[i], color=colors[i], label=labels[i])
    for i in range(len(arr)):
        ax.bar(arr[i], costs[i], color=colors[i], label=labels[i])
        ax.text(arr[i], costs[i], f'{costs[i]:.2f}', ha='center', va='bottom')
        
    plt.xlabel('Approach')
    plt.ylabel('Cost')
    plt.ylim(0, 1.0)
    plt.title('')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def plot2(arr, total_tasks, total_successful):
    fig, ax = plt.subplots()
    success_ratio = [total_successful[i] / total_tasks[i] for i in range(len(arr))]
    plt.title('')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def plot2(arr, total_tasks, total_successful):
    fig, ax = plt.subplots()
    success_ratio = [total_successful[i] / total_tasks[i] for i in range(len(arr))]
    colors = ['#ADD8E6', '#FFA07A']
    labels = ["Sorting tasks", "Without sorting"]
    
    for i in range(len(arr)):
        ax.bar(arr[i], success_ratio[i], color=colors[i], label=labels[i])
        ax.text(arr[i], success_ratio[i], f'{success_ratio[i]:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Approach')
    plt.ylabel('Success Ratio')
    plt.ylim(0, 1.0)
    plt.title('')
    plt.legend()
    plt.grid(axis='y')
    plt.tick_params(axis='x', which='both', bottom=False)  # Hide x-axis tick marks
    plt.show()

    
def plot_task_bar(num_of_nodes, total_tasks, total_successful):
    bar_width = 0.35
    index = np.arange(len(num_of_nodes))

    plt.bar(index, total_tasks, bar_width, label='Total Tasks', color='#ADD8E6')  
    plt.bar(index + bar_width, total_successful, bar_width, label='Successful Tasks', color='#FFA07A')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Count')
    plt.title('Bar Plot of Total and Successful Tasks vs Number of Nodes')
    plt.xticks(index + bar_width / 2, num_of_nodes)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_success_ratio_vs_execution_time(execution_time, total_tasks, total_successful):
    success_ratio = [total_successful[i] / total_tasks[i] for i in range(len(execution_time))]
    plt.plot(execution_time, success_ratio, marker='o')
    plt.xlabel('Relative Deadline K')
    plt.ylabel('Success Ratio')
    plt.title('Success Ratio vs Relative Deadline K')
    plt.grid(True)
    plt.show()
    

def plot_cost_vs_K(k_array, costs):
    # success_ratio = [total_successful[i] / total_tasks[i] for i in range(len(execution_time))]
    plt.plot(k_array, costs, marker='o')
    plt.xlabel('Number of Fog Nodes')
    plt.ylabel('Cost')
    plt.title('Cost vs Number of Fog Nodes')
    plt.grid(True)
    plt.show()
    
def plot_task_bar_execution_time(execution_time, total_tasks, total_successful):
    bar_width = 0.35
    index = np.arange(len(execution_time))

    plt.bar(index, total_tasks, bar_width, label='Total Tasks', color='#ADD8E6')  
    plt.bar(index + bar_width, total_successful, bar_width, label='Successful Tasks', color='#FFA07A')

    plt.xlabel('Relative Deadline K')
    plt.ylabel('Count')
    plt.title('Bar Plot of Total and Successful Tasks vs Relative Deadline K')
    plt.xticks(index + bar_width / 2, execution_time)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate random jobs
def generate_jobs():
    user = random.randint(1, constants.Num_of_users)

    while constants.cnt <= 600:
        local_completion()
        generate_random_jobs(constants.cnt)
        global_scheduler() 
        local_scheduler()
        timestamps.append(constants.cnt)
        successful_jobs.append(constants.total_successful_tasks)
        constants.cnt += 1
    while constants.cnt <= 800:
        local_completion()
        global_scheduler()    
        local_scheduler()
        timestamps.append(constants.cnt)
        successful_jobs.append(constants.total_successful_tasks)
        constants.cnt += 1
    headers = ["Current Time", "Execution Time", "User", "Reliability Requirement"]
    table = tabulate(constants.jobs_data, headers=headers, tablefmt="grid")

    # Write to file
    # open n append mode
    # with open("output.txt", "a") as file:
    #   file.write(table)
    #  file.write("\n\n")
    
        
if __name__ == '__main__':
    timestamps = []
    successful_jobs = []
    num_of_nodes = [2, 5, 10, 15, 20, 50, 100]
    total_tasks = [0, 0, 0, 0, 0, 0]
    total_successful = [0, 0, 0, 0, 0, 0]
    execution_time_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    jobs_per_sec = [2, 4, 6, 8, 10]
    k_array = [15, 20, 30, 50, 100]
    
    arr = [0, 1]
    total_tasks = [0] * len(arr)
    total_successful = [0] * len(arr)
    costs = [0] * len(arr)
    
    for i in range(0,1):
        constants.K = 20
        constants.Num_of_nodes = 10
        constants.fog_nodes = []
        constants.global_queue = []
        constants.cnt = 0
        constants.dropped_tasks = []
        constants.completed_tasks = []
        constants.num = 1
        constants.Cost = 0  
        constants.total_tasks_generated = 0
        constants.total_successful_tasks = 0
        constants.active = 0
        constants.time_window = 10
        constants.tasks_window = 5
        constants.total_time = 0
        constants.Execution_time = 10
        constants.Num_of_jobs_per_sec = 6

        execution_time = random.uniform(1, constants.Execution_time)
        user = random.randint(1, constants.Num_of_users)
        reliability_requirement = random.uniform(0, 0.5)
        dummy_task = Task(-1, execution_time, -1, reliability_requirement)

        initialize_trust_values(constants.Num_of_nodes)    

        generate_jobs()
        print("Total tasks generated: ", constants.total_tasks_generated)
        print("Total successful tasks: ", constants.total_successful_tasks)
        print()
        print("--------------------DROPPED Tasks----------------------------")
        print(len(constants.dropped_tasks))
        print()
        print_jobs_in_queue(constants.dropped_tasks)
        print()
        print("---------------------COMPLETED tasks-------------------------")
        print()
        print(len(constants.completed_tasks))
        print_jobs_in_queue(constants.completed_tasks)
        print(constants.active)
        print_jobs_after_completion(constants.completed_tasks)
        
        print("---------------------Final Shared Trust-------------------------")
        print_final_shared_trust()
        
        print("---------------------Cost per Job-------------------------")
        print((constants.Cost + len(constants.dropped_tasks))/constants.total_tasks_generated)
        
        costs[i] = (constants.Cost + len(constants.dropped_tasks))/constants.total_tasks_generated      
        
        print("Total time taken: ", constants.total_time)
        
        total_tasks[i] = constants.total_tasks_generated
        total_successful[i] = constants.total_successful_tasks