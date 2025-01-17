#!/usr/bin/env python3
import sys, collections

# Used for drawing trellis
import networkx as nx
import random
import matplotlib.pyplot as plt
import operator


class HMM_Parser():
    def parse(self, hmm_filename):
        ## Variables for parsing results
        # Number of parsed initial lines
        parsed_init_num = 0

        # Number of parsed transition lines
        parsed_trans_num = 0

        # Number of parsed emission lines
        parsed_emiss_num = 0

        # Initial state probabilities: Dictionary of state -> probability
        initial_state_probs = collections.defaultdict(float)

        # Transition probabilities: Dictionary of start state -> end state -> probability
        transition_probs = collections.defaultdict(lambda: collections.defaultdict(float))

        # Set of states in HMM
        states = set()

        # Emission probabilities: Dictionary of state -> emission -> probability
        emission_probs = collections.defaultdict(lambda: collections.defaultdict(float))

        # Set of emitted items in HMM
        emissions = set()

        # Dictionary of emissions -> set of states that emitted that item
        emission_to_states = collections.defaultdict(set)

        # Open HMM file
        with open(hmm_filename, 'r') as infile:
            parse_state = None

            # Iterate through HMM file
            for line in infile:
                # Strip any trailing whitespace from line (including newlines)
                line = line.rstrip()

                # If we encounter an empty line, skip it
                if line == "":
                    continue

                # If we encounter line starting with "%init_states", the following lines will be about initial states
                elif "%initial_states" == line:
                    parse_state = "init"

                # If we encounter line starting with "%transitions", the following lines will be about transitions
                elif "%transitions" == line:
                    parse_state = "transition"

                # If we encounter line starting with "%emissions", the following lines will be about emissions
                elif "%emissions" == line:
                    parse_state = "emission"

                # If we are parsing initial states, extract state and probability of being in state
                elif parse_state == "init":
                    # Extract state and probability
                    (state, probability) = self.parse_init_line(line)

                    # Update dict of state->prob
                    initial_state_probs[state] = probability

                    # Update set of encountered states
                    states.add(state)

                    # Update count of parsed init lines
                    parsed_init_num += 1


                # If we are parsing transitions, extract transition probability
                elif parse_state == "transition":
                    # Extract transition info
                    (start_state, end_state, probability) = self.parse_transition_line(line)

                    # Check this isn't a duplicate transition
                    if start_state in transition_probs and end_state in transition_probs[start_state]:
                        print("warning: duplicated transition {} {}".format(start_state, end_state))
                        sys.exit()
                    else:
                        transition_probs[start_state][end_state] = probability

                    # Update set of encountered states
                    states |= {start_state, end_state}

                    # Update count of parsed transition lines
                    parsed_trans_num += 1


                # If we are parsing emissions, extract emission probability
                elif parse_state == "emission":
                    # Extract emission info
                    (state, emission, probability) = self.parse_emission_line(line)

                    # Check this isn't a duplicate emission
                    if state in emission_probs and emission in emission_probs[state]:
                        print("warning: duplicated emission {} {}".format(state, emission))
                        sys.exit()
                    else:
                        emission_probs[state][emission] = probability

                    # Update set of encountered states
                    states |= {state}

                    # Update set of emitted items
                    emissions |= {emission}

                    # Update mapping of emitted items to states
                    emission_to_states[emission] |= {state}

                    # Update count of parsed emission lines
                    parsed_emiss_num += 1

                # We should never hit this condition if our HMM is valid
                else:
                    print("Unexpected item encountered in parsing:{}".format(line))
                    sys.exit()

        # Put all of the HMM information in a single dictionary
        return {
            "parsed_init_num": parsed_init_num,
            "parsed_trans_num": parsed_trans_num,
            "parsed_emiss_num": parsed_emiss_num,
            "initial_state_probs": initial_state_probs,
            "initial_states": initial_state_probs.keys(),
            "transition_probs": transition_probs,
            "states": states,
            "emission_probs": emission_probs,
            "emissions": emissions,
            "emission_to_states": emission_to_states,
        }

    ## Extract the last integer in a space separated string
    # e.g. "State 2" -> 2
    def _get_last_int_in_line(self, line):
        try:
            return int(line.split()[-1])
        except:
            print("Error: failed to parse line:{}".format(line))
            sys.exit()
            return 0

    # Extract information about initial states from a line
    # "BOS 1.0" -> ("BOS", 1.0)
    def parse_init_line(self, line):
        # Split line on spaces
        split_line = line.split()

        # Extract state and prob
        state = split_line[0]
        prob = float(split_line[1])
        return (state, prob)

    # Extract information about transitions from a line
    # "BOS DT 0.5" -> ("BOS", "DT", 0.5)
    def parse_transition_line(self, line):
        # Split line on spaces
        split_line = line.split()

        # Extract states and prob
        start_state = split_line[0]
        end_state = split_line[1]
        prob = float(split_line[2])

        # Verify probability
        if prob < 0.0 or prob > 1.0:
            print("Error: a prob is not in [0,1] range:{}".format(prob))
            sys.exit()
        return (start_state, end_state, prob)

    # Extract information about emissions from a line
    # "DT the 0.5" -> ("DT", "the", 0.5)
    def parse_emission_line(self, line):
        # Split line on spaces
        split_line = line.split()

        # Extract states and prob
        state = split_line[0]
        emission = split_line[1]
        prob = float(split_line[2])

        # Verify probability
        if prob < 0.0 or prob > 1.0:
            print("Error: a prob is not in [0,1] range:{}".format(prob))
            sys.exit()
        return (state, emission, prob)


class HMMTrellis():
    def __init__(self, hmm_instance, input_arr):
        self.hmm = hmm_instance
        self.input_arr = input_arr
        self.plt = plt
        self.nx = nx

        self.trellis = self.nx.Graph()
        self.node_positions = dict()
        self.labels = dict()
        self.node_map = dict()


        # Clear graph
        plt.clf()
        plt.cla()
        plt.close()

        # Create all the basic nodes in our trellis and position them in a grid
        idx = 1
        for row_idx in range(len(self.hmm.states)):
            for col_idx in range(len(input_arr) + 1):
                # Add the node to our trellis
                self.trellis.add_node(idx)

                # Set the (x,y) position of this node
                self.node_positions[idx] = (col_idx + 1, row_idx)

                # Set the label on this node to empty string
                self.labels[idx] = ""

                # Remember the id of this (row_idx, col_idx) node
                self.node_map[(row_idx, col_idx)] = idx
                idx += 1

    # Set the text on a given node
    def set_node_label(self, row_idx, col_idx, label):
        node_id = self.node_map[(row_idx, col_idx)]
        self.labels[node_id] = label

    # Add an edge between trellis nodes
    def add_edge(self, start_row_idx, start_col_idx, end_row_idx, end_col_idx, color='black'):
        # Get id of nodes
        start_node_id = self.node_map[(start_row_idx, start_col_idx)]
        end_node_id = self.node_map[(end_row_idx, end_col_idx)]

        # Create edge
        edge = (start_node_id, end_node_id)
        self.trellis.add_edge(*edge, color=color)

    # Change the color of an existing edge
    def change_edge_color(self, start_row_idx, start_col_idx, end_row_idx, end_col_idx, color):
        # Get id of nodes
        start_node_id = self.node_map[(start_row_idx, start_col_idx)]
        end_node_id = self.node_map[(end_row_idx, end_col_idx)]

        # Get edge
        edge = (start_node_id, end_node_id)

        # Find the edge in the trellis, and update color
        if edge[0] in self.trellis and edge[1] in self.trellis[edge[0]]:
            self.trellis[edge[0]][edge[1]]['color'] = color
        if edge[1] in self.trellis and edge[0] in self.trellis[edge[1]]:
            self.trellis[edge[1]][edge[0]]['color'] = color

    # Used by display() and save_to_file() to generate the trellis
    def _generate_graph(self, title):
        # Draw the main trellis
        edge_color_list = [self.trellis[e[0]][e[1]]['color'] for e in self.trellis.edges()]
        self.nx.draw(self.trellis, self.node_positions, labels=self.labels, edge_labels=True, edge_color=edge_color_list,
                node_size=800, font_size=8)

        # Set up the row and column labels
        trellis_labels = ["States", "Initial States"] + self.input_arr
        for row_idx in range(len(trellis_labels)):
            self.plt.text(row_idx, len(self.hmm.states) - 0.5, trellis_labels[row_idx], horizontalalignment='center',
                     verticalalignment='center')
        for col_idx in range(len(self.hmm.states)):
            self.plt.text(0, col_idx, self.hmm.idx_to_state[col_idx], horizontalalignment='center',
                     verticalalignment='center')

        # Set the title
        x_pos = (len(self.input_arr) + 1) / float(2)
        y_pos = len(self.hmm.states)
        self.plt.text(x_pos, y_pos, title, horizontalalignment='center', verticalalignment='center')

        # Set the window range that the user sees
        self.plt.xlim(left=-0.3, right=len(trellis_labels) - 1 + 0.6)
        self.plt.ylim(top=len(self.hmm.states) + 0.5)

    # Display the trellis in a popup window
    def display(self, title):
        # Generate the trellis graph
        self._generate_graph(title)

        # Display the trellis
        self.plt.show()

    # Save the trellis image to a file
    def save_to_file(self, filename, title):
        # Generate the trellis graph
        self._generate_graph(title)

        # Save the trellis graph to file
        self.plt.savefig(filename)


# HMM Class
class HMM():
    def __init__(self, hmm_filename):
        hmm_data = self.__load_hmm(hmm_filename)
        self.emissions = hmm_data['emissions']
        self.states = hmm_data['states']
        self.initial_state_probabilities = hmm_data['initial_state_probs']
        self.initial_states = hmm_data['initial_states']
        self.state_to_idx, self.idx_to_state = self._create_state_to_idx_mapping(self.states, self.initial_states)
        self.emission_to_states = hmm_data['emission_to_states']
        self.emission_probabilities = hmm_data['emission_probs']
        self.transition_probabilities = hmm_data['transition_probs']

    # Load a hmm model file
    def __load_hmm(self, hmm_filename):
        return HMM_Parser().parse(hmm_filename)

    # Map each state to an index (Used by Viterbi)
    # i.e. first state -> 0, second state -> 1, etc...
    # We also map index to state:
    # i.e. 0 -> first state, 1 -> second state, etc...
    # These indexes will follow an ordering such that initial states come last
    #  This is useful when generating the trellis
    def _create_state_to_idx_mapping(self, states, initial_states):
        state_to_idx = collections.defaultdict(int)
        idx_to_state = collections.defaultdict(str)

        nonstart_states = list(states.difference(initial_states))
        nonstart_states.sort(reverse=True)

        initial_states = list(initial_states)
        initial_states.sort(reverse=True)

        all_states = nonstart_states + initial_states
        for state in all_states:
            idx = len(state_to_idx)
            state_to_idx[state] = idx
            idx_to_state[idx] = state
        return state_to_idx, idx_to_state


    # Run forward on a given input string

    def forward(self, input_str):
        # Remove any blank characters at the end of the string
        input_str = input_str.rstrip()

        # Split the input on spaces
        input_seq = input_str.split()


        result = [{}]
        # initialize base case
        # Initial state doesn't care about emission probabilities - we are not emitting anything at initial state
        for y in self.states:
            result[0][y] = self.initial_state_probabilities[y]

        # forward algorithm calculations
        # We want to iterate from 1 to len(input_seq)+1
        # idx of 1 is where we start emitting
        # len(input_seq)+1 b/c we want to cover the whole input_seq, and result is an array with len(input_seq)+1 entries
        for t in range(1, len(input_seq) + 1):
            result.append({})
            print("Processing emission: {}".format(input_seq[t-1]))
            for y in self.states:
                sum = 0.0
                for y0 in self.states:
                    # We want input_seq[t-1] because our for loop makes t start at 1, not 0
                    sum += (result[t-1][y0] * self.transition_probabilities[y0][y] * self.emission_probabilities[y][input_seq[t-1]])
                result[t][y] = sum
                print("\tProbability of ending up in {} state for emission sequence up to {}: {}".format(y,input_seq[t-1],result[t][y]))
        prob = 0.0
        for s in self.states:
            prob += result[-1][s]


        '''
        result_word = ""
        for idx, x in enumerate(result):
            if idx-1 == -1:
                print("input: initial state, forward: {}".format(x))
            else:
                result_word = result_word + max(x.items(), key=operator.itemgetter(1))[0]

        print("input:{}".format(input_seq))
        print("result: {}".format(result_word))
        #print("probability:{}".format(prob))
        '''
        # Set up the trellis
        trellis2 = HMMTrellis(self, input_seq)
        for input in range(len(result)):
            for state in result[input]:
                # Set the node's label as the probability of arriving at that node
                if result[input][state] == 0:
                    label = "0"
                elif result[input][state] == 1:
                    label = "1"
                else:
                    label = '{:.1e}'.format(result[input][state])

                trellis2.set_node_label(self.state_to_idx[state], input, label)

        # Add all valid HMM edges (Except for edges from the initial state)
        for input in range(len(result)):
            for state in result[input]:
                # If we are looking at a node in last column,
                #  we ignore, since edges are in the right-ward direction from current node
                # (i.e. there are no rightward edges starting from this node)
                if input + 1 == len(result):
                    continue

                # Iterate over all nodes of the next column
                for state2 in result[input + 1]:
                    # Check if this is a valid state transition
                    start_state = state
                    end_state = state2

                    # Skip edge creation if this is a non-valid start state
                    if input == 0 and start_state not in self.initial_states:
                        continue

                    # If this a valid transition (probability > 0), we draw an edge
                    if self.transition_probabilities[start_state][end_state] > 0:
                        trellis2.add_edge(self.state_to_idx[state], input, self.state_to_idx[state2], input + 1)


        # Display the trellis
        #trellis.display("Forward Trellis")
        trellis2.save_to_file("Forward.png", "Forward Trellis")

    # Run viterbi on a given input string
    def viterbi(self, input_str):
        def get_init_states_with_transition(curr_state):
            # States to return
            return_states = set()

            init_states = self.initial_states

            # Only return init states which have a transition to the current state
            for init_state in init_states:
                if init_state in self.transition_probabilities and curr_state in self.transition_probabilities[
                    init_state]:
                    return_states |= {init_state}

            return return_states

        def get_possible_states_with_transition(prev_observation, curr_state):
            # States to return
            return_states = set()

            possible_states = self.emission_to_states[prev_observation]

            # Only return states which have a transition to the current state
            for state in possible_states:
                if state in self.transition_probabilities and curr_state in self.transition_probabilities[state]:
                    return_states |= {state}

            return return_states

        # Remove any blank characters at the end of the string
        input_str = input_str.rstrip()

        # Split the input on spaces
        input_seq = input_str.split()

        # Check that every item in input_seq is a valid emission in HMM
        for observation in input_seq:
            # If we encounter item not in HMM, we quit
            if observation not in self.emissions:
                print("Error: Encountered observation ({}) in input that is not in HMM".format(observation))
                sys.exit()

        ## Set up our trellis and backpointers
        # Our trellis is a (len(input_seq)+1) row, len(self.states) col
        delta = [[0.0 for _ in range(len(input_seq) + 1)] for _ in range(len(self.states))]
        back_p = [[-1 for _ in range(len(input_seq) + 1)] for _ in range(len(self.states))]

        # Initialize our trellis with values for first column
        for state in self.states:
            # Get the table index corresponding to this state
            state_idx = self.state_to_idx[state]

            # Update the first column with probability for starting at that state
            if state in self.initial_states:
                delta[state_idx][0] = self.initial_state_probabilities[state]
            else:
                delta[state_idx][0] = 0

            # The backpointer of the first column should point to nothing
            back_p[state_idx][0] = -1

        # Iterate through our sequence for every obvservation in our input sequence
        for observation_idx, observation in enumerate(input_seq):
            print("Processing emission: {}".format(observation))
            # This is the trellis column index of the obvservation we are looking at
            observation_table_idx = observation_idx + 1

            # Get a set of states that can emit for this observation
            valid_states = self.emission_to_states[observation]
            print("    Valid states for this emission:"+str(valid_states))

            # Check each one to calculate probabilities of that state
            for valid_state in valid_states:
                # Get the trellis row index of this state
                curr_state_idx = self.state_to_idx[valid_state]

                # Get emission probability for this state
                emission_prob = self.emission_probabilities[valid_state][observation]

                # We will get set of valid states for previous observation
                valid_prev_observation_states = set()

                # Get the trellis column index of the previous observation
                prev_observation_idx = observation_idx - 1

                # If our current observation is the first of the sequence...
                if prev_observation_idx < 0:
                    # ...then our previous states are those that are in the initial state set, AND
                    # which have a transition to the current state
                    valid_prev_observation_states = get_init_states_with_transition(valid_state)

                # If there has been a previous observation in the sequence...
                else:
                    # ...then our previous states are those emit the previous observation, AND
                    # which have a transition to the current state
                    prev_observation = input_seq[prev_observation_idx]
                    valid_prev_observation_states = get_possible_states_with_transition(prev_observation, valid_state)

                # Now we find the most optimal transition from a previous state to the current state
                max_prob = 0
                max_state_idx = -1
                # Calculate the total probability from a previous state to the current state
                for valid_prev_observation_state in valid_prev_observation_states:
                    # Get the trellis row index of the previous state we are looking at
                    prev_state_idx = self.state_to_idx[valid_prev_observation_state]

                    # Get the probability of that previous state from the trellis
                    state_prob = delta[prev_state_idx][observation_table_idx - 1]

                    # Get the transition proability from that previous state to the current state
                    transition_prob = self.transition_probabilities[valid_prev_observation_state][valid_state]

                    # The total probability is the product of those two probabilities
                    total_prob = state_prob * transition_prob

                    # Update best probability
                    if max_prob < total_prob:
                        max_prob = total_prob
                        max_state_idx = prev_state_idx

                # If we found a valid transition to the current state, calculate the overall probability for that
                if max_prob > 0:
                    current_prob = max_prob * emission_prob
                    best_prev_state = max_state_idx
                    print("    Probability of ending up in {} state for emission sequence up to '{}': {:.5f}".format(self.idx_to_state[best_prev_state], observation, current_prob))

                # If we did not find a valid transition to current state, probability of state is 0
                else:
                    current_prob = 0
                    best_prev_state = -1
                    print("    Impossible to end up in any state for emission sequence up to {}".format(observation))


                # Update current cell
                delta[curr_state_idx][observation_table_idx] = current_prob

                # Update backpointer to best previous state
                back_p[curr_state_idx][observation_table_idx] = best_prev_state

        # Find the row that has the highest value in the final column of trellis
        # This is our best state sequence
        highest_prob = 0
        highest_prob_idx = -1
        for state in self.states:
            state_idx = self.state_to_idx[state]
            delta_val = delta[state_idx][len(input_seq)]
            if delta_val > highest_prob:
                highest_prob = delta_val
                highest_prob_idx = state_idx

        # If we couldn't find a valid state sequence
        if highest_prob_idx == -1:
            # We output that no sequence was found
            output = input_str + " => " + " *none*" + "\nProbability of sequence: " '0' + '\n'
            print(output)

        # If we did find a valid state sequence
        else:
            # We will iterate backwards through the trellis to find the best state idx sequence
            state_seq = [highest_prob_idx]
            for state_idx in range(len(input_seq), 0, -1):
                bp = int(back_p[highest_prob_idx][state_idx])
                state_seq.append(bp)
                highest_prob_idx = bp

            # Since our state sequence is backwards, we reverse it
            state_seq = list(reversed(state_seq))

            # We convert our state (index) sequence to a sequence of states
            state_seq_outputs = [self.idx_to_state[state_idx] for state_idx in state_seq]

            # Pretty print the results
            # Get the direct string representation of results
            output = input_str + " => " + " ".join(state_seq_outputs) + "\nProbability of sequence:" + str(
                highest_prob) + '\n'

            # Print all results
            print(output)

        # Set up the trellis
        trellis = HMMTrellis(self, input_seq)
        for row_idx in range(len(delta)):
            for col_idx in range(len(delta[0])):
                # Set the node's label as the probability of arriving at that node
                #  (this is based on Viterbi calculation)
                if delta[row_idx][col_idx] == 0:
                    label = "0"
                elif delta[row_idx][col_idx] == 1:
                    label = "1"
                else:
                    label = '{:.1e}'.format(delta[row_idx][col_idx])
                trellis.set_node_label(row_idx, col_idx, label)

        # Add all valid HMM edges (Except for edges from the initial state)
        for row_idx in range(len(delta)):
            for col_idx in range(len(delta[0])):
                # If we are looking at a node in last column,
                #  we ignore, since edges are in the right-ward direction from current node
                # (i.e. there are no rightward edges starting from this node)
                if col_idx + 1 == len(delta[0]):
                    continue

                # Iterate over all nodes of the next column
                for next_row_idx in range(len(delta)):
                    # Check if this is a valid state transition
                    start_state = self.idx_to_state[row_idx]
                    end_state = self.idx_to_state[next_row_idx]

                    # Skip edge creation if this is a non-valid start state
                    if col_idx == 0 and start_state not in self.initial_states:
                        continue

                    # If this a valid transition (probability > 0), we draw an edge
                    if self.transition_probabilities[start_state][end_state] > 0:
                        trellis.add_edge(row_idx, col_idx, next_row_idx, col_idx + 1)

        # Highlight Viterbi edges
        for row_idx in range(len(delta)):
            for col_idx in range(len(delta[0])):
                # See if this node has a backpointer
                if back_p[row_idx][col_idx] != -1:
                    # If it does, show the backpointer edge in red
                    back_pointer_row = back_p[row_idx][col_idx]
                    trellis.change_edge_color(row_idx, col_idx, back_pointer_row, col_idx - 1, 'red')

        # Display the trellis
        # trellis.display("Viterbi Trellis")
        trellis.save_to_file("viterbi.png", "Viterbi Trellis")


if __name__ == "__main__":
    # Get number of args (-1 to exclude the original file being counted as arg)
    num_args = len(sys.argv) - 1
    required_num_args = 2

    # Verify correct number of args passed
    if num_args == required_num_args:
        hmm_file = sys.argv[1]
        sequence = sys.argv[2]
    else:
        print("Invalid number of arguments. Expected:", file=sys.stderr)
        print("python3 hmm.py <hmm_file> \"<sequence of observations separated by space>\"", file=sys.stderr)
        print("Example: python3 hmm.py pos.hmm \"Time flies like an arrow\"", file=sys.stderr)
        sys.exit(-1)
    
    # Suppress warning from visualization package
    import warnings
    warnings.filterwarnings("ignore")

    print("Input: " + sequence)
    hmm = HMM(hmm_file)
    hmmf = HMM(hmm_file)
    print()
    print("Viterbi algorithm")
    hmm.viterbi(sequence)

    print()
    print("Forward algorithm")
    hmmf.forward(sequence)
    print("Viterbi and Forward algorithm trellis saved to file.")

