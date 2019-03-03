#!/usr/bin/env python3
import sys, collections
from prettytable import PrettyTable


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
    def parse_init_line(self,line):
        # Split line on spaces
        split_line = line.split()

        # Extract state and prob
        state = split_line[0]
        prob = float(split_line[1])
        return (state, prob)
    
    # Extract information about transitions from a line
    # "BOS DT 0.5" -> ("BOS", "DT", 0.5)
    def parse_transition_line(self,line):
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
    def parse_emission_line(self,line):
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


# HMM Class
class HMM():
    def __init__(self, hmm_filename):
        hmm_data = self.__load_hmm(filename)
        self.emissions = hmm_data['emissions']
        self.states = hmm_data['states']
        self.initial_state_probabilities = hmm_data['initial_state_probs']
        self.initial_states = hmm_data['initial_states']
        self.state_to_idx, self.idx_to_state = self._create_state_to_idx_mapping(self.states)
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
    def _create_state_to_idx_mapping(self, states):
        state_to_idx = collections.defaultdict(int)
        idx_to_state = collections.defaultdict(str)
        for state in states:
            idx = len(state_to_idx)
            state_to_idx[state] = idx
            idx_to_state[idx] = state
        return state_to_idx, idx_to_state

    # Run viterbi on a given input string
    def viterbi(self, input_str):
        def get_init_states_with_transition(curr_state):
            # States to return
            return_states = set()

            init_states = self.initial_states

            # Only return init states which have a transition to the current state
            for init_state in init_states:
                if init_state in self.transition_probabilities and curr_state in self.transition_probabilities[init_state]:
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
                print("Error: Encountered observation in input that is not in HMM")
                sys.exit()
        
        ## Set up our trellis and backpointers
        # Our trellis is a (len(input_seq)+1) row, len(self.states) col
        delta = [[0.0 for _ in range(len(input_seq)+1)] for _ in range(len(self.states))]
        back_p = [[0 for _ in range(len(input_seq)+1)] for _ in range(len(self.states))]

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
            # This is the trellis column index of the obvservation we are looking at
            observation_table_idx = observation_idx+1

            # Get a set of states that can emit for this observation
            valid_states = self.emission_to_states[observation]

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
                max_prob = 0.0
                max_state_idx = -1
                # Calculate the total probability from a previous state to the current state
                for valid_prev_observation_state in valid_prev_observation_states:
                    # Get the trellis row index of the previous state we are looking at
                    prev_state_idx = self.state_to_idx[valid_prev_observation_state]
                    
                    # Get the probability of that previous state from the trellis
                    state_prob = delta[prev_state_idx][observation_table_idx-1]

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
                
                # If we did not find a valid transition to current state, probability of state is 0
                else:
                    current_prob = 0
                    best_prev_state = -1

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
        
        # Create a trellis visualization
        # new_delta = delta.T

        # This will be the headers of our table
        table_headers = [str(idx+1) + ") " + item for idx, item in enumerate(input_seq)]
        table_headers.insert(0, "[States]")
        table_headers.insert(1, "[Initial probabilities]")

        # Create a PrettyTable to display trellis
        trellis = PrettyTable()
        for idx, row in enumerate(delta):
            row = ["%.8f" % number for number in row]
            state_name = self.idx_to_state[idx]
            new_row = []
            for item in row:
                try:
                    item = float(item)
                    if item > 0:
                        item = '\033[92m' + str(item) + '\033[0m'
                    new_row.append(str(item))
                except ValueError:
                    new_row.append(item)
            new_row.insert(0, state_name)
            trellis.add_row(new_row)
        trellis.field_names = table_headers

        # Print trellis
        print(trellis)

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
            output = input_str + " => " + " ".join(state_seq_outputs) + "\nProbability of sequence:" + str(highest_prob) + '\n'

            # Print all results
            print(output)

        
    
if __name__ == "__main__":
    filename = "hmm_ex1"
    hmm = HMM(filename)
    viterbi_res = hmm.viterbi("the store sold the book")
