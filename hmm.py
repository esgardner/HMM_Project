#!/usr/bin/env python3
import sys, math, numpy, collections


class HMM_Parser():
    def parse(self, hmm_filename):
        # Variables for parsing results
        parsed_init_num = 0
        parsed_trans_num = 0
        parsed_emiss_num = 0
        init_state_probs = collections.defaultdict(float)
        transition_probs = collections.defaultdict(lambda: collections.defaultdict(float))
        states = set()
        emission_probs = collections.defaultdict(lambda: collections.defaultdict(float))
        emissions = set()
        emission_to_states = collections.defaultdict(set)

        with open(hmm_filename, 'r') as infile:
            parse_state = None
            for line in infile:
                # Strip any trailing whitespace from line (including newlines)
                line = line.rstrip()
                split_line = line.split()
                
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
                    init_state_probs[state] = probability

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
                        print("warning: duplicated transition {} {}".format(start_state, end_state),file=sys.stderr)
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
                        print("warning: duplicated emission {} {}".format(state, emission), file=sys.stderr)
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
                
                # We should never hit this condition
                else:
                    print("Unexpected item encountered in parsing:{}".format(line), file=sys.stderr)
                    sys.exit()


        return {
            "parsed_init_num": parsed_init_num,
            "parsed_trans_num": parsed_trans_num,
            "parsed_emiss_num": parsed_emiss_num,
            "init_state_probs": collections.defaultdict(float),
            "transition_probs": collections.defaultdict(float),
            "states": set(),
            "emission_probs": collections.defaultdict(lambda: collections.defaultdict(float)),
            "emissions": set(),
            "emission_to_states": collections.defaultdict(set),
        }
        

    def _get_last_int_in_line(self, line):
        try:
            return int(line.split()[-1])
        except:
            print("Error: failed to parse line:{}".format(line), file=sys.stderr)
            sys.exit()
            return 0

    def parse_init_line(self,line):
        # Split line on spaces
        split_line = line.split()

        # Extract state and prob
        state = split_line[0]
        prob = float(split_line[1])
        return (state, prob)
    
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
        self.__load_hmm(filename)
    
    def __load_hmm(self, hmm_filename):
        hmm_data = HMM_Parser().parse(hmm_filename)
        
    
if __name__ == "__main__":
    filename = "hmm_ex1"
    hmm = HMM(filename)
