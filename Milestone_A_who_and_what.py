#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Hidden Markov Models: Algorithms and Applications']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Project in CSE 415, University of Washington, Winter, 2019
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Hidden_Markov_Models_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:

erica = Partner("Gardner", "Erica", "erstgard")
einar = Partner("Horn", "Einar", "einarh")
team = [erica, einar]

OPTION = '3'
# Legal options are 1, 2, and 3.

title = "HMMs for two different domains"

approach = '''We will first decide on a fixed format for HMM files. Then, we will create HMM model 
files for two domains: POS (Part-of-speech) tagging and a second 
domain yet to be decided. Next, we will begin our implementation process 
by first building a library to interact with the HMM files, and then implement 
the Forward Algorithm and Viterbi. Finally, we will provide a way for the user 
to specify a custom input sequence to our HMM algorithms, and add a 
GUI component that displays the HMM trellis.'''

workload_distribution = '''Both of us will be involved in determining the format for the HMM files and models. 
Each of us will be responsible for one domain model. Erica will handle the forward algorithm and Einar 
will do Viterbi algorithm as well as the visual component for the trellis diagram.'''

reference1 = '''Medium article on hidden markov Models by Sanjay Dorairaj;
    URL: https://medium.com/@postsanjay/hidden-markov-models-simplified-c3f58728caab'''

reference2 = '''Freecodecamp article on POS tagging and HMMs by Sachin Malhotra and Divya Godayal,
    URL: https://medium.freecodecamp.org/an-introduction-to-part-of-speech-tagging-and-the-hidden-markov-model-953d45338f24'''

our_submission = Who_and_what([erica, einar], OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())
