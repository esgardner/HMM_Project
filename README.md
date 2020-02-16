# HMM_Project

## Project Overview
Our program implements the Viterbi and Forward algorithms for HMMs. For each algorithm, the user can input an observation sequence and see the results of the algorithm in a trellis image. The PrettyTable python package is used to display the trellis. These algorithms can be applied to various domains using a model with values for transitions, emissions, and initial states. For this project, we have chosen POS-tagging and snowfall prediction as our models.

## Techniques and Description 
Hidden Markov Models (HMMs) are a type of graphical markov model. They have a series of hidden variables (states)  and a set of observed variables. We can use the observed variables to predict the sequence of hidden variables. We implemented two algorithms that can be used to calculate probabilities for HMMs. We then applied these algorithms to two models. For the snowfall prediction model, the states are inches of snow and the observations are flight delays. For the POS tagging model, the states are parts of speech and the observations are words in a sentence. 

The first algorithm is the Viterbi algorithm, a dynamic programming algorithm that takes an observation sequence, and returns the state sequence that most likely produced the observations. 

The second is the forward algorithm, which is also a dynamic programming algorithm. This algorithm sums over the possible hidden state sequences to calculate how likely the observed sequence is. The results of the forward algorithm can be represented in a trellis, which is the output of running the forward algorithm in our code (see below for an example trellis). In the trellis, each cell represents the probability of being in the corresponding  state given the first t observations. 

HMMs are defined by an initial distribution, transitions, and emissions. The transitions values represent the probability of a transition from a state i to a state j. For the snowfall model, the states are 0-5 and they correspond to inches of snow. If there was no snow reported, that corresponds to the state 0. If there was <= 1 inch of snow, that corresponds to state 1, and so on up to state 5.  The transition data comes from the [National Weather Service](https://w2.weather.gov/climate/) and is based on reported snowfall in Chicago each day for the months of November 2018, December 2018, and January 2019. Laplace smoothing was used for calculating the probabilities. The emission data comes from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov) information about flight delays. The data used is for O’Hare airport and represents the number of flights delayed due to weather for each day in the time period from November 2018 - January 2019. The number of delays were broken down into three groups for the observations. <= 100 flights delayed corresponds to observation L (low), <=200 corresponds to the observation M (moderate) and > 200 flights delayed corresponds to the observation H (high). Using this model and our algorithms we can take a sequence of flight delay observations (for example L H L) and generate a trellis with the corresponding probability values for the various amounts of snowfall.

In the POS tagging model, the hidden states correspond to POS tags, where each POS tag state can emit a word. For example, the state DT (Determiner) will emit items like “the” or “a”. The model was handwritten based on intuition, and was made to be able to correctly tag the phrase “time flies like an arrow”, as well as other phrases that contain those same words (but tagged differently), such as “a fly”, “I like flies”. 

## Demo Instructions
The program can be run in the command line or in an IDE, such as PyCharm by specifying the parameters. 

Here is an example of how to run the program: 
Python3 hmm.py pos.hmm “Time flies like an arrow”

The name of the program file is hmm.py and to run it you need to pass it two arguments: the first is the model file (either pos.hmm or weather_flights.hmm). The second is the space-separated observation sequence, in this case the sentence “time flies like an arrow’.  





