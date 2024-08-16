Hidden Markov Model: Parts-of-Speech Tagging Artificial Intelligence Project

Created as part of Society of Hispanic Engineers at the University of Illinois Urbana-Champaign
by Nathaniel Kusiolek in October 2023

The program predicts which part-of-speech each word is in a given input file (ex: browncorpus-dev.txt).
The program is trained using a separate file as its training data (ex: browncorpus-training.txt).
When the program runs, it outputs the AI's prediction accuracy on the test data.
It also outputs:
the accuracy of words with multiple tags
the accuracy of words that only appear in the test data (unseen words in the training data)
the words that were predicted incorrectly the most
the words that were predicted correctly the most


List of files:
data folder: -training files are used to train the AI, -dev files are the input given to the AI
main.py - Main program, runs the other 3 files (provided by SHPE)
utilities.py - Utility functions to test the AI functionality and accuracy (provided by SHPE)
base_viterbi.py - HMM Viterbi algorithm to predict the part-of-speech of the next word (created by me)
optimized_viterbi.py - more advanced version of the Viterbi algorithm that handles special word cases (created by me)


Instructions:
1. Open windows terminal in the main project folder.
2. Run one of the following: 
	python main.py --train data/browncorpus-training.txt --test data/browncorpus-dev.txt --algorithm base_viterbi
	python main.py --train data/browncorpus-training.txt --test data/browncorpus-dev.txt --algorithm optimized_viterbi
	python main.py --train data/minitest-training.txt --test data/minitest-dev.txt --algorithm base_viterbi
	python main.py --train data/minitest-training.txt --test data/minitest-dev.txt --algorithm optimized_viterbi
3. Output is accuracy of the AI's predictions.
4. Note: base_viterbi takes a minute or two to run, optimized_viterbi can take 5+ minutes to finish.
5. Can add any .txt files to the data folder to train the AI with different data or test the AI with different data.


How it works:
The Viterbi tagger is an implementation of the HMM trellis (Viterbi) decoding algorithm. That is, the probability of each tag depends only on the previous tag, and the probability of each word depends only on the corresponding tag. This model will need to estimate three sets of probabilities:
Initial probabilities
Transition probabilities (How often does tag b follow tag a?)
Emission probabilities (How often does tag t yield word w?) 

Five steps:
1. Count occurrences of tags, tag pairs, tag/word pairs.
2. Compute smoothed probabilities.
3. Take the log of each probability.
4. Construct the trellis. Notice that for each tag/time pair, you must store not only the probability of the best path but also pointer to the previous tag/time pair in that path.
5. Return the best path through the trellis by backtracking. 

Laplace smoothing is a good choice for a smoothing method to increase performance.

For example, to smooth the emission probabilities, consider each tag individually. For some tag T, we need to ensure that Pe(W|T)
produces a non-zero number no matter what word W we give it. We use Laplace smoothing to fill in a probability for "UNKNOWN," to use as the shared probability value for all words W that were not seen in the training data for said tag T. The emission probabilities Pe(W|T) should add up to 1 when we keep T fixed but sum over all words W (including UNKNOWN). Mathematically,

α = Laplace smoothing constant
VT = number of unique words seen in training data for tag T
nT = total number of words in training data for tag T
Pe(UNKNOWN|T) = α/(nT+α(VT+1))
Pe(W|T) = (count(W)+α)/(nT+α(VT+1))

Now repeat the Laplace smoothing process to smooth the emission probabilities for all the other tags, one tag at a time.

Similarly, to smooth the transition probabilities, consider one specific tag T. Use Laplace smoothing to fill any zeroes in the probabilities of which tags can follow T. The transition probabilities Pe(Tb|T) should add up to 1 when we keep T constant and sum across all following tags Tb. Now repeat this process, replacing T with all the other possible first tags. The Laplace smoothing constant α should be the same for all first tags, but it might be different from the constant that you used for the emission probabilities.


base_viterbi:
The above description of the Viterbi has only a 93% accuracy because it does very poorly on unseen words. It's assuming that all tags have similar probability for these words, but we know that a new word is much more likely to have the tag NOUN than CONJ. We can improve emission smoothing to match the real probabilities for unseen words.

Words that occur only once in the training data ("hapax" words) have a distribution similar to the words that appear only in the test/development data. We can extract these words from the training data and calculate the probability of each tag on them. When we do our Laplace smoothing of the emission probabilities for tag T, scale Laplace smoothing constant α by the corresponding probability that tag T occurs among the set hapax words.

This optimized version of the Viterbi code has a significantly better unseen word accuracy on the Brown development dataset (66.44%). It also beat the baseline on overall accuracy (95.62%).


optimized_viterbi:
Notice that words with certain prefixes and certain suffixes typically have certain limited types of tags. For example, words with suffix "-ly" have several possible tags but the tag distribution is very different from that of the full set of hapax words. You can do a better job of handling these words by changing the emissions probabilities generated for them.

I created a long list of different prefixes and suffixes that had different tag distributions than the rest of the unseen words. The full list is: -ing, -ly, -ion, -er, -en, -ity, -ness, -ed, -es, -al, -ive, -ic, -ous, -able, inter-, -co, -at, -ful, -a, -i, and -s.

Using this method, the model solution gets 76.31% accuracy on unseen words, and over 96.07% accuracy overall. (Both numbers on the Brown development dataset.)
