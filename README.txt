//File: README.txt
//Author: Jonathan Carpenter
//Email: carpenter.102@wright.edu

This program performs text classification in a couple of ways.
The first method constructs features using Stemming and Stop words
and then selects features using Chi Square.
The second method constructs features using term weighted frequency, TFIDF,
and then selects features using Gain Ratio.
Finally models are constructed for both results using J48 decision tree,
Niave Bayes, and support vector machine with SMO.

To run this program on unix:
   1)unzip Carpenter_project2.zip
   2)type:  java -cp .:weka.jar project2 <int>
   	  where <int> is the number of features to be selected. 
