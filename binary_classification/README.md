# Strategic Fairness Project

The goal of group-fair classification algorithms is to correct the unfair treatment a 
minority or disadvantaged group might receive due to inherent bias in the training dataset.
Without such algorithms, we note that those in the disadvantaged group have an incentive
to report themselved as belonging to the preferred group.

Our goal here is to observe whether the use of fair classification algorithms creates 
a reverse-incentive. That is, whether those in the preferred group are now likely to 
report themselves as belonging to the disadvanaged group.

We focus on post-processed fair algorithms like Demographic Parity, Equalized Odds,
and equalized opportunity

# Datasets used in this project

* Compas Dataset
    ** Race
    ** Sex

* Lawschool Dataset
    ** Race
    ** Sex

* Income Dataset
    ** Race
    ** Sex

# Classifiers used in this model

We use only hard classifiers in this work (output of classification is hard label instead
of a probability since all post processed classifiers we consider are generally based on 
the base output being 0/1 (as they are equalizing the fp and tn rates)

## ERM classifier

This is a control classifier that simply minimizes the loss. This is used as the first 
step for all post-processed classifiers. Currently using a logistic regression model
to implement this, but we can change this if needed. **See erm\_classifier.py**

## Equalized Odds classifier

This is an equalized odds  classifier. We allow for the fairness constraint to be
both hard and soft. **See equalized\_odds\_classifier.py**

## Equalized Opportunity classifier

This is an equalized opportunity  classifier. We allow for the fairness constraint to be
both hard and soft. **See equalized\_opportunity\_classifier.py**

## Demographic Parity classifier

This is an DP classifier. We allow for the fairness constraint to be
both hard and soft. **See demographic\_parity\_classifier.py**

