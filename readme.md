# PHOEBE BUFFAY

Have you ever thought "I want to use the PHOEBE code to make inferences 
about a systems parameters, but it is just so slow."? Well, we had the 
same thought. More importantly, we wanted to use PHOEBE for vague 
physics reasons, pay homage to Lisa Kudrow, and make an amazing pun 
all at the same time. Hence we have PHOEBEBuffay - an emulator for the
PHOEBE2 modelling code. 

## Some interesting engineering points
The point of an emulator is generally to do something more efficiently.
But, we want to make sure that the emulator is giving us results that
make sense. Making an emulator for a code like PHOEBE makes addressing 
this tractable since the cost of a single model isn't insane like you 
have in full-blown cosmological simulations. 

Another interesting point that we want to build into our emulator is 
the bounds of the physical parameter space. We're going to train our
emulator on a parameter space that makes physical sense (more or less). 
There are situations in which we could propose a parameter vector 
that results in a binary that experiences Roche Lobe overflow - meaning
that PHOEBE would produce nonsense results. We want to make sure that
the emulator doesn't return us something that it thinks is good, but is
actually out of bounds.

## People involved
Nora Eisner
Cole Johnston
Valentina Tardugno
David Hogg
