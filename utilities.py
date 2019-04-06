import tensorflow as tf
import numpy
import sys, os
import math

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0
    
def rampdown(epoch, rampdown_length, total_epoch):
    if epoch >= (total_epoch - rampdown_length):
        ep = (epoch - (total_epoch - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0 
     

if __name__ == "__main__":
    tf.app.run()
