
# wheelchair handrim contact detection
further implementation of the work hppd in which hand position wrt handrim was expressed using mediapipe and contact was assessed by means of torque signal collected by the ergometer
This project aims to develop a vision system that can detect whether or not the hand is in contact with the handrim during wheelchair propulsion.
Different measuring systems are being used apart from the camera, we have a wheelchair ergometer as well as a measuring wheel.
Previously the ground truth for the contact, to actually compare the vision system output, was done through the torque signal extracted from the ergometer. The current challenge is to switch the ground truth with a signal coming from a capacitive sleeve around the handrim, which we believe is more reliable and more accurate than the previous one. 
