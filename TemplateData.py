# This class stores the vertices of the text boxes in which the ID field
# is expected to be AFTER the image has been aligned with its proper template.
#
# key::
#	x1,y1 ------
#	|          |	field = ((x1, y1), (x2, y2))
#	|          |
#	|          |
#      	--------x2,y2
#
# Name of the class is formatted ST_O, where ST is a placeholder for the
# state's 2-letter abbreviation, and O represents the orientation of the
# document or driver's license (V or H)

class SS_H:
	orientation = "HORIZONTAL"

class TX_V:
	state = "TEXAS"
	orientation = "VERTICAL"
	dob = ((1320, 1565), (2145, 1730))
        last = ((171, 2310), (2095, 2430))
       	first = ((165, 2437), (2074, 2552))
        address = ((171, 2679), (1749, 2904))

class TX_H:
	state = "TEXAS"
	orientation = "HORIZONTAL"
	# TODO

# TODO don't know if oregon has a vertical template? class OR_H2 TODO

class OR_H:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = ((633,308), (1025,353))
	first = ((635,306), (1023, 350))
	last = ((0,0), (0,0)) # TODO
	address = ((36, 779), (611,873))
