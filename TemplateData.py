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
# state/document's shortened abbreviation and O represents the orientation of the
# document or driver's license (V or H)

class SSN_H:
	orientation = "HORIZONTAL"

class TX_V:
	state = "TEXAS"
	orientation = "VERTICAL"
	dob = ((1320, 1565), (2160, 1775))
        last = ((171, 2310), (2095, 2430))
       	first = ((165, 2437), (2074, 2552))
        address = ((171, 2679), (1749, 2904))
	expiration = ((1390, 2163), (2140, 2350))

class TX_H:
	state = "TEXAS"
	orientation = "HORIZONTAL"
	dob = ((632, 445), (1058, 516))
	last = ((506, 510), (1271, 566))
	first = ((506, 560), (1271, 621))
	address = ((512, 635), (1259, 760))
	expiration = ((1175, 368), (1535, 460))

# TODO Oregon might have more than one horizontal template? Not sure. TODO

class OR_H:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = ((633,308), (1025,353))
	first = ((635,306), (1023, 350))
	last = ((0,0), (0,0)) # TODO
	address = ((36, 779), (611,873))
	expiration = ((0,0), (0,0)) # TODO
