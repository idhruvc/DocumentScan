# This class stores the vertices of the text boxes in which the data field
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
#
# The integer member of each class called "nameFormat" represents how many lines the
# name takes up in the image.
# 1 -> name takes up 1 line, assumed to be FN LN
# 2 -> name takes up 1 line, assumed to be LN, FN (comma separated)
# 3 -> name takes up 2 lines, LN on first line, FN on second.
# 4 -> name takes up 2 lines, FN on first line, LN on second.


class SSN_H:
	orientation = "HORIZONTAL"
	nameFormat = 1

class TX_V:
	state = "TEXAS"
	orientation = "VERTICAL"
	dob = ((1320, 1565), (2200, 1725))
	name = ((165, 2310), (2095, 2552))
        address = ((171, 2679), (1749, 2904))
	expiration = ((1390, 2063), (2200, 2350))
	nameFormat = 3

class TX_H:
	state = "TEXAS"
	orientation = "HORIZONTAL"
	dob = ((632, 425), (1200, 516))
	name = ((506, 510), (1271, 621))
	address = ((512, 635), (1259, 760))
	expiration = ((1175, 368), (1535, 460))
	nameFormat = 3

class OR_H2:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = None
	name = None
	address = None
	expiration = None
	nameFormat = 2

class OR_H:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = ((532, 306), (699, 344))
	name = ((59, 525), (798, 568))
	address = ((59, 560), (798, 655))
	expiration = ((849, 194), (1050, 239))
	nameFormat = 2
