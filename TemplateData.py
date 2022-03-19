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
	ssn = ((630, 490), (1395, 580))
	name = ((270, 670), (1736, 781))
	nameFormat = 1

class TX_V:
	state = "TEXAS"
	orientation = "VERTICAL"
	dob = ((1320, 1565), (2200, 1725))
	name = ((169, 2310), (2095, 2552))
	address = ((171, 2679), (1749, 2874))
	expiration = ((1390, 2115), (2200, 2310))
	nameFormat = 3

class TX_H:
	state = "TEXAS"
	orientation = "HORIZONTAL"
	dob = ((800, 545), (1360, 635))
	name = ((645, 650), (1600, 767))
	address = ((645, 845), (1600, 975))
	expiration = ((1510, 460), (1955, 545))
	nameFormat = 3

class OR_H2:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = ((173,515), (478,572))
	name = ((51,927), (900, 983))
	address = ((40, 970), (850, 1202))
	expiration = ((723, 318), (1069, 393))
	nameFormat = 2

class OR_H:
	state = "OREGON"
	orientation = "HORIZONTAL"
	dob = ((700, 406), (941, 460))
	name = ((27, 727), (793, 777))
	address = ((27, 777), (800, 907))
	expiration = ((1159,252), (1450, 315))
	nameFormat = 2
