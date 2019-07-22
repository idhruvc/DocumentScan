class License:
	state = ''
	orientation = ''
	dob = '0/0/0000'
	first = ''
	last = ''
	address = ''
	expiration = ''

	def __str__(self):
		temp = ''
        	temp = temp + 'STATE:\t' + self.state + '\n'
		temp = temp + 'ORIEN:\t' + self.orientation + '\n'
		temp = temp + 'FIRST:\t' + self.first + '\n'
		temp = temp + 'LAST:\t' + self.last + '\n'
		temp = temp + 'DOB:\t' + self.dob + '\n'
		temp = temp + 'EXP:\t' + self.expiration + '\n'
		temp = temp + 'ADDR:\t' + self.address.replace('\n', ' ') + '\n'
		return temp


class SocialSecurity:
	name = ''
	ssn = ''
	
	def __str__(self):
		temp = ''
		temp = temp + 'NAME:\t' + self.name + '\n'
		temp = temp + 'SSN:\t' + self.ssn + '\n'
