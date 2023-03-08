import numpy as np

def phasor_data(u, y, time, freq):

	# numpy array for cosine wave
	Rc =   np.cos(freq * time)
	Rc.resize(Rc.shape[0], 1)

	# numpy array for sine wave
	Rs = - np.sin(freq * time)
	Rs.resize(Rs.shape[0], 1)

	R0 = np.concatenate((Rc, Rs), axis=1)

	# make regression to callculate the phasor U(s) and Y(s) of input signal u(t) and output signal y(t)
	U = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(R0), R0)), np.transpose(R0)), u)
	Y = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(R0), R0)), np.transpose(R0)), y)

	U = complex(U[0], U[1])
	Y = complex(Y[0], Y[1])

	return U, Y

# actaul parameters
kp   = 2.
p    = 5.

# time series (s)
time = np.linspace(0., 60., 60 * 1000 + 1)

U_list = []
Y_list = [] 

for freq in np.logspace(-1., 1., 21):

	# actual magnitude and phase
	TF  = kp * (p / (freq * 1j + p))
	mag = abs(TF)
	phs = np.angle(TF)

	# input and output signals
	u = 1.0 * np.sin(freq * time)
	y = mag * np.sin(freq * time + phs)

	# generate phasor data
	U, Y = phasor_data(u, y, time, freq)
	U_list.append([np.real(U)])
	U_list.append([np.imag(U)])
	Y_list.append([np.real(Y), np.real(Y * freq * 1j)])
	Y_list.append([np.imag(Y), np.imag(Y * freq * 1j)])

U_list = np.resize(U_list, [len(U_list), 1])	
Y_list = np.resize(Y_list, [len(Y_list), 2])	

# pseudo-inverse for regression
X0  = np.dot(np.dot(np.linalg.inv(np.dot(Y_list.T, Y_list)), Y_list.T), U_list)

# identified parameters
print "kp =",
print 1.0  / X0[0, 0]
print " p =",
print X0[0, 0] / X0[1, 0]