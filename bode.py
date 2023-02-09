import numpy as np

def bode_data(u, y, time, freq):

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

	# phasor of the transfer function
	CrossSpectrum = Y * U.conjugate()
	PowerSpectrum = U * U.conjugate()
	TF            = CrossSpectrum / PowerSpectrum

	# magnitude (dB) and phase (degree) of the transfer function
	magnitude_dB =  20. * np.log10(abs(TF))
	phase_dG     = 180. * (np.angle(TF) / np.pi)

	return magnitude_dB, phase_dG

# frequency (rad/s)
freq = 10.
# time series (s)
time = np.linspace(0., 60., 60 * 1000 + 1) 

# actual magnitude and phase
mag = 5.0
phs = np.pi / 4.

# input and output signals
u = 1.0 * np.sin(freq * time)
y = mag * np.sin(freq * time + phs)

# generate bode data
dB, dG = bode_data(u, y, time, freq)
print np.round(10. ** (dB/20.), 3), np.round(dG, 3)