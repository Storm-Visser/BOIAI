#%%
import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 0 to 120
x = np.linspace(0, 128, 1000)  # Generate 1000 points between 0 and 120

# Compute the y values using the sine function
y = np.sin(x)  # Scale the sine wave to range from 0 to 1

# Plot the sine wave
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print(2**-8)
print(2**15/128)
print(2**15)
print(2**7)
