import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Define the parameters and the function
theta = np.linspace(0, 2 * np.pi, 500)
i = 1j

# Initialize alpha
initial_alpha = np.pi / 4

# Define the initial expression and modulus squared
expression = (
    (1 - i) * np.sin(initial_alpha) ** 2
    - 4 * (1 + i) * np.sin(theta) ** 4
    + 4 * (1 + i) * np.sin(theta) ** 2
    - 1
)
modulus_squared = np.abs(expression) ** 2

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)
(line,) = ax.plot(theta, modulus_squared, label=r"$|\ldots|^2$")
ax.set_title("Modulus Squared Expression for α")
ax.set_xlabel(r"$\theta$ (radians)")
ax.set_ylabel(r"$|f(\theta)|^2$")
ax.grid(True)
ax.legend()

# Add a slider for alpha
alpha_slider_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03])
alpha_slider = Slider(alpha_slider_ax, "α", 0, 2 * np.pi, valinit=initial_alpha)


# Update function for the slider
def update(val):
    alpha = alpha_slider.val
    expression = (
        (1 - i) * np.sin(alpha) ** 2
        - 4 * (1 + i) * np.sin(theta) ** 4
        + 4 * (1 + i) * np.sin(theta) ** 2
        - 1
    )
    line.set_ydata(np.abs(expression) ** 2)
    fig.canvas.draw_idle()


alpha_slider.on_changed(update)

plt.show()
