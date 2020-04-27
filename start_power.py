from attack.mnist_power.save_images_mnist_power import mnist_power_save_attacks

powers = [0, 1/10, 1/3, 1/2, 1, 2, 3, 10]

for power in powers:
    mnist_power_save_attacks(True, power, "test", 0, 100)

mnist_power_save_attacks(False, 1, "test", 0, 100)
