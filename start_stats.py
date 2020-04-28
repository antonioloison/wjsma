from stats.stats import average_stat

powers = [0, 1/10, 1/3, 1/2, 1, 2, 3, 10]

for power in powers:
    print("POWER: ", power)
    average_stat("attack/mnist_power/wjsma_test_" + str(power) + "/")

print("JSMA")
average_stat("attack/mnist/jsma_test_1/")
