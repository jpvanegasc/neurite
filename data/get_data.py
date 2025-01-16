import keras


def format_data(mnist_data):
    (X_train, Y_train), (X_test, Y_test) = mnist_data

    # Turn the 28*28 images into 784-dimensional vectors where 0<=x[i]<=1
    X_train = X_train.reshape(60000, 784).astype("float32") / 255
    X_test = X_test.reshape(10000, 784).astype("float32") / 255

    return X_train, Y_train, X_test, Y_test


def main():
    data = keras.datasets.mnist.load_data()
    X_train, Y_train, _, _ = format_data(data)

    for i in range(10):
        csv_string = ""
        for value in X_train[i]:
            csv_string += str(value) + ","
        with open(f"{i}-value_{Y_train[i]}.csv", "w") as f:
            f.write(csv_string)


def test(filename):
    with open(filename, "r") as f:
        data = f.read().split(",")
    prettyprint_input_to_terminal(data)


def prettyprint_input_to_terminal(data):
    for i in range(28):
        for j in range(28):
            if float(data[i * 28 + j]) > 0.5:
                print("#", end="")
            elif float(data[i * 28 + j]) > 0:
                print("+", end="")
            else:
                print(" ", end="")
        print()


if __name__ == "__main__":
    test("0-value_5.csv")
