import yaml


def load_network():
    with open("network.yaml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)


if __name__ == "__main__":
    load_network()
