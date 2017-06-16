import pbrc
import grnn
import plant

def test_plant():
    for i in range(100):
        data = plant.get_next_data()
        print("x: {0:>30}; y: {1:>15}".format(str(data['x']), str(data['y'])))

def main():
    #test_plant()

if __name__ == "__main__":
    main()