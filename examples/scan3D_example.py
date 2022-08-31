import sys
sys.path.append("..")

from src.scan3D import Scan3D

def scan3D_example():
    scan = Scan3D()
    scan.connect()
    scan.print_device_info()
    scan.start()
    scan.capture()
    scan.write_depth_and_ir_to_file()
    scan.stop()

if __name__ == '__main__':
    scan3D_example()