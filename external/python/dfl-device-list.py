import sys
import ctypes
import os
import argparse

class Device(object):
    def __init__(self, index, name, total_mem, free_mem, cc=0):
        self.index = index
        self.name = name
        self.cc = cc
        self.total_mem = total_mem
        self.total_mem_gb = total_mem / 1024**3
        self.free_mem = free_mem
        self.free_mem_gb = free_mem / 1024**3

    def __str__(self):
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb :.3}]"

class Devices(object):
    all_devices = None

    def __init__(self, devices):
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __getitem__(self, key):
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        for device in self.devices:
            yield device

    def get_best_device(self):
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_device_by_index(self, idx):
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append (device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb*(1024**3):
                result.append (device)
        return Devices(result)

    @staticmethod
    def initialize_main_env():
        min_cc = int(os.environ.get("TF_MIN_REQ_CAP", 35))
        libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except:
                continue
            else:
                break
        else:
            return Devices([])

        nGpus = ctypes.c_int()
        name = b' ' * 200
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        freeMem = ctypes.c_size_t()
        totalMem = ctypes.c_size_t()

        result = ctypes.c_int()
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        error_str = ctypes.c_char_p()

        devices = []

        if cuda.cuInit(0) == 0 and \
            cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
            for i in range(nGpus.value):
                if cuda.cuDeviceGet(ctypes.byref(device), i) != 0 or \
                    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) != 0 or \
                    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != 0:
                    continue

                if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0:
                    if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                        cc = cc_major.value * 10 + cc_minor.value
                        if cc >= min_cc:
                            devices.append ( {'name'      : name.split(b'\0', 1)[0].decode(),
                                              'total_mem' : totalMem.value,
                                              'free_mem'  : freeMem.value,
                                              'cc'        : cc
                                              })
                    cuda.cuCtxDetach(context)

        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(devices))
        for i, device in enumerate(devices):
            os.environ[f'NN_DEVICE_{i}_NAME'] = device['name']
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(device['total_mem'])
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(device['free_mem'])
            os.environ[f'NN_DEVICE_{i}_CC'] = str(device['cc'])

    @staticmethod
    def getDevices():
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in main process.")
            devices = []
            for i in range ( int(os.environ['NN_DEVICES_COUNT']) ):
                devices.append ( Device(index=i,
                                        name=os.environ[f'NN_DEVICE_{i}_NAME'],
                                        total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                                        free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']),
                                        cc=int(os.environ[f'NN_DEVICE_{i}_CC']) ))
            Devices.all_devices = Devices(devices)

        return Devices.all_devices

def parse_additional_paths(paths):
    paths_split = paths.split(',')

    for path in paths_split:
        if len(path) == 0:
            continue

        path_split = path.split('%')
        if len (path_split) != 2:
            continue

        if path_split[0] in os.environ:
            os.environ[path_split[0]] += ';' + path_split[1]
        else:
            os.environ[path_split[0]] = path_split[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths")
    args = parser.parse_args()

    if args.paths is not None:
        parse_additional_paths(args.paths)

    Devices.initialize_main_env()
    devices = Devices.getDevices()

    for device in devices:
        sys.stdout.write(str(device.index) + '%' + device.name + '\n')
