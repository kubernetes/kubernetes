// Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

package cuda

// #cgo LDFLAGS: -lcudart_static
// #include <stdlib.h>
// #include <cuda_runtime_api.h>
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type MemoryInfo struct {
	ECC       bool
	Global    uint
	Shared    uint // includes L1 cache
	Constant  uint
	L2Cache   uint
	Bandwidth uint
}

type Device struct {
	handle C.int

	Family string
	Arch   string
	Cores  uint
	Memory MemoryInfo
}

func cudaErr(ret C.cudaError_t) error {
	if ret == C.cudaSuccess {
		return nil
	}
	err := C.GoString(C.cudaGetErrorString(ret))
	return errors.New(err)
}

var archToFamily = map[string]string{
	"1": "Tesla",
	"2": "Fermi",
	"3": "Kepler",
	"5": "Maxwell",
}

var archToCoresPerSM = map[string]uint{
	"1.0": 8,   // Tesla Generation (SM 1.0) G80 class
	"1.1": 8,   // Tesla Generation (SM 1.1) G8x G9x class
	"1.2": 8,   // Tesla Generation (SM 1.2) GT21x class
	"1.3": 8,   // Tesla Generation (SM 1.3) GT20x class
	"2.0": 32,  // Fermi Generation (SM 2.0) GF100 GF110 class
	"2.1": 48,  // Fermi Generation (SM 2.1) GF10x GF11x class
	"3.0": 192, // Kepler Generation (SM 3.0) GK10x class
	"3.2": 192, // Kepler Generation (SM 3.2) TK1 class
	"3.5": 192, // Kepler Generation (SM 3.5) GK11x GK20x class
	"3.7": 192, // Kepler Generation (SM 3.7) GK21x class
	"5.0": 128, // Maxwell Generation (SM 5.0) GM10x class
	"5.2": 128, // Maxwell Generation (SM 5.2) GM20x class
	"5.3": 128, // Maxwell Generation (SM 5.3) TX1 class
}

func GetDriverVersion() (string, error) {
	var driver C.int

	err := cudaErr(C.cudaDriverGetVersion(&driver))
	d := fmt.Sprintf("%d.%d", int(driver)/1000, int(driver)%100/10)
	return d, err
}

func NewDevice(busID string) (*Device, error) {
	var (
		dev  C.int
		prop C.struct_cudaDeviceProp
	)

	id := C.CString(busID)
	if err := cudaErr(C.cudaDeviceGetByPCIBusId(&dev, id)); err != nil {
		return nil, err
	}
	C.free(unsafe.Pointer(id))

	if err := cudaErr(C.cudaGetDeviceProperties(&prop, dev)); err != nil {
		return nil, err
	}
	arch := fmt.Sprintf("%d.%d", prop.major, prop.minor)
	cores, ok := archToCoresPerSM[arch]
	if !ok {
		return nil, fmt.Errorf("unsupported CUDA arch: %s", arch)
	}

	// Destroy the active CUDA context
	cudaErr(C.cudaDeviceReset())

	return &Device{
		handle: dev,
		Family: archToFamily[arch[:1]],
		Arch:   arch,
		Cores:  cores * uint(prop.multiProcessorCount),
		Memory: MemoryInfo{
			ECC:       bool(prop.ECCEnabled != 0),
			Global:    uint(prop.totalGlobalMem / (1024 * 1024)),
			Shared:    uint(prop.sharedMemPerMultiprocessor / 1024),
			Constant:  uint(prop.totalConstMem / 1024),
			L2Cache:   uint(prop.l2CacheSize / 1024),
			Bandwidth: 2 * uint((prop.memoryClockRate/1000)*(prop.memoryBusWidth/8)) / 1000,
		},
	}, nil
}

func CanAccessPeer(dev1, dev2 *Device) (bool, error) {
	var ok C.int

	err := cudaErr(C.cudaDeviceCanAccessPeer(&ok, dev1.handle, dev2.handle))
	return (ok != 0), err
}
