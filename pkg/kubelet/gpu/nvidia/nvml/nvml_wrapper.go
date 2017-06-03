/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nvml

// #include <stdlib.h>
/*
#define NVML_DEVICE_NAME_BUFFER_SIZE                  64

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,          //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef struct nvmlDevice {
    char * name;
} nvmlDevice_t;

typedef unsigned int uint;

nvmlReturn_t my_nvmlInit(void *f)
{
   size_t (*nvmlInit_v2)();
   nvmlInit_v2 = (size_t (*)())f;
   return nvmlInit_v2();
}

nvmlReturn_t my_nvmlShutdown(void *f)
{
    size_t (*nvmlShutdown)();
    nvmlShutdown = (size_t (*)())f;
    return nvmlShutdown();
}

nvmlReturn_t my_nvmlSystemGetDriverVersion(void *f, const char *version, const uint *length)
{
	size_t (*nvmlSystemGetDriverVersion)(const char *, const uint *);
	nvmlSystemGetDriverVersion = (size_t (*)(const char *, const uint *))f;
	return nvmlSystemGetDriverVersion(version, length);
}

nvmlReturn_t my_nvmlDeviceGetCount(void *f, const uint *dev)
{
    size_t (*nvmlDeviceGetCount)(const uint *);
    nvmlDeviceGetCount = (size_t (*)(const uint *))f;
    return nvmlDeviceGetCount(dev);
}

nvmlReturn_t my_nvmlDeviceGetHandleByIndex(void *f, const uint n, const nvmlDevice_t *addr)
{
    size_t (*nvmlDeviceGetHandleByIndex)(const uint , const nvmlDevice_t *);
    nvmlDeviceGetHandleByIndex = (size_t (*)(const uint , const nvmlDevice_t *))f;
    return nvmlDeviceGetHandleByIndex(n, addr);
}

nvmlReturn_t my_nvmlDeviceGetName(void *f, const nvmlDevice_t dev, const char *name, const uint buffer)
{
    size_t (*nvmlDeviceGetName)(const nvmlDevice_t , const char *, const uint);
    nvmlDeviceGetName = (size_t (*)(const nvmlDevice_t , const char *, const uint))f;
    return nvmlDeviceGetName(dev, name, buffer);
}

nvmlReturn_t my_nvmlDeviceGetMinorNumber(void *f, const nvmlDevice_t dev, const uint *minor)
{
    size_t (*nvmlDeviceGetMinorNumber)(const nvmlDevice_t , const uint *);
    nvmlDeviceGetMinorNumber = (size_t (*)(const nvmlDevice_t, const uint *))f;
    return nvmlDeviceGetMinorNumber(dev, minor);
}

char* my_nvmlErrorString(void *f, const nvmlReturn_t ret)
{
    size_t (*nvmlErrorString)();
    nvmlErrorString = (size_t (*)(const nvmlReturn_t))f;
    return (void *)nvmlErrorString(ret);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type NvmlWrapper struct{}

var (
	handle     *LibHandle
	handle_map map[string]unsafe.Pointer
	lib_files  = []string{"libnvidia-ml.so.1"}
	lib_names  = []string{
		"nvmlInit_v2",
		"nvmlShutdown",
		"nvmlSystemGetDriverVersion",
		"nvmlErrorString",
		"nvmlDeviceGetCount",
		"nvmlDeviceGetHandleByIndex",
		"nvmlDeviceGetMinorNumber",
		"nvmlDeviceGetName",
	}
)

/* Inner nvml wappers */
func (nvml *NvmlWrapper) initLib() error {
	handle_map = make(map[string]unsafe.Pointer)
	libs := lib_files
	h, err := GetHandle(libs)
	if err != nil {
		return fmt.Errorf("couldn't get a handle to the library: %v", err)
	}
	for _, v := range lib_names {
		p, err := h.GetSymbolPointer(v)
		if err != nil {
			return fmt.Errorf("couldn't get symbol `%v`: %v", v, err)
		}
		handle_map[v] = p
	}
	handle = h
	return nil
}

func (nvml *NvmlWrapper) nvmlError(ret C.nvmlReturn_t) error {
	if ret == C.NVML_SUCCESS {
		return nil
	}
	err := C.GoString(C.my_nvmlErrorString(handle_map["nvmlErrorString"], ret))
	return fmt.Errorf("NVML Error: %v", err)
}

func (nvml *NvmlWrapper) nvmlInit() error {
	if err := nvml.initLib(); err != nil {
		return err
	}
	return nvml.nvmlError(C.my_nvmlInit(handle_map["nvmlInit_v2"]))
}

func (nvml *NvmlWrapper) nvmlShutdown() error {
	defer handle.Close()
	return nvml.nvmlError(C.my_nvmlShutdown(handle_map["nvmlShutdown"]))
}

func (nvml *NvmlWrapper) nvmlSystemGetDriverVersion() (string, uint, error) {
	var length C.uint
	var version string
	cVersion := C.CString(version)
	err := nvml.nvmlError(C.my_nvmlSystemGetDriverVersion(handle_map["nvmlSystemGetDriverVersion"], cVersion, &length))
	version = C.GoString(cVersion)
	C.free(unsafe.Pointer(cVersion))
	return version, uint(length), err
}

func (nvml *NvmlWrapper) nvmlGetDeviceCount() (uint, error) {
	var num C.uint
	err := nvml.nvmlError(C.my_nvmlDeviceGetCount(handle_map["nvmlDeviceGetCount"], &num))
	return uint(num), err
}

func (nvml *NvmlWrapper) nvmlDeviceGetHandleByIndex(idx uint) (C.nvmlDevice_t, error) {
	var dev C.nvmlDevice_t
	err := nvml.nvmlError(C.my_nvmlDeviceGetHandleByIndex(handle_map["nvmlDeviceGetHandleByIndex"], C.uint(idx), &dev))
	return dev, err
}

func (nvml *NvmlWrapper) nvmlDeviceGetName(dev C.nvmlDevice_t) (string, error) {
	var name string
	cName := C.CString(name)
	err := nvml.nvmlError(C.my_nvmlDeviceGetName(handle_map["nvmlDeviceGetName"], dev, cName, C.NVML_DEVICE_NAME_BUFFER_SIZE))
	name = C.GoString(cName)
	C.free(unsafe.Pointer(cName))
	return name, err
}

func (nvml *NvmlWrapper) nvmlDeviceGetMinorNumber(dev C.nvmlDevice_t) (uint, error) {
	var minor C.uint
	err := nvml.nvmlError(C.my_nvmlDeviceGetMinorNumber(handle_map["nvmlDeviceGetMinorNumber"], dev, &minor))
	return uint(minor), err
}

/* Helper functions */
// NvmlInit() initializes nvml lib.
func (nvml *NvmlWrapper) NvmlInit() error {
	return nvml.nvmlInit()
}

// NvmlShutdown() shutdowns nvml lib.
func (nvml *NvmlWrapper) NvmlShutdown() error {
	return nvml.nvmlShutdown()
}

// NvmlGetDriverVersion() returns GPU driver version.
func (nvml *NvmlWrapper) NvmlGetDriverVersion() (string, error) {
	version, _, err := nvml.nvmlSystemGetDriverVersion()
	if err != nil {
		return "N/A", nil
	}

	return version, nil
}

// NvmlGetDeviceCount() returns GPU numbers detected by nvml
func (nvml *NvmlWrapper) NvmlGetDeviceCount() (uint, error) {
	return nvml.nvmlGetDeviceCount()
}

// NvmlGetDeviceNameByIdx(idx uint) returns GPU name (such as "TeslaK80") by its index.
func (nvml *NvmlWrapper) NvmlGetDeviceNameByIdx(idx uint) (string, error) {
	var name string
	var err error
	var dev C.nvmlDevice_t

	if dev, err = nvml.nvmlDeviceGetHandleByIndex(idx); err != nil {
		// use a large number to notify minor number error
		return "NA", err
	}
	if name, err = nvml.nvmlDeviceGetName(dev); err != nil {
		return "NA", err
	}
	return name, nil
}

// NvmlGetDeviceMinorByIdx(idx uint) returns GPU minor number X used in "/dev/nvidiaX" by its index. -1 for error.
func (nvml *NvmlWrapper) NvmlGetDeviceMinorByIdx(idx uint) (int, error) {
	var minor uint
	var err error
	var dev C.nvmlDevice_t

	if dev, err = nvml.nvmlDeviceGetHandleByIndex(idx); err != nil {
		// use a large number to notify minor number error
		return -1, err
	}
	if minor, err = nvml.nvmlDeviceGetMinorNumber(dev); err != nil {
		return -1, err
	}

	return int(minor), nil
}
func NewNvmlWrapper() Nvml {
	return &NvmlWrapper{}
}
