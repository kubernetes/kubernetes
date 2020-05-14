// +build libipmctl,cgo

// Copyright 2020 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nvm

// #cgo pkg-config: libipmctl
// #include <nvm_management.h>
import "C"
import (
	"fmt"
	info "github.com/google/cadvisor/info/v1"
	"sync"

	"k8s.io/klog/v2"
)

var (
	isNVMLibInitialized = false
	nvmLibMutex         = sync.Mutex{}
)

func init() {
	nvmLibMutex.Lock()
	defer nvmLibMutex.Unlock()
	cErr := C.nvm_init()
	if cErr != C.NVM_SUCCESS {
		// Unfortunately klog does not seem to work here. I believe it's better to
		// output information using fmt rather then let it disappear silently.
		fmt.Printf("libipmctl initialization failed with status %d", cErr)
	}
	isNVMLibInitialized = true
}

// getAvgPowerBudget retrieves configured power budget
// (in watts) for NVM devices. When libipmct is not available
// zero is returned.
func getAvgPowerBudget() (uint, error) {
	// Get number of devices on the platform
	// see: https://github.com/intel/ipmctl/blob/v01.00.00.3497/src/os/nvm_api/nvm_management.h#L1478
	count := C.uint(0)
	err := C.nvm_get_number_of_devices(&count)
	if err != C.NVM_SUCCESS {
		klog.Warningf("Unable to get number of NVM devices. Status code: %d", err)
		return uint(0), fmt.Errorf("Unable to get number of NVM devices. Status code: %d", err)
	}

	// Load basic device information for all the devices
	// to obtain UID of the first one.
	devices := make([]C.struct_device_discovery, count)
	err = C.nvm_get_devices(&devices[0], C.uchar(count))
	if err != C.NVM_SUCCESS {
		klog.Warningf("Unable to get all NVM devices. Status code: %d", err)
		return uint(0), fmt.Errorf("Unable to get all NVM devices. Status code: %d", err)
	}

	// Power budget is same for all the devices
	// so we can rely on any of them.
	device := C.struct_device_details{}
	err = C.nvm_get_device_details(&devices[0].uid[0], &device)
	if err != C.NVM_SUCCESS {
		uid := C.GoString(&devices[0].uid[0])
		klog.Warningf("Unable to get details of NVM device %q. Status code: %d", uid, err)
		return uint(0), fmt.Errorf("Unable to get details of NVM device %q. Status code: %d", uid, err)
	}

	return uint(device.avg_power_budget / 1000), nil
}

// getCapacities retrieves the total NVM capacity in bytes for memory mode and app direct mode
func getCapacities() (uint64, uint64, error) {
	caps := C.struct_device_capacities{}
	err := C.nvm_get_nvm_capacities(&caps)
	if err != C.NVM_SUCCESS {
		klog.Warningf("Unable to get NVM capacity. Status code: %d", err)
		return uint64(0), uint64(0), fmt.Errorf("Unable to get NVM capacity. Status code: %d", err)
	}
	return uint64(caps.memory_capacity), uint64(caps.app_direct_capacity), nil
}

// GetInfo returns information specific for non-volatile memory modules
func GetInfo() (info.NVMInfo, error) {
	nvmLibMutex.Lock()
	defer nvmLibMutex.Unlock()

	nvmInfo := info.NVMInfo{}
	if !isNVMLibInitialized {
		klog.V(1).Info("libimpctl has not been initialized. NVM information will not be available")
		return nvmInfo, nil
	}

	var err error
	nvmInfo.MemoryModeCapacity, nvmInfo.AppDirectModeCapacity, err = getCapacities()
	if err != nil {
		return info.NVMInfo{}, fmt.Errorf("Unable to get NVM capacities, err: %s", err)
	}

	nvmInfo.AvgPowerBudget, err = getAvgPowerBudget()
	if err != nil {
		return info.NVMInfo{}, fmt.Errorf("Unable to get NVM average power budget, err: %s", err)
	}
	return nvmInfo, nil
}

// Finalize un-initializes libipmctl. See https://github.com/google/cadvisor/issues/2457.
func Finalize() {
	nvmLibMutex.Lock()
	defer nvmLibMutex.Unlock()

	klog.V(1).Info("Attempting to un-initialize libipmctl")
	if !isNVMLibInitialized {
		klog.V(1).Info("libipmctl has not been initialized; not un-initializing.")
		return
	}

	C.nvm_uninit()
	isNVMLibInitialized = false
}
