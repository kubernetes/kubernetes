// +build go1.7

package vmutils

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/management/virtualmachine"
	vmdisk "github.com/Azure/azure-sdk-for-go/management/virtualmachinedisk"
)

// ConfigureWithNewDataDisk adds configuration for a new (empty) data disk
func ConfigureWithNewDataDisk(role *vm.Role, label, destinationVhdStorageURL string, sizeInGB int, cachingType vmdisk.HostCachingType) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	appendDataDisk(role, vm.DataVirtualHardDisk{
		DiskLabel:           label,
		HostCaching:         cachingType,
		LogicalDiskSizeInGB: sizeInGB,
		MediaLink:           destinationVhdStorageURL,
	})

	return nil
}

// ConfigureWithExistingDataDisk adds configuration for an existing data disk
func ConfigureWithExistingDataDisk(role *vm.Role, diskName string, cachingType vmdisk.HostCachingType) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	appendDataDisk(role, vm.DataVirtualHardDisk{
		DiskName:    diskName,
		HostCaching: cachingType,
	})

	return nil
}

// ConfigureWithVhdDataDisk adds configuration for adding a vhd in a storage
// account as a data disk
func ConfigureWithVhdDataDisk(role *vm.Role, sourceVhdStorageURL string, cachingType vmdisk.HostCachingType) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	appendDataDisk(role, vm.DataVirtualHardDisk{
		SourceMediaLink: sourceVhdStorageURL,
		HostCaching:     cachingType,
	})

	return nil
}

func appendDataDisk(role *vm.Role, disk vm.DataVirtualHardDisk) {
	disk.Lun = len(role.DataVirtualHardDisks)
	role.DataVirtualHardDisks = append(role.DataVirtualHardDisks, disk)
}
