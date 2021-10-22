// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
	vmdisk "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachinedisk"
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
