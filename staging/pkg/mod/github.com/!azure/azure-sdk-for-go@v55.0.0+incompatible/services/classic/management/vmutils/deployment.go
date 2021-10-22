// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"fmt"

	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

// ConfigureDeploymentFromRemoteImage configures VM Role to deploy from a remote
// image source. "remoteImageSourceURL" can be any publically accessible URL to
// a VHD file, including but not limited to a SAS Azure Storage blob url. "os"
// needs to be either "Linux" or "Windows". "label" is optional.
func ConfigureDeploymentFromRemoteImage(
	role *vm.Role,
	remoteImageSourceURL string,
	os string,
	newDiskName string,
	destinationVhdStorageURL string,
	label string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.OSVirtualHardDisk = &vm.OSVirtualHardDisk{
		RemoteSourceImageLink: remoteImageSourceURL,
		MediaLink:             destinationVhdStorageURL,
		DiskName:              newDiskName,
		OS:                    os,
		DiskLabel:             label,
	}
	return nil
}

// ConfigureDeploymentFromPlatformImage configures VM Role to deploy from a
// platform image. See osimage package for methods to retrieve a list of the
// available platform images. "label" is optional.
func ConfigureDeploymentFromPlatformImage(
	role *vm.Role,
	imageName string,
	mediaLink string,
	label string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.OSVirtualHardDisk = &vm.OSVirtualHardDisk{
		SourceImageName: imageName,
		MediaLink:       mediaLink,
	}
	return nil
}

// ConfigureDeploymentFromPublishedVMImage configures VM Role to deploy from
// a published (public) VM image.
func ConfigureDeploymentFromPublishedVMImage(
	role *vm.Role,
	vmImageName string,
	mediaLocation string,
	provisionGuestAgent bool) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.VMImageName = vmImageName
	role.MediaLocation = mediaLocation
	role.ProvisionGuestAgent = provisionGuestAgent
	return nil
}

// ConfigureDeploymentFromUserVMImage configures VM Role to deploy from a previously
// captured (user generated) VM image.
func ConfigureDeploymentFromUserVMImage(
	role *vm.Role,
	vmImageName string) error {
	if role == nil {
		return fmt.Errorf(errParamNotSpecified, "role")
	}

	role.VMImageName = vmImageName
	return nil
}

// ConfigureDeploymentFromExistingOSDisk configures VM Role to deploy from an
// existing disk. 'label' is optional.
func ConfigureDeploymentFromExistingOSDisk(role *vm.Role, osDiskName, label string) error {
	role.OSVirtualHardDisk = &vm.OSVirtualHardDisk{
		DiskName:  osDiskName,
		DiskLabel: label,
	}
	return nil
}
