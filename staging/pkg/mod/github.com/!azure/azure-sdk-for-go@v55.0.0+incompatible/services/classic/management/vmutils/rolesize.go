// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"fmt"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
	lc "github.com/Azure/azure-sdk-for-go/services/classic/management/location"
	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

// IsRoleSizeValid retrieves the available rolesizes using
// vmclient.GetRoleSizeList() and returns whether that the provided roleSizeName
// is part of that list
func IsRoleSizeValid(vmclient vm.VirtualMachineClient, roleSizeName string) (bool, error) {
	if roleSizeName == "" {
		return false, fmt.Errorf(errParamNotSpecified, "roleSizeName")
	}

	roleSizeList, err := vmclient.GetRoleSizeList()
	if err != nil {
		return false, err
	}

	for _, roleSize := range roleSizeList.RoleSizes {
		if roleSize.Name == roleSizeName {
			return true, nil
		}
	}

	return false, nil
}

// IsRoleSizeAvailableInLocation retrieves all available sizes in the specified
// location and returns whether that the provided roleSizeName is part of that list.
func IsRoleSizeAvailableInLocation(managementclient management.Client, location, roleSizeName string) (bool, error) {
	if location == "" {
		return false, fmt.Errorf(errParamNotSpecified, "location")
	}
	if roleSizeName == "" {
		return false, fmt.Errorf(errParamNotSpecified, "roleSizeName")
	}

	locationClient := lc.NewClient(managementclient)
	locationInfo, err := getLocation(locationClient, location)
	if err != nil {
		return false, err
	}

	for _, availableRoleSize := range locationInfo.VirtualMachineRoleSizes {
		if availableRoleSize == roleSizeName {
			return true, nil
		}
	}

	return false, nil
}

func getLocation(c lc.LocationClient, location string) (*lc.Location, error) {
	if location == "" {
		return nil, fmt.Errorf(errParamNotSpecified, "location")
	}

	locations, err := c.ListLocations()
	if err != nil {
		return nil, err
	}

	for _, existingLocation := range locations.Locations {
		if existingLocation.Name != location {
			continue
		}

		return &existingLocation, nil
	}
	return nil, fmt.Errorf("Invalid location: %s. Available locations: %s", location, locations)
}
