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
