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

package gce

import (
	"fmt"
	"net/http"

	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

type FakeCloudAddressService struct {
	count int
	// reservedAddrs tracks usage of IP addresses
	// Key is the IP address as a string
	reservedAddrs map[string]bool
	// addrsByRegionAndName
	// Outer key is for region string; inner key is for address name.
	addrsByRegionAndName map[string]map[string]*compute.Address
}

func NewFakeCloudAddressService() *FakeCloudAddressService {
	return &FakeCloudAddressService{
		reservedAddrs:        make(map[string]bool),
		addrsByRegionAndName: make(map[string]map[string]*compute.Address),
	}
}

func (cas *FakeCloudAddressService) ReserveRegionAddress(addr *compute.Address, region string) error {
	if addr.Address == "" {
		addr.Address = fmt.Sprintf("1.2.3.%d", cas.count)
		cas.count++
	}

	if cas.reservedAddrs[addr.Address] {
		return &googleapi.Error{Code: http.StatusConflict}
	}

	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		cas.addrsByRegionAndName[region] = make(map[string]*compute.Address)
	}

	if _, exists := cas.addrsByRegionAndName[region][addr.Name]; exists {
		return &googleapi.Error{Code: http.StatusConflict}
	}

	cas.addrsByRegionAndName[region][addr.Name] = addr
	cas.reservedAddrs[addr.Address] = true
	return nil
}

func (cas *FakeCloudAddressService) GetRegionAddress(name, region string) (*compute.Address, error) {
	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	}

	if addr, exists := cas.addrsByRegionAndName[region][name]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	} else {
		return addr, nil
	}
}

func (cas *FakeCloudAddressService) GetRegionAddressByIP(region, ipAddress string) (*compute.Address, error) {
	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	}

	for _, addr := range cas.addrsByRegionAndName[region] {
		if addr.Address == ipAddress {
			return addr, nil
		}
	}
	return nil, makeGoogleAPINotFoundError("")
}
