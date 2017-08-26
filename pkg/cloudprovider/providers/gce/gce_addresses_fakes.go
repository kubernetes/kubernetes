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
	"encoding/json"
	"fmt"
	"net/http"

	computealpha "google.golang.org/api/compute/v0.alpha"
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
	addrsByRegionAndName map[string]map[string]*computealpha.Address
}

// FakeCloudAddressService Implements CloudAddressService
var _ CloudAddressService = &FakeCloudAddressService{}

func NewFakeCloudAddressService() *FakeCloudAddressService {
	return &FakeCloudAddressService{
		reservedAddrs:        make(map[string]bool),
		addrsByRegionAndName: make(map[string]map[string]*computealpha.Address),
	}
}

// SetRegionalAddresses sets the addresses of ther region. This is used for
// setting the test environment.
func (cas *FakeCloudAddressService) SetRegionalAddresses(region string, addrs []*computealpha.Address) {
	// Reset addresses in the region.
	cas.addrsByRegionAndName[region] = make(map[string]*computealpha.Address)

	for _, addr := range addrs {
		cas.reservedAddrs[addr.Address] = true
		cas.addrsByRegionAndName[region][addr.Name] = addr
	}
}

func (cas *FakeCloudAddressService) ReserveAlphaRegionAddress(addr *computealpha.Address, region string) error {
	if addr.Address == "" {
		addr.Address = fmt.Sprintf("1.2.3.%d", cas.count)
		cas.count++
	}

	if cas.reservedAddrs[addr.Address] {
		return &googleapi.Error{Code: http.StatusConflict}
	}

	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		cas.addrsByRegionAndName[region] = make(map[string]*computealpha.Address)
	}

	if _, exists := cas.addrsByRegionAndName[region][addr.Name]; exists {
		return &googleapi.Error{Code: http.StatusConflict}
	}

	cas.addrsByRegionAndName[region][addr.Name] = addr
	cas.reservedAddrs[addr.Address] = true
	return nil
}

func (cas *FakeCloudAddressService) ReserveRegionAddress(addr *compute.Address, region string) error {
	alphaAddr := convertToAlphaAddress(addr)
	return cas.ReserveAlphaRegionAddress(alphaAddr, region)
}

func (cas *FakeCloudAddressService) GetAlphaRegionAddress(name, region string) (*computealpha.Address, error) {
	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	}

	if addr, exists := cas.addrsByRegionAndName[region][name]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	} else {
		return addr, nil
	}
}

func (cas *FakeCloudAddressService) GetRegionAddress(name, region string) (*compute.Address, error) {
	addr, err := cas.GetAlphaRegionAddress(name, region)
	if addr != nil {
		return convertToV1Address(addr), err
	}
	return nil, err
}

func (cas *FakeCloudAddressService) GetRegionAddressByIP(region, ipAddress string) (*compute.Address, error) {
	if _, exists := cas.addrsByRegionAndName[region]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	}

	for _, addr := range cas.addrsByRegionAndName[region] {
		if addr.Address == ipAddress {
			return convertToV1Address(addr), nil
		}
	}
	return nil, makeGoogleAPINotFoundError("")
}

func convertToV1Address(object gceObject) *compute.Address {
	enc, err := object.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	var addr compute.Address
	if err := json.Unmarshal(enc, &addr); err != nil {
		panic(fmt.Sprintf("Failed to convert GCE apiObject %v to v1 address: %v", object, err))
	}
	return &addr
}

func convertToAlphaAddress(object gceObject) *computealpha.Address {
	enc, err := object.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	var addr computealpha.Address
	if err := json.Unmarshal(enc, &addr); err != nil {
		panic(fmt.Sprintf("Failed to convert GCE apiObject %v to alpha address: %v", object, err))
	}
	return &addr
}
