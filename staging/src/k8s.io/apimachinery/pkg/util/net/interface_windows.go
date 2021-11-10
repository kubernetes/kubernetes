//go:build windows
// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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

package net

import (
	"fmt"
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"

	"k8s.io/klog/v2"
)

var getIPForwardEntry2 = windows.NewLazySystemDLL("iphlpapi.dll").NewProc("GetIpForwardEntry2")

// ipForwardRow2 describes an entry in the Windows routing table
type ipForwardRow2 struct {
	InterfaceLuid        netLUID
	InterfaceIndex       uint32
	DestinationPrefix    ipAddressPrefix
	NextHop              rawSockaddrInet
	SitePrefixLength     uint8
	ValidLifetime        uint32
	PreferredLifetime    uint32
	Metric               uint32
	Protocol             uint32
	Loopback             uint8
	AutoconfigureAddress uint8
	Publish              uint8
	Immortal             uint8
	Age                  uint32
	Origin               uint32
}

// netLUID identifies an NDIS network interface
type netLUID struct {
	Value uint64
}

// ipAddressPrefix describes an IP address prefix
type ipAddressPrefix struct {
	Prefix       rawSockaddrInet
	PrefixLength uint8
}

// rawSockaddrInet is a union which can describe either an IPv4 or IPv6 address.
type rawSockaddrInet struct {
	Family uint16
	Data   [26]byte
}

// getDefaultRouteEntry returns the default route for the given family, or nil if the route does not exist
func getDefaultRouteEntry(family uint16) (*ipForwardRow2, error) {
	// Use the GetIpForwardEntry2 function to retrieve the default route
	// The destination prefix will be 0.0.0.0/0 for IPv4 and ::0/0 for IPv6.
	defaultRoute := &ipForwardRow2{
		DestinationPrefix: ipAddressPrefix{
			Prefix: rawSockaddrInet{
				Family: family,
				Data:   [26]byte{},
			},
			PrefixLength: 0,
		},
	}
	returnCode, _, _ := getIPForwardEntry2.Call(uintptr(unsafe.Pointer(defaultRoute)))
	if syscall.Errno(returnCode) == windows.ERROR_NOT_FOUND {
		// Route with this destination is not present, do nothing
		return nil, nil
	} else if returnCode != 0 {
		return nil, fmt.Errorf("windows return code %d", returnCode)
	}
	return defaultRoute, nil
}

// getDefaultRouteInterface returns the interface associated with the default route
func getDefaultRouteInterface(addressFamilies AddressFamilyPreference) (*net.Interface, error) {
	var searchFamilies []uint16
	for _, addrFamily := range addressFamilies {
		if addrFamily == familyIPv4 {
			searchFamilies = append(searchFamilies, syscall.AF_INET)
		} else if addrFamily == familyIPv6 {
			searchFamilies = append(searchFamilies, syscall.AF_INET6)
		}
	}
	for _, family := range searchFamilies {
		route, err := getDefaultRouteEntry(family)
		if err != nil {
			continue
		}
		if route == nil {
			continue
		}
		intf, err := net.InterfaceByIndex(int(route.InterfaceIndex))
		if err != nil {
			continue
		}
		if isInterfaceUp(intf) {
			return intf, nil
		}
	}
	return nil, fmt.Errorf("no default route found")
}

func chooseHostInterface(addressFamilies AddressFamilyPreference) (net.IP, error) {
	var nw networkInterfacer = networkInterface{}
	intf, err := getDefaultRouteInterface(addressFamilies)
	if err != nil {
		// If there are issues discovering the default route's interface, choose any valid interface.
		return chooseIPFromHostInterfaces(nw, addressFamilies)
	}
	klog.V(4).Infof("Found default route's interface %s", intf.Name)
	for _, family := range addressFamilies {
		ip, err := getValidIP(intf, family, nw)
		if err != nil {
			return nil, err
		}
		if ip != nil {
			klog.V(4).Infof("Found active IP %v ", ip)
			return ip, nil
		}
	}
	return nil, fmt.Errorf("unable to find suitable IP")
}
