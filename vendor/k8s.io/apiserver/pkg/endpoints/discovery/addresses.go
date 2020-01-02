/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"net"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type Addresses interface {
	ServerAddressByClientCIDRs(net.IP) []metav1.ServerAddressByClientCIDR
}

// DefaultAddresses is a default implementation of Addresses that will work in most cases
type DefaultAddresses struct {
	// CIDRRules is a list of CIDRs and Addresses to use if a client is in the range
	CIDRRules []CIDRRule

	// DefaultAddress is the address (hostname or IP and port) that should be used in
	// if no CIDR matches more specifically.
	DefaultAddress string
}

// CIDRRule is a rule for adding an alternate path to the master based on matching CIDR
type CIDRRule struct {
	IPRange net.IPNet

	// Address is the address (hostname or IP and port) that should be used in
	// if this CIDR matches
	Address string
}

func (d DefaultAddresses) ServerAddressByClientCIDRs(clientIP net.IP) []metav1.ServerAddressByClientCIDR {
	addressCIDRMap := []metav1.ServerAddressByClientCIDR{
		{
			ClientCIDR:    "0.0.0.0/0",
			ServerAddress: d.DefaultAddress,
		},
	}

	for _, rule := range d.CIDRRules {
		addressCIDRMap = append(addressCIDRMap, rule.ServerAddressByClientCIDRs(clientIP)...)
	}
	return addressCIDRMap
}

func (d CIDRRule) ServerAddressByClientCIDRs(clientIP net.IP) []metav1.ServerAddressByClientCIDR {
	addressCIDRMap := []metav1.ServerAddressByClientCIDR{}

	if d.IPRange.Contains(clientIP) {
		addressCIDRMap = append(addressCIDRMap, metav1.ServerAddressByClientCIDR{
			ClientCIDR:    d.IPRange.String(),
			ServerAddress: d.Address,
		})
	}
	return addressCIDRMap
}
