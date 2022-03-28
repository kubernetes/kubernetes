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

package iptables

import (
	"fmt"

	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	netutils "k8s.io/utils/net"
)

// LocalTrafficDetector in a interface to take action (jump) based on whether traffic originated locally
// at the node or not
type LocalTrafficDetector interface {
	// IsImplemented returns true if the implementation does something, false otherwise
	IsImplemented() bool

	// IfLocal returns iptables arguments that will match traffic from a pod
	IfLocal() []string

	// IfNotLocal returns iptables arguments that will match traffic that is not from a pod
	IfNotLocal() []string
}

type noOpLocalDetector struct{}

// NewNoOpLocalDetector is a no-op implementation of LocalTrafficDetector
func NewNoOpLocalDetector() LocalTrafficDetector {
	return &noOpLocalDetector{}
}

func (n *noOpLocalDetector) IsImplemented() bool {
	return false
}

func (n *noOpLocalDetector) IfLocal() []string {
	return nil // no-op; matches all traffic
}

func (n *noOpLocalDetector) IfNotLocal() []string {
	return nil // no-op; matches all traffic
}

type detectLocalByCIDR struct {
	ifLocal    []string
	ifNotLocal []string
}

// NewDetectLocalByCIDR implements the LocalTrafficDetector interface using a CIDR. This can be used when a single CIDR
// range can be used to capture the notion of local traffic.
func NewDetectLocalByCIDR(cidr string, ipt utiliptables.Interface) (LocalTrafficDetector, error) {
	if netutils.IsIPv6CIDRString(cidr) != ipt.IsIPv6() {
		return nil, fmt.Errorf("CIDR %s has incorrect IP version: expect isIPv6=%t", cidr, ipt.IsIPv6())
	}
	_, _, err := netutils.ParseCIDRSloppy(cidr)
	if err != nil {
		return nil, err
	}
	return &detectLocalByCIDR{
		ifLocal:    []string{"-s", cidr},
		ifNotLocal: []string{"!", "-s", cidr},
	}, nil
}

func (d *detectLocalByCIDR) IsImplemented() bool {
	return true
}

func (d *detectLocalByCIDR) IfLocal() []string {
	return d.ifLocal
}

func (d *detectLocalByCIDR) IfNotLocal() []string {
	return d.ifNotLocal
}

type detectLocalByBridgeInterface struct {
	ifLocal    []string
	ifNotLocal []string
}

// NewDetectLocalByBridgeInterface implements the LocalTrafficDetector interface using a bridge interface name.
// This can be used when a bridge can be used to capture the notion of local traffic from pods.
func NewDetectLocalByBridgeInterface(interfaceName string) (LocalTrafficDetector, error) {
	if len(interfaceName) == 0 {
		return nil, fmt.Errorf("no bridge interface name set")
	}
	return &detectLocalByBridgeInterface{
		ifLocal:    []string{"-i", interfaceName},
		ifNotLocal: []string{"!", "-i", interfaceName},
	}, nil
}

func (d *detectLocalByBridgeInterface) IsImplemented() bool {
	return true
}

func (d *detectLocalByBridgeInterface) IfLocal() []string {
	return d.ifLocal
}

func (d *detectLocalByBridgeInterface) IfNotLocal() []string {
	return d.ifNotLocal
}

type detectLocalByInterfaceNamePrefix struct {
	ifLocal    []string
	ifNotLocal []string
}

// NewDetectLocalByInterfaceNamePrefix implements the LocalTrafficDetector interface using an interface name prefix.
// This can be used when a pod interface name prefix can be used to capture the notion of local traffic. Note
// that this will match on all interfaces that start with the given prefix.
func NewDetectLocalByInterfaceNamePrefix(interfacePrefix string) (LocalTrafficDetector, error) {
	if len(interfacePrefix) == 0 {
		return nil, fmt.Errorf("no interface prefix set")
	}
	return &detectLocalByInterfaceNamePrefix{
		ifLocal:    []string{"-i", interfacePrefix + "+"},
		ifNotLocal: []string{"!", "-i", interfacePrefix + "+"},
	}, nil
}

func (d *detectLocalByInterfaceNamePrefix) IsImplemented() bool {
	return true
}

func (d *detectLocalByInterfaceNamePrefix) IfLocal() []string {
	return d.ifLocal
}

func (d *detectLocalByInterfaceNamePrefix) IfNotLocal() []string {
	return d.ifNotLocal
}
