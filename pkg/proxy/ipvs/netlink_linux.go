// +build linux

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

package ipvs

import (
	"fmt"
	"net"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
)

type netlinkHandle struct {
	netlink.Handle
	isIPv6 bool
}

// NewNetLinkHandle will create a new NetLinkHandle
func NewNetLinkHandle(isIPv6 bool) NetLinkHandle {
	return &netlinkHandle{netlink.Handle{}, isIPv6}
}

// EnsureAddressBind checks if address is bound to the interface and, if not, binds it. If the address is already bound, return true.
func (h *netlinkHandle) EnsureAddressBind(address, devName string) (exist bool, err error) {
	dev, err := h.LinkByName(devName)
	if err != nil {
		return false, fmt.Errorf("error get interface: %s, err: %v", devName, err)
	}
	addr := net.ParseIP(address)
	if addr == nil {
		return false, fmt.Errorf("error parse ip address: %s", address)
	}
	if err := h.AddrAdd(dev, &netlink.Addr{IPNet: netlink.NewIPNet(addr)}); err != nil {
		// "EEXIST" will be returned if the address is already bound to device
		if err == unix.EEXIST {
			return true, nil
		}
		return false, fmt.Errorf("error bind address: %s to interface: %s, err: %v", address, devName, err)
	}
	return false, nil
}

// UnbindAddress makes sure IP address is unbound from the network interface.
func (h *netlinkHandle) UnbindAddress(address, devName string) error {
	dev, err := h.LinkByName(devName)
	if err != nil {
		return fmt.Errorf("error get interface: %s, err: %v", devName, err)
	}
	addr := net.ParseIP(address)
	if addr == nil {
		return fmt.Errorf("error parse ip address: %s", address)
	}
	if err := h.AddrDel(dev, &netlink.Addr{IPNet: netlink.NewIPNet(addr)}); err != nil {
		if err != unix.ENXIO {
			return fmt.Errorf("error unbind address: %s from interface: %s, err: %v", address, devName, err)
		}
	}
	return nil
}

// EnsureDummyDevice is part of interface
func (h *netlinkHandle) EnsureDummyDevice(devName string) (bool, error) {
	_, err := h.LinkByName(devName)
	if err == nil {
		// found dummy device
		return true, nil
	}
	dummy := &netlink.Dummy{
		LinkAttrs: netlink.LinkAttrs{Name: devName},
	}
	return false, h.LinkAdd(dummy)
}

// DeleteDummyDevice is part of interface.
func (h *netlinkHandle) DeleteDummyDevice(devName string) error {
	link, err := h.LinkByName(devName)
	if err != nil {
		_, ok := err.(netlink.LinkNotFoundError)
		if ok {
			return nil
		}
		return fmt.Errorf("error deleting a non-exist dummy device: %s, %v", devName, err)
	}
	dummy, ok := link.(*netlink.Dummy)
	if !ok {
		return fmt.Errorf("expect dummy device, got device type: %s", link.Type())
	}
	return h.LinkDel(dummy)
}

// ListBindAddress will list all IP addresses which are bound in a given interface
func (h *netlinkHandle) ListBindAddress(devName string) ([]string, error) {
	dev, err := h.LinkByName(devName)
	if err != nil {
		return nil, fmt.Errorf("error get interface: %s, err: %v", devName, err)
	}
	addrs, err := h.AddrList(dev, 0)
	if err != nil {
		return nil, fmt.Errorf("error list bound address of interface: %s, err: %v", devName, err)
	}
	var ips []string
	for _, addr := range addrs {
		ips = append(ips, addr.IP.String())
	}
	return ips, nil
}

// GetLocalAddresses lists LOCAL type IP addresses by iterating over network devices and getting their IP addresses.
// If dev is not specified, all LOCAL type IP addresses of host network devices will be returned.
// If dev is specified, only LOCAL type IP addresses of dev will be returned.
// If filterDev is specified, the result will discard IP addresses of filterDev.
func (h *netlinkHandle) GetLocalAddresses(dev, filterDev string) (sets.String, error) {
	var ifaces []net.Interface
	var err error
	var filterIfaceIndex = -1

	if dev != "" {
		// get specified interface by name
		iface, err := net.InterfaceByName(dev)
		if err != nil {
			return nil, fmt.Errorf("error get device %s, err: %v", dev, err)
		}
		ifaces = []net.Interface{*iface}
	} else {
		// list all interfaces
		ifaces, err = net.Interfaces()
		if err != nil {
			return nil, fmt.Errorf("error list all interfaces: %v", err)
		}
	}

	if filterDev != "" {
		iface, err := net.InterfaceByName(filterDev)
		if err != nil {
			return nil, fmt.Errorf("error get filter device %s, err: %v", filterDev, err)
		}
		filterIfaceIndex = iface.Index
	}

	res := sets.NewString()

	// iterate over each interface
	for _, iface := range ifaces {
		// skip filterDev
		if iface.Index == filterIfaceIndex {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			return nil, fmt.Errorf("error get IP addresses of interface %s: %v", iface.Name, err)
		}
		// iterate over addresses on each interface
		for _, addr := range addrs {
			var ip net.IP
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip == nil || ip.IsLinkLocalUnicast() {
				continue
			}
			if h.isIPv6 {
				if ip.To4() == nil {
					// insert IPv6 addresses
					res.Insert(ip.String())
				}
			} else {
				if ip.To4() != nil {
					// insert IPv4 addresses
					res.Insert(ip.String())
				}
			}
		}
	}

	return res, nil
}

var _ NetLinkHandle = &netlinkHandle{}
