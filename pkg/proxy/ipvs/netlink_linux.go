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
}

// NewNetLinkHandle will crate a new NetLinkHandle
func NewNetLinkHandle() NetLinkHandle {
	return &netlinkHandle{netlink.Handle{}}
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

// GetLocalAddresses lists all LOCAL type IP addresses from host based on filter device.
// If dev is not specified, it's equivalent to exec:
// $ ip route show table local type local proto kernel
// 10.0.0.1 dev kube-ipvs0  scope host  src 10.0.0.1
// 10.0.0.10 dev kube-ipvs0  scope host  src 10.0.0.10
// 10.0.0.252 dev kube-ipvs0  scope host  src 10.0.0.252
// 100.106.89.164 dev eth0  scope host  src 100.106.89.164
// 127.0.0.0/8 dev lo  scope host  src 127.0.0.1
// 127.0.0.1 dev lo  scope host  src 127.0.0.1
// 172.17.0.1 dev docker0  scope host  src 172.17.0.1
// 192.168.122.1 dev virbr0  scope host  src 192.168.122.1
// Then cut the unique src IP fields,
// --> result set: [10.0.0.1, 10.0.0.10, 10.0.0.252, 100.106.89.164, 127.0.0.1, 192.168.122.1]

// If dev is specified, it's equivalent to exec:
// $ ip route show table local type local proto kernel dev kube-ipvs0
// 10.0.0.1  scope host  src 10.0.0.1
// 10.0.0.10  scope host  src 10.0.0.10
// Then cut the unique src IP fields,
// --> result set: [10.0.0.1, 10.0.0.10]

// If filterDev is specified, the result will discard route of specified device and cut src from other routes.
func (h *netlinkHandle) GetLocalAddresses(dev, filterDev string) (sets.String, error) {
	chosenLinkIndex, filterLinkIndex := -1, -1
	if dev != "" {
		link, err := h.LinkByName(dev)
		if err != nil {
			return nil, fmt.Errorf("error get device %s, err: %v", filterDev, err)
		}
		chosenLinkIndex = link.Attrs().Index
	} else if filterDev != "" {
		link, err := h.LinkByName(filterDev)
		if err != nil {
			return nil, fmt.Errorf("error get filter device %s, err: %v", filterDev, err)
		}
		filterLinkIndex = link.Attrs().Index
	}

	routeFilter := &netlink.Route{
		Table:    unix.RT_TABLE_LOCAL,
		Type:     unix.RTN_LOCAL,
		Protocol: unix.RTPROT_KERNEL,
	}
	filterMask := netlink.RT_FILTER_TABLE | netlink.RT_FILTER_TYPE | netlink.RT_FILTER_PROTOCOL

	// find chosen device
	if chosenLinkIndex != -1 {
		routeFilter.LinkIndex = chosenLinkIndex
		filterMask |= netlink.RT_FILTER_OIF
	}
	routes, err := h.RouteListFiltered(netlink.FAMILY_ALL, routeFilter, filterMask)
	if err != nil {
		return nil, fmt.Errorf("error list route table, err: %v", err)
	}
	res := sets.NewString()
	for _, route := range routes {
		if route.LinkIndex == filterLinkIndex {
			continue
		}
		if route.Src != nil {
			res.Insert(route.Src.String())
		}
	}
	return res, nil
}
