//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"fmt"
	"strconv"
)

// Elements in the balanced list
type SimpleDevice struct {
	zone             int
	nodeId, deviceId string
}

// Pretty pring a SimpleDevice
func (s *SimpleDevice) String() string {
	return fmt.Sprintf("{Z:%v N:%v D:%v}",
		s.zone,
		s.nodeId,
		s.deviceId)
}

// Simple Devices so that we have no pointers and no race conditions
type SimpleDevices []SimpleDevice

// A node is a collection of devices
type SimpleNode []*SimpleDevice

// A zone is a collection of nodes
type SimpleZone []SimpleNode

// The allocation ring will contain a map composed of all
// the devices available in the cluster.  Call Rebalance()
// for it to create a balanced list.
type SimpleAllocatorRing struct {

	// Map [zone] to [node] to slice of SimpleDevices
	ring         map[int]map[string][]*SimpleDevice
	balancedList SimpleDevices
}

// Create a new simple ring
func NewSimpleAllocatorRing() *SimpleAllocatorRing {
	s := &SimpleAllocatorRing{}
	s.ring = make(map[int]map[string][]*SimpleDevice)

	return s
}

// Convert the ring map into a consumable list of lists.
// This allows the rebalancer to go through the lists and remove
// elements as it balances
func (s *SimpleAllocatorRing) createZoneLists() []SimpleZone {
	zones := make([]SimpleZone, 0)

	for _, n := range s.ring {

		zone := make([]SimpleNode, 0)
		for _, d := range n {
			zone = append(zone, d)
		}
		zones = append(zones, zone)
	}

	return zones
}

// Add a device to the ring map
func (s *SimpleAllocatorRing) Add(d *SimpleDevice) {

	if nodes, ok := s.ring[d.zone]; ok {
		if _, ok := nodes[d.nodeId]; ok {
			nodes[d.nodeId] = append(nodes[d.nodeId], d)
		} else {
			nodes[d.nodeId] = []*SimpleDevice{d}
		}
	} else {
		s.ring[d.zone] = make(map[string][]*SimpleDevice)
		s.ring[d.zone][d.nodeId] = []*SimpleDevice{d}
	}

	s.balancedList = nil
}

// Remove device from the ring map
func (s *SimpleAllocatorRing) Remove(d *SimpleDevice) {

	if nodes, ok := s.ring[d.zone]; ok {
		if devices, ok := nodes[d.nodeId]; ok {
			for index, device := range devices {
				if device.deviceId == d.deviceId {
					// Found device, now delete it from the ring map
					nodes[d.nodeId] = append(nodes[d.nodeId][:index], nodes[d.nodeId][index+1:]...)

					if len(nodes[d.nodeId]) == 0 {
						delete(nodes, d.nodeId)
					}
					if len(s.ring[d.zone]) == 0 {
						delete(s.ring, d.zone)
					}
				}
			}
		}
	}

	s.balancedList = nil
}

// Rebalance the ring and place the rebalanced list
// into balancedList.
// The idea is to setup an array/slice where each continguous SimpleDevice
// is from either a different zone, or node.
func (s *SimpleAllocatorRing) Rebalance() {

	if s.balancedList != nil {
		return
	}

	// Copy map data to slices
	zones := s.createZoneLists()

	// Create a list
	list := make(SimpleDevices, 0)

	// Populate the list
	var device *SimpleDevice
	for i := 0; len(zones) != 0; i++ {
		zone := i % len(zones)
		nodes := zones[zone]
		node := i % len(nodes)
		devices := nodes[node]

		// pop device
		device = devices[len(devices)-1]
		devices = devices[:len(devices)-1]
		nodes[node] = devices

		list = append(list, *device)

		if len(devices) == 0 {
			// delete node
			nodes = append(nodes[:node], nodes[node+1:]...)
			zones[zone] = nodes
		}

		if len(nodes) == 0 {
			// delete zone
			zones = append(zones[:zone], zones[zone+1:]...)
		}
	}

	s.balancedList = list
}

// Use a uuid to point at a position in the ring.  Return a list of devices
// from that point in the ring.
func (s *SimpleAllocatorRing) GetDeviceList(uuid string) SimpleDevices {

	if s.balancedList == nil {
		s.Rebalance()
	}
	if len(s.balancedList) == 0 {
		return SimpleDevices{}
	}

	// Create a new list to avoid race conditions
	devices := make(SimpleDevices, len(s.balancedList))
	copy(devices, s.balancedList)

	// Instead of using 8 characters to convert to a int32, use 7 which avoids
	// negative numbers
	index64, err := strconv.ParseInt(uuid[:7], 16, 32)
	if err != nil {
		logger.Err(err)
		return devices
	}

	// Point to a position on the ring
	index := int(index64) % len(s.balancedList)

	// Return a list according to the position in the list
	return append(devices[index:], devices[:index]...)

}
