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
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
	"reflect"
	"testing"
)

func TestNewSimpleAllocatorRing(t *testing.T) {
	r := NewSimpleAllocatorRing()
	tests.Assert(t, r != nil)
	tests.Assert(t, r.ring != nil)
}

func TestSimpleAllocatorRingAddRemove(t *testing.T) {
	r := NewSimpleAllocatorRing()
	tests.Assert(t, r != nil)

	// Add one device
	sd := &SimpleDevice{
		zone:     100,
		nodeId:   "abc",
		deviceId: "def",
	}

	r.Add(sd)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)

	// Add new zone
	sd2 := &SimpleDevice{
		zone:     10,
		nodeId:   "ace",
		deviceId: "dea",
	}
	r.Add(sd2)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, r.ring[sd2.zone][sd2.nodeId][0] == sd2)
	tests.Assert(t, r.balancedList == nil)

	// Add new node to same zone
	sd3 := &SimpleDevice{
		zone:     10,
		nodeId:   "aed",
		deviceId: "daa",
	}
	r.Add(sd3)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, r.ring[sd2.zone][sd2.nodeId][0] == sd2)
	tests.Assert(t, r.ring[sd2.zone][sd3.nodeId][0] == sd3)
	tests.Assert(t, r.balancedList == nil)

	// Add new disk to same node
	sd4 := &SimpleDevice{
		zone:     10,
		nodeId:   "aed",
		deviceId: "dbb",
	}
	r.Add(sd4)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, r.ring[sd2.zone][sd2.nodeId][0] == sd2)
	tests.Assert(t, r.ring[sd2.zone][sd3.nodeId][0] == sd3)
	tests.Assert(t, r.ring[sd2.zone][sd3.nodeId][1] == sd4)
	tests.Assert(t, len(r.ring[sd2.zone][sd3.nodeId]) == 2)
	tests.Assert(t, r.balancedList == nil)

	// Remove sd4
	r.Remove(sd4)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, r.ring[sd2.zone][sd2.nodeId][0] == sd2)
	tests.Assert(t, r.ring[sd2.zone][sd3.nodeId][0] == sd3)
	tests.Assert(t, len(r.ring[sd2.zone][sd3.nodeId]) == 1)
	tests.Assert(t, len(r.ring[sd2.zone]) == 2)
	tests.Assert(t, r.balancedList == nil)

	// Remove sd3
	r.Remove(sd3)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, r.ring[sd2.zone][sd2.nodeId][0] == sd2)
	tests.Assert(t, len(r.ring[sd2.zone]) == 1)
	tests.Assert(t, len(r.ring) == 2)
	tests.Assert(t, r.balancedList == nil)

	// Remove sd2
	r.Remove(sd2)
	tests.Assert(t, r.ring[sd.zone][sd.nodeId][0] == sd)
	tests.Assert(t, len(r.ring) == 1)
	tests.Assert(t, r.balancedList == nil)

	// Remove sd
	r.Remove(sd)
	tests.Assert(t, len(r.ring) == 0)
	tests.Assert(t, r.balancedList == nil)

}

func TestSimpleAllocatorCreateZoneLists(t *testing.T) {
	r := NewSimpleAllocatorRing()
	tests.Assert(t, r != nil)

	// Add one device
	sd := &SimpleDevice{
		zone:     100,
		nodeId:   "abc",
		deviceId: "def",
	}

	// Add new zone
	sd2 := &SimpleDevice{
		zone:     10,
		nodeId:   "ace",
		deviceId: "dea",
	}

	// Add new node to same zone
	sd3 := &SimpleDevice{
		zone:     10,
		nodeId:   "aed",
		deviceId: "daa",
	}

	// Add new disk to same node
	sd4 := &SimpleDevice{
		zone:     10,
		nodeId:   "aed",
		deviceId: "dbb",
	}

	r.Add(sd)
	r.Add(sd2)
	r.Add(sd3)
	r.Add(sd4)

	// Get a
	zones := r.createZoneLists()
	tests.Assert(t, zones != nil)
	tests.Assert(t, len(zones) == 2) // two zones
	tests.Assert(t, len(zones[0]) == 1 || len(zones[0]) == 2)

	// Check the two zones
	for i := 0; i < 2; i++ {
		if len(zones[i]) == 1 {
			tests.Assert(t, zones[i][0][0] == sd)
		} else {
			tests.Assert(t, len(zones[i]) == 2)

			// Check the two nodes
			tests.Assert(t, len(zones[i][0]) == 1 || len(zones[i][0]) == 2)
			for j := 0; j < 2; j++ {
				if len(zones[i][j]) == 1 {
					tests.Assert(t, zones[i][j][0] == sd2)
				} else {
					tests.Assert(t, zones[i][j][0] == sd3 || zones[i][j][0] == sd4)
				}
			}
		}
	}
}

func TestSimpleAllocatorRingRebalance(t *testing.T) {
	r := NewSimpleAllocatorRing()
	tests.Assert(t, r != nil)

	zones, nodes, drives := 10, 100, 48

	// Add 10*100*48 devices to the ring
	for z := 0; z < zones; z++ {

		// Generate nodes for this zone
		for n := 0; n < nodes; n++ {
			nid := utils.GenUUID()

			// Generate drives for this node
			for d := 0; d < drives; d++ {
				did := utils.GenUUID()

				// Setup simple device
				dev := &SimpleDevice{
					zone:     z,
					deviceId: did,
					nodeId:   nid,
				}
				r.Add(dev)
			}
		}
	}
	tests.Assert(t, r.balancedList == nil)

	// Rebalance
	r.Rebalance()
	tests.Assert(t, r.balancedList != nil)
	tests.Assert(t, len(r.balancedList) == zones*nodes*drives)

	// Check balance
	// 1. No zones should be next to eachother in the list
	// 2. Every other element should not have the same node
	for i := range r.balancedList[:len(r.balancedList)-1] {
		tests.Assert(t, r.balancedList[i].zone != r.balancedList[i+1].zone)
	}
	for i := range r.balancedList[:len(r.balancedList)-2] {
		tests.Assert(t, r.balancedList[i].nodeId != r.balancedList[i+2].nodeId)
	}
}

func TestSimpleAllocatorGetDeviceList(t *testing.T) {
	r := NewSimpleAllocatorRing()
	tests.Assert(t, r != nil)

	zones, nodes, drives := 1, 2, 4

	// Create ring
	for z := 0; z < zones; z++ {

		// Generate nodes for this zone
		for n := 0; n < nodes; n++ {
			nid := utils.GenUUID()

			// Generate drives for this node
			for d := 0; d < drives; d++ {
				did := utils.GenUUID()

				// Setup simple device
				dev := &SimpleDevice{
					zone:     z,
					deviceId: did,
					nodeId:   nid,
				}
				r.Add(dev)
			}
		}
	}
	tests.Assert(t, r.balancedList == nil)

	// Rebalance
	r.Rebalance()
	tests.Assert(t, r.balancedList != nil)
	tests.Assert(t, len(r.balancedList) == zones*nodes*drives)

	// Get a list for a brick with "00000" id
	// It should return a list equal to balancedList
	tests.Assert(t,
		reflect.DeepEqual(r.GetDeviceList("0000000"), r.balancedList))
	tests.Assert(t,
		reflect.DeepEqual(r.GetDeviceList("0000001"), append(r.balancedList[1:], r.balancedList[0])))

	// 14 is larger than 1*2*4, 8.. So the index is 14%8 = 6
	tests.Assert(t,
		reflect.DeepEqual(r.GetDeviceList("000000e"), append(r.balancedList[6:], r.balancedList[:6]...)))
}
