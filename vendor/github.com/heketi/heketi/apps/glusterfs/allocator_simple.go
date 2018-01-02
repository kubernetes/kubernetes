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
	"github.com/boltdb/bolt"
	"sync"
)

// Simple allocator contains a map to rings of clusters
type SimpleAllocator struct {
	rings map[string]*SimpleAllocatorRing
	lock  sync.Mutex
}

// Create a new simple allocator
func NewSimpleAllocator() *SimpleAllocator {
	s := &SimpleAllocator{}
	s.rings = make(map[string]*SimpleAllocatorRing)
	return s
}

// Create a new simple allocator and initialize it with data from the db
func NewSimpleAllocatorFromDb(db *bolt.DB) *SimpleAllocator {

	s := NewSimpleAllocator()

	err := db.View(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}

		for _, clusterId := range clusters {
			cluster, err := NewClusterEntryFromId(tx, clusterId)
			if err != nil {
				return err
			}

			for _, nodeId := range cluster.Info.Nodes {
				node, err := NewNodeEntryFromId(tx, nodeId)
				if err != nil {
					return err
				}

				// Check node is online
				if !node.isOnline() {
					continue
				}

				for _, deviceId := range node.Devices {
					device, err := NewDeviceEntryFromId(tx, deviceId)
					if err != nil {
						return err
					}

					// Check device is online
					if !device.isOnline() {
						continue
					}

					// Add device to ring
					err = s.AddDevice(cluster, node, device)
					if err != nil {
						return err
					}

				}
			}
		}
		return nil
	})
	if err != nil {
		return nil
	}

	return s

}

func (s *SimpleAllocator) AddDevice(cluster *ClusterEntry,
	node *NodeEntry,
	device *DeviceEntry) error {

	s.lock.Lock()
	defer s.lock.Unlock()

	// Create a new cluster id if one is not available
	clusterId := cluster.Info.Id
	if _, ok := s.rings[clusterId]; !ok {
		s.rings[clusterId] = NewSimpleAllocatorRing()
	}

	s.rings[clusterId].Add(&SimpleDevice{
		zone:     node.Info.Zone,
		nodeId:   node.Info.Id,
		deviceId: device.Info.Id,
	})

	return nil

}

func (s *SimpleAllocator) RemoveDevice(cluster *ClusterEntry,
	node *NodeEntry,
	device *DeviceEntry) error {

	s.lock.Lock()
	defer s.lock.Unlock()

	// Check the cluster id is in the map
	clusterId := cluster.Info.Id
	if _, ok := s.rings[clusterId]; !ok {
		logger.LogError("Unknown cluster id requested: %v", clusterId)
		return ErrNotFound
	}

	// Remove device from ring
	s.rings[clusterId].Remove(&SimpleDevice{
		zone:     node.Info.Zone,
		nodeId:   node.Info.Id,
		deviceId: device.Info.Id,
	})

	return nil
}

func (s *SimpleAllocator) RemoveCluster(clusterId string) error {

	s.lock.Lock()
	defer s.lock.Unlock()

	// Check the cluster id is in the map
	if _, ok := s.rings[clusterId]; !ok {
		logger.LogError("Unknown cluster id requested: %v", clusterId)
		return ErrNotFound
	}

	// Remove cluster from map
	delete(s.rings, clusterId)

	return nil
}

func (s *SimpleAllocator) getDeviceList(clusterId, brickId string) (SimpleDevices, error) {
	s.lock.Lock()
	defer s.lock.Unlock()

	if _, ok := s.rings[clusterId]; !ok {
		logger.LogError("Unknown cluster id requested: %v", clusterId)
		return nil, ErrNotFound
	}

	ring := s.rings[clusterId]
	ring.Rebalance()
	devicelist := ring.GetDeviceList(brickId)

	return devicelist, nil

}

func (s *SimpleAllocator) GetNodes(clusterId, brickId string) (<-chan string,
	chan<- struct{}, <-chan error) {

	// Initialize channels
	device, done := make(chan string), make(chan struct{})

	// Make sure to make a buffered channel for the error, so we can
	// set it and return
	errc := make(chan error, 1)

	// Get the list of devices for this brick id
	devicelist, err := s.getDeviceList(clusterId, brickId)

	if err != nil {
		errc <- err
		close(device)
		return device, done, errc
	}

	// Start generator in a new goroutine
	go func() {
		defer func() {
			errc <- nil
			close(device)
		}()

		for _, d := range devicelist {
			select {
			case device <- d.deviceId:
			case <-done:
				return
			}
		}

	}()

	return device, done, errc
}
