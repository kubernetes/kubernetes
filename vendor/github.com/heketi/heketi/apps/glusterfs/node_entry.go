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
	"bytes"
	"encoding/gob"
	"fmt"
	"sort"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/lpabon/godbc"
)

type NodeEntry struct {
	Entry

	Info    api.NodeInfo
	Devices sort.StringSlice
}

func NewNodeEntry() *NodeEntry {
	entry := &NodeEntry{}
	entry.Devices = make(sort.StringSlice, 0)
	entry.SetOnline()

	return entry
}

func NewNodeEntryFromRequest(req *api.NodeAddRequest) *NodeEntry {
	godbc.Require(req != nil)

	node := NewNodeEntry()
	node.Info.Id = utils.GenUUID()
	node.Info.ClusterId = req.ClusterId
	node.Info.Hostnames = req.Hostnames
	node.Info.Zone = req.Zone

	return node
}

func NewNodeEntryFromId(tx *bolt.Tx, id string) (*NodeEntry, error) {
	godbc.Require(tx != nil)

	entry := NewNodeEntry()
	err := EntryLoad(tx, entry, id)
	if err != nil {
		return nil, err
	}

	return entry, nil
}

func (n *NodeEntry) registerManageKey(host string) string {
	return "MANAGE" + host
}

func (n *NodeEntry) registerStorageKey(host string) string {
	return "STORAGE" + host
}

// Verify gluster process in the node and return the manage hostname of a node in the cluster
func GetVerifiedManageHostname(db *bolt.DB, e executors.Executor, clusterId string) (string, error) {
	godbc.Require(clusterId != "")
	var cluster *ClusterEntry
	var node *NodeEntry
	var err error
	err = db.View(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, clusterId)
		return err
	})

	if err != nil {
		return "", err
	}

	for _, n := range cluster.Info.Nodes {
		var newNode *NodeEntry
		err = db.View(func(tx *bolt.Tx) error {
			var err error
			newNode, err = NewNodeEntryFromId(tx, n)
			return err
		})

		if err != nil {
			//pass on to next node
			continue
		}

		// Ignore if the node is not online
		if !newNode.isOnline() {
			continue
		}
		err = e.GlusterdCheck(newNode.ManageHostName())
		if err != nil {
			logger.Info("Glusterd not running in %v", newNode.ManageHostName())
			continue
		}
		node = newNode
		break
	}
	if node != nil {
		return node.ManageHostName(), nil
	}
	return "", ErrNotFound
}

func (n *NodeEntry) Register(tx *bolt.Tx) error {

	// Save manage hostnames
	for _, h := range n.Info.Hostnames.Manage {
		val, err := EntryRegister(tx, n, n.registerManageKey(h), []byte(n.Info.Id))
		if err == ErrKeyExists {
			// Now check if the node actually exists.  This only happens
			// when the application crashes and it doesn't clean up stale
			// registrations.
			conflictId := string(val)
			_, err := NewNodeEntryFromId(tx, conflictId)
			if err == ErrNotFound {
				// (stale) There is actually no conflict, we can allow
				// the registration
				return nil
			} else if err != nil {
				return logger.Err(err)
			}

			// Return that we found a conflict
			return fmt.Errorf("Hostname %v already used by node with id %v\n",
				h, conflictId)
		} else if err != nil {
			return err
		}
	}

	// Save storage hostnames
	for _, h := range n.Info.Hostnames.Storage {
		val, err := EntryRegister(tx, n, n.registerStorageKey(h), []byte(n.Info.Id))
		if err == ErrKeyExists {

			// Check if it exists
			conflictId := string(val)
			_, err := NewNodeEntryFromId(tx, conflictId)
			if err == ErrNotFound {
				// (stale) There is actually no conflict, we can allow
				// the registration
				return nil
			} else if err != nil {
				return logger.Err(err)
			}

			// Return that we found a conflict
			return fmt.Errorf("Hostname %v already used by node with id %v\n",
				h, conflictId)
		} else if err != nil {
			return err
		}
	}

	return nil

}

func (n *NodeEntry) Deregister(tx *bolt.Tx) error {

	// Remove manage hostnames from Db
	for _, h := range n.Info.Hostnames.Manage {
		err := EntryDelete(tx, n, n.registerManageKey(h))
		if err != nil {
			return err
		}
	}

	// Remove storage hostnames
	for _, h := range n.Info.Hostnames.Storage {
		err := EntryDelete(tx, n, n.registerStorageKey(h))
		if err != nil {
			return err
		}
	}

	return nil
}

func (n *NodeEntry) BucketName() string {
	return BOLTDB_BUCKET_NODE
}

func (n *NodeEntry) Save(tx *bolt.Tx) error {
	godbc.Require(tx != nil)
	godbc.Require(len(n.Info.Id) > 0)

	return EntrySave(tx, n, n.Info.Id)

}

func (n *NodeEntry) ManageHostName() string {
	godbc.Require(n.Info.Hostnames.Manage != nil)
	godbc.Require(len(n.Info.Hostnames.Manage) > 0)

	return n.Info.Hostnames.Manage[0]
}

func (n *NodeEntry) StorageHostName() string {
	godbc.Require(n.Info.Hostnames.Storage != nil)
	godbc.Require(len(n.Info.Hostnames.Storage) > 0)

	return n.Info.Hostnames.Storage[0]
}

func (n *NodeEntry) IsDeleteOk() bool {
	// Check if the nodes still has drives
	if len(n.Devices) > 0 {
		return false
	}
	return true
}

func (n *NodeEntry) ConflictString() string {
	return fmt.Sprintf("Unable to delete node [%v] because it contains devices", n.Info.Id)
}

func (n *NodeEntry) Delete(tx *bolt.Tx) error {
	godbc.Require(tx != nil)

	// Check if the nodes still has drives
	if !n.IsDeleteOk() {
		logger.Warning(n.ConflictString())
		return ErrConflict
	}

	return EntryDelete(tx, n, n.Info.Id)
}

func (n *NodeEntry) removeAllDisksFromRing(tx *bolt.Tx,
	a Allocator) error {

	cluster, err := NewClusterEntryFromId(tx, n.Info.ClusterId)
	if err != nil {
		return err
	}

	for _, deviceId := range n.Devices {
		device, err := NewDeviceEntryFromId(tx, deviceId)
		if err != nil {
			return err
		}

		// Remove device
		err = a.RemoveDevice(cluster, n, device)
		if err != nil {
			return err
		}
	}

	return nil
}

func (n *NodeEntry) addAllDisksToRing(tx *bolt.Tx,
	a Allocator) error {

	cluster, err := NewClusterEntryFromId(tx, n.Info.ClusterId)
	if err != nil {
		return err
	}

	// Add all devices
	for _, deviceId := range n.Devices {
		device, err := NewDeviceEntryFromId(tx, deviceId)
		if err != nil {
			return err
		}

		// Add device
		if device.isOnline() {
			err = a.AddDevice(cluster, n, device)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (n *NodeEntry) SetState(db *bolt.DB, e executors.Executor,
	a Allocator,
	s api.EntryState) error {

	// Check current state
	switch n.State {

	// Node is in removed/failed state
	case api.EntryStateFailed:
		switch s {
		case api.EntryStateFailed:
			return nil
		case api.EntryStateOnline:
			return fmt.Errorf("Cannot move a failed/removed node to online state")
		case api.EntryStateOffline:
			return fmt.Errorf("Cannot move a failed/removed node to offline state")
		default:
			return fmt.Errorf("Unknown state type: %v", s)
		}

	// Node is in enabled/online state
	case api.EntryStateOnline:
		switch s {
		case api.EntryStateOnline:
			return nil
		case api.EntryStateOffline:
			// Remove all disks from Ring
			err := db.Update(func(tx *bolt.Tx) error {
				err := n.removeAllDisksFromRing(tx, a)
				if err != nil {
					return err
				}

				// Save state
				n.State = s
				// Save new state
				err = n.Save(tx)
				if err != nil {
					return err
				}
				return nil
			})
			if err != nil {
				return err
			}
		case api.EntryStateFailed:
			return fmt.Errorf("Node must be offline before remove operation is performed, node:%v", n.Info.Id)
		default:
			return fmt.Errorf("Unknown state type: %v", s)
		}

	// Node is in disabled/offline state
	case api.EntryStateOffline:
		switch s {
		case api.EntryStateOffline:
			return nil
		case api.EntryStateOnline:
			// Add all disks back
			err := db.Update(func(tx *bolt.Tx) error {
				err := n.addAllDisksToRing(tx, a)
				if err != nil {
					return err
				}
				n.State = s
				err = n.Save(tx)
				if err != nil {
					return err
				}
				return nil
			})
			if err != nil {
				return err
			}
		case api.EntryStateFailed:
			for _, id := range n.Devices {
				var d *DeviceEntry
				err := db.View(func(tx *bolt.Tx) error {
					var err error
					d, err = NewDeviceEntryFromId(tx, id)
					if err != nil {
						return err
					}
					return nil
				})
				err = d.Remove(db, e, a)
				if err != nil {
					if err == ErrNoReplacement {
						return logger.LogError("Unable to remove node [%v] as no device was found to replace device [%v]", n.Info.Id, d.Id())
					}
					return err
				}
			}

			// Make the state change to failed
			err := db.Update(func(tx *bolt.Tx) error {
				n.State = s
				err := n.Save(tx)
				if err != nil {
					return err
				}
				return nil
			})
			if err != nil {
				return err
			}

		default:
			return fmt.Errorf("Unknown state type: %v", s)
		}
	}
	return nil
}

func (n *NodeEntry) NewInfoReponse(tx *bolt.Tx) (*api.NodeInfoResponse, error) {

	godbc.Require(tx != nil)

	info := &api.NodeInfoResponse{}
	info.ClusterId = n.Info.ClusterId
	info.Hostnames = n.Info.Hostnames
	info.Id = n.Info.Id
	info.Zone = n.Info.Zone
	info.State = n.State
	info.DevicesInfo = make([]api.DeviceInfoResponse, 0)

	// Add each drive information
	for _, deviceid := range n.Devices {
		device, err := NewDeviceEntryFromId(tx, deviceid)
		if err != nil {
			return nil, err
		}

		driveinfo, err := device.NewInfoResponse(tx)
		if err != nil {
			return nil, err
		}
		info.DevicesInfo = append(info.DevicesInfo, *driveinfo)
	}

	return info, nil
}

func (n *NodeEntry) Marshal() ([]byte, error) {
	var buffer bytes.Buffer
	enc := gob.NewEncoder(&buffer)
	err := enc.Encode(*n)

	return buffer.Bytes(), err
}

func (n *NodeEntry) Unmarshal(buffer []byte) error {
	dec := gob.NewDecoder(bytes.NewReader(buffer))
	err := dec.Decode(n)
	if err != nil {
		return err
	}

	// Make sure to setup arrays if nil
	if n.Devices == nil {
		n.Devices = make(sort.StringSlice, 0)
	}

	return nil
}

func (n *NodeEntry) DeviceAdd(id string) {
	godbc.Require(!utils.SortedStringHas(n.Devices, id))

	n.Devices = append(n.Devices, id)
	n.Devices.Sort()
}

func (n *NodeEntry) DeviceDelete(id string) {
	n.Devices = utils.SortedStringsDelete(n.Devices, id)
}

func NodeEntryUpgrade(tx *bolt.Tx) error {
	return nil
}
