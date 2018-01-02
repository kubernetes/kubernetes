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
	"encoding/json"
	"net/http"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
)

func (a *App) DeviceAdd(w http.ResponseWriter, r *http.Request) {

	var msg api.DeviceAddRequest
	err := utils.GetJsonFromRequest(r, &msg)
	if err != nil {
		http.Error(w, "request unable to be parsed", 422)
		return
	}

	// Check the message has devices
	if msg.Name == "" {
		http.Error(w, "no devices added", http.StatusBadRequest)
		return
	}

	// Create device entry
	device := NewDeviceEntryFromRequest(&msg)

	// Check the node is in the db
	var node *NodeEntry
	err = a.db.Update(func(tx *bolt.Tx) error {
		var err error
		node, err = NewNodeEntryFromId(tx, msg.NodeId)
		if err == ErrNotFound {
			http.Error(w, "Node id does not exist", http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		// Register device
		err = device.Register(tx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return err
		}

		return nil
	})
	if err != nil {
		return
	}

	// Log the devices are being added
	logger.Info("Adding device %v to node %v", msg.Name, msg.NodeId)

	// Add device in an asynchronous function
	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (seeOtherUrl string, e error) {

		defer func() {
			if e != nil {
				a.db.Update(func(tx *bolt.Tx) error {
					err := device.Deregister(tx)
					if err != nil {
						logger.Err(err)
						return err
					}

					return nil
				})
			}
		}()

		// Setup device on node
		info, err := a.executor.DeviceSetup(node.ManageHostName(),
			device.Info.Name, device.Info.Id)
		if err != nil {
			return "", err
		}

		// Create an entry for the device and set the size
		device.StorageSet(info.Size)
		device.SetExtentSize(info.ExtentSize)

		// Setup garbage collector on error
		defer func() {
			if e != nil {
				a.executor.DeviceTeardown(node.ManageHostName(),
					device.Info.Name,
					device.Info.Id)
			}
		}()

		// Save on db
		err = a.db.Update(func(tx *bolt.Tx) error {

			nodeEntry, err := NewNodeEntryFromId(tx, msg.NodeId)
			if err != nil {
				return err
			}

			// Add device to node
			nodeEntry.DeviceAdd(device.Info.Id)

			clusterEntry, err := NewClusterEntryFromId(tx, nodeEntry.Info.ClusterId)
			if err != nil {
				return err
			}

			// Commit
			err = nodeEntry.Save(tx)
			if err != nil {
				return err
			}

			// Save drive
			err = device.Save(tx)
			if err != nil {
				return err
			}

			// Add to allocator
			err = a.allocator.AddDevice(clusterEntry, nodeEntry, device)
			if err != nil {
				return err
			}

			return nil

		})
		if err != nil {
			return "", err
		}

		logger.Info("Added device %v", msg.Name)

		// Done
		// Returning a null string instructs the async manager
		// to return http status of 204 (No Content)
		return "", nil
	})

}

func (a *App) DeviceInfo(w http.ResponseWriter, r *http.Request) {

	// Get device id from URL
	vars := mux.Vars(r)
	id := vars["id"]

	// Get device information
	var info *api.DeviceInfoResponse
	err := a.db.View(func(tx *bolt.Tx) error {
		entry, err := NewDeviceEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, "Id not found", http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		info, err = entry.NewInfoResponse(tx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		return nil
	})
	if err != nil {
		return
	}

	// Write msg
	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(info); err != nil {
		panic(err)
	}

}

func (a *App) DeviceDelete(w http.ResponseWriter, r *http.Request) {
	// Get the id from the URL
	vars := mux.Vars(r)
	id := vars["id"]

	// Check request
	var (
		device  *DeviceEntry
		node    *NodeEntry
		cluster *ClusterEntry
	)
	err := a.db.View(func(tx *bolt.Tx) error {
		var err error
		// Access device entry
		device, err = NewDeviceEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return logger.Err(err)
		}

		// Check if we can delete the device
		if !device.IsDeleteOk() {
			http.Error(w, device.ConflictString(), http.StatusConflict)
			logger.LogError(device.ConflictString())
			return ErrConflict
		}

		// Access node entry
		node, err = NewNodeEntryFromId(tx, device.NodeId)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return logger.Err(err)
		}

		// Save cluster to update allocator
		cluster, err = NewClusterEntryFromId(tx, node.Info.ClusterId)
		if err != nil {
			return logger.Err(err)
		}

		return nil
	})
	if err != nil {
		return
	}

	// Delete device
	logger.Info("Deleting device %v on node %v", device.Info.Id, device.NodeId)
	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (string, error) {

		// Teardown device
		err := a.executor.DeviceTeardown(node.ManageHostName(),
			device.Info.Name, device.Info.Id)
		if err != nil {
			return "", err
		}

		// Remove device from allocator
		err = a.allocator.RemoveDevice(cluster, node, device)
		if err != nil {
			return "", err
		}

		// Get info from db
		err = a.db.Update(func(tx *bolt.Tx) error {

			// Access node entry
			node, err := NewNodeEntryFromId(tx, device.NodeId)
			if err == ErrNotFound {
				logger.Critical(
					"Node id %v pointed to by device %v, but it is not in the db",
					device.NodeId,
					device.Info.Id)
				return err
			} else if err != nil {
				logger.Err(err)
				return err
			}

			// Delete device from node
			node.DeviceDelete(device.Info.Id)

			// Save node
			node.Save(tx)

			// Delete device from db
			err = device.Delete(tx)
			if err != nil {
				logger.Err(err)
				return err
			}

			// Deregister device
			err = device.Deregister(tx)
			if err != nil {
				logger.Err(err)
				return err
			}

			return nil

		})
		if err != nil {
			return "", err
		}

		// Show that the key has been deleted
		logger.Info("Deleted node [%s]", id)

		return "", nil
	})

}

func (a *App) DeviceSetState(w http.ResponseWriter, r *http.Request) {
	// Get the id from the URL
	vars := mux.Vars(r)
	id := vars["id"]
	var device *DeviceEntry

	// Unmarshal JSON
	var msg api.StateRequest
	err := utils.GetJsonFromRequest(r, &msg)
	if err != nil {
		http.Error(w, "request unable to be parsed", 422)
		return
	}

	// Check for valid id, return immediately if not valid
	err = a.db.View(func(tx *bolt.Tx) error {
		device, err = NewDeviceEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, "Id not found", http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		return nil
	})
	if err != nil {
		return
	}

	// Set state
	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (string, error) {
		err = device.SetState(a.db, a.executor, a.allocator, msg.State)
		if err != nil {
			return "", err
		}
		return "", nil
	})
}
