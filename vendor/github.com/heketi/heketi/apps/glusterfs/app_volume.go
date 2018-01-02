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
	"fmt"
	"math"
	"net/http"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	"github.com/heketi/heketi/pkg/db"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
)

const (
	VOLUME_CREATE_MAX_SNAPSHOT_FACTOR = 100
)

func (a *App) VolumeCreate(w http.ResponseWriter, r *http.Request) {

	var msg api.VolumeCreateRequest
	err := utils.GetJsonFromRequest(r, &msg)
	if err != nil {
		http.Error(w, "request unable to be parsed", 422)
		return
	}

	switch {
	case msg.Gid < 0:
		http.Error(w, "Bad group id less than zero", http.StatusBadRequest)
		logger.LogError("Bad group id less than zero")
		return
	case msg.Gid >= math.MaxInt32:
		http.Error(w, "Bad group id equal or greater than 2**32", http.StatusBadRequest)
		logger.LogError("Bad group id equal or greater than 2**32")
		return
	}

	switch msg.Durability.Type {
	case api.DurabilityEC:
	case api.DurabilityReplicate:
	case api.DurabilityDistributeOnly:
	case "":
		msg.Durability.Type = api.DurabilityDistributeOnly
	default:
		http.Error(w, "Unknown durability type", http.StatusBadRequest)
		logger.LogError("Unknown durability type")
		return
	}

	if msg.Size < 1 {
		http.Error(w, "Invalid volume size", http.StatusBadRequest)
		logger.LogError("Invalid volume size")
		return
	}
	if msg.Snapshot.Enable {
		if msg.Snapshot.Factor < 1 || msg.Snapshot.Factor > VOLUME_CREATE_MAX_SNAPSHOT_FACTOR {
			http.Error(w, "Invalid snapshot factor", http.StatusBadRequest)
			logger.LogError("Invalid snapshot factor")
			return
		}
	}

	if msg.Durability.Type == api.DurabilityReplicate {
		if msg.Durability.Replicate.Replica > 3 {
			http.Error(w, "Invalid replica value", http.StatusBadRequest)
			logger.LogError("Invalid replica value")
			return
		}
	}

	if msg.Durability.Type == api.DurabilityEC {
		d := msg.Durability.Disperse
		// Place here correct combinations
		switch {
		case d.Data == 2 && d.Redundancy == 1:
		case d.Data == 4 && d.Redundancy == 2:
		case d.Data == 8 && d.Redundancy == 3:
		case d.Data == 8 && d.Redundancy == 4:
		default:
			http.Error(w,
				fmt.Sprintf("Invalid dispersion combination: %v+%v", d.Data, d.Redundancy),
				http.StatusBadRequest)
			logger.LogError(fmt.Sprintf("Invalid dispersion combination: %v+%v", d.Data, d.Redundancy))
			return
		}
	}

	// Check that the clusters requested are available
	err = a.db.View(func(tx *bolt.Tx) error {

		// :TODO: All we need to do is check for one instead of gathering all keys
		clusters, err := ClusterList(tx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}
		if len(clusters) == 0 {
			http.Error(w, fmt.Sprintf("No clusters configured"), http.StatusBadRequest)
			logger.LogError("No clusters configured")
			return ErrNotFound
		}

		// Check the clusters requested are correct
		for _, clusterid := range msg.Clusters {
			_, err := NewClusterEntryFromId(tx, clusterid)
			if err != nil {
				http.Error(w, fmt.Sprintf("Cluster id %v not found", clusterid), http.StatusBadRequest)
				logger.LogError(fmt.Sprintf("Cluster id %v not found", clusterid))
				return err
			}
		}

		return nil
	})
	if err != nil {
		return
	}

	vol := NewVolumeEntryFromRequest(&msg)

	if uint64(msg.Size)*GB < vol.Durability.MinVolumeSize() {
		http.Error(w, fmt.Sprintf("Requested volume size (%v GB) is "+
			"smaller than the minimum supported volume size (%v)",
			msg.Size, vol.Durability.MinVolumeSize()),
			http.StatusBadRequest)
		logger.LogError(fmt.Sprintf("Requested volume size (%v GB) is "+
			"smaller than the minimum supported volume size (%v)",
			msg.Size, vol.Durability.MinVolumeSize()))
		return
	}

	// Add device in an asynchronous function
	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (string, error) {

		logger.Info("Creating volume %v", vol.Info.Id)
		err := vol.Create(a.db, a.executor, a.allocator)
		if err != nil {
			logger.LogError("Failed to create volume: %v", err)
			return "", err
		}

		logger.Info("Created volume %v", vol.Info.Id)

		// Done
		return "/volumes/" + vol.Info.Id, nil
	})

}

func (a *App) VolumeList(w http.ResponseWriter, r *http.Request) {

	var list api.VolumeListResponse

	// Get all the cluster ids from the DB
	err := a.db.View(func(tx *bolt.Tx) error {
		var err error

		list.Volumes, err = VolumeList(tx)
		if err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		logger.Err(err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Send list back
	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(list); err != nil {
		panic(err)
	}
}

func (a *App) VolumeInfo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var info *api.VolumeInfoResponse
	err := a.db.View(func(tx *bolt.Tx) error {
		entry, err := NewVolumeEntryFromId(tx, id)
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

	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(info); err != nil {
		panic(err)
	}

}

func (a *App) VolumeDelete(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var volume *VolumeEntry
	err := a.db.View(func(tx *bolt.Tx) error {

		var err error
		volume, err = NewVolumeEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		if volume.Info.Name == db.HeketiStorageVolumeName {
			err := fmt.Errorf("Cannot delete volume containing the Heketi database")
			http.Error(w, err.Error(), http.StatusConflict)
			return err
		}

		return nil

	})
	if err != nil {
		return
	}

	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (string, error) {

		// Actually destroy the Volume here
		err := volume.Destroy(a.db, a.executor)

		// If it fails for some reason, we will need to add to the DB again
		// or hold state on the entry "DELETING"

		// Show that the key has been deleted
		if err != nil {
			logger.LogError("Failed to delete volume %v: %v", volume.Info.Id, err)
			return "", err
		}

		logger.Info("Deleted volume [%s]", id)
		return "", nil

	})

}

func (a *App) VolumeExpand(w http.ResponseWriter, r *http.Request) {
	logger.Debug("In VolumeExpand")

	vars := mux.Vars(r)
	id := vars["id"]

	var msg api.VolumeExpandRequest
	err := utils.GetJsonFromRequest(r, &msg)
	if err != nil {
		http.Error(w, "request unable to be parsed", 422)
		return
	}
	logger.Debug("Msg: %v", msg)

	if msg.Size < 1 {
		http.Error(w, "Invalid volume size", http.StatusBadRequest)
		return
	}
	logger.Debug("Size: %v", msg.Size)

	var volume *VolumeEntry
	err = a.db.View(func(tx *bolt.Tx) error {

		var err error
		volume, err = NewVolumeEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
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

	// Expand volume in an asynchronous function
	a.asyncManager.AsyncHttpRedirectFunc(w, r, func() (string, error) {

		logger.Info("Expanding volume %v", volume.Info.Id)
		err := volume.Expand(a.db, a.executor, a.allocator, msg.Size)
		if err != nil {
			logger.LogError("Failed to expand volume %v", volume.Info.Id)
			return "", err
		}

		logger.Info("Expanded volume %v", volume.Info.Id)

		return "/volumes/" + volume.Info.Id, nil
	})

}
