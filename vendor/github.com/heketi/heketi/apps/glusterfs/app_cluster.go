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
)

func (a *App) ClusterCreate(w http.ResponseWriter, r *http.Request) {

	// Create a new ClusterInfo
	entry := NewClusterEntryFromRequest()

	// Add cluster to db
	err := a.db.Update(func(tx *bolt.Tx) error {
		err := entry.Save(tx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		return nil

	})
	if err != nil {
		return
	}

	// Send back we created it (as long as we did not fail)
	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusCreated)
	if err := json.NewEncoder(w).Encode(entry.Info); err != nil {
		panic(err)
	}
}

func (a *App) ClusterList(w http.ResponseWriter, r *http.Request) {

	var list api.ClusterListResponse

	// Get all the cluster ids from the DB
	err := a.db.View(func(tx *bolt.Tx) error {
		var err error

		list.Clusters, err = ClusterList(tx)
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

func (a *App) ClusterInfo(w http.ResponseWriter, r *http.Request) {

	// Get the id from the URL
	vars := mux.Vars(r)
	id := vars["id"]

	// Get info from db
	var info *api.ClusterInfoResponse
	err := a.db.View(func(tx *bolt.Tx) error {

		// Create a db entry from the id
		entry, err := NewClusterEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return err
		}

		// Create a response from the db entry
		info, err = entry.NewClusterInfoResponse(tx)
		if err != nil {
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

func (a *App) ClusterDelete(w http.ResponseWriter, r *http.Request) {

	// Get the id from the URL
	vars := mux.Vars(r)
	id := vars["id"]

	// Delete cluster from db
	err := a.db.Update(func(tx *bolt.Tx) error {

		// Access cluster entry
		entry, err := NewClusterEntryFromId(tx, id)
		if err == ErrNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
			return err
		} else if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return logger.Err(err)
		}

		err = entry.Delete(tx)
		if err != nil {
			if err == ErrConflict {
				http.Error(w, entry.ConflictString(), http.StatusConflict)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return err
		}

		return nil
	})
	if err != nil {
		return
	}

	// Update allocator hat the cluster has been removed
	a.allocator.RemoveCluster(id)

	// Show that the key has been deleted
	logger.Info("Deleted cluster [%s]", id)

	// Write msg
	w.WriteHeader(http.StatusOK)
}
