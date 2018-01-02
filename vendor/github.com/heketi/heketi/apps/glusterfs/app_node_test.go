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
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func init() {
	// turn off logging
	logger.SetLevel(utils.LEVEL_NOLOG)
}

func TestNodeAddBadRequests(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// ClusterCreate JSON Request
	request := []byte(`{
		bad json
    }`)

	// Post bad JSON
	r, err := http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == 422)

	// Make a request without hostnames
	request = []byte(`{
		"cluster" : "123",
		"hostname" : {}
    }`)

	// Post bad JSON
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)

	// Make a request with only manage hostname
	request = []byte(`{
		"cluster" : "123",
		"hostnames" : {
			"manage" : [ "manage.hostname.com" ]
		},
		"zone" : 10
    }`)

	// Post bad JSON
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest, *r)

	// Make a request with only storage hostname
	request = []byte(`{
		"cluster" : "123",
		"hostnames" : {
			"storage" : [ "storage.hostname.com" ]
		},
		"zone" : 10
    }`)

	// Post bad JSON
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)

	// Make a request where the hostnames are empty strings
	request = []byte(`{
		"cluster" : "123",
		"hostnames" : {
			"storage" : [ "" ],
			"manage" : [ "" ]
		},
		"zone" : 10
    }`)

	// Check that it returns that the cluster id is not found
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	s, err := utils.GetStringFromResponse(r)
	tests.Assert(t, err == nil)
	tests.Assert(t, strings.Contains(s, "empty string"))

	// Make a request where the zone is missing
	request = []byte(`{
		"cluster" : "123",
		"hostnames" : {
			"storage" : [ "storage.hostname.com" ],
			"manage" : [ "manage.hostname.com"  ]
		}
    }`)

	// Check that it returns that the cluster id is not found
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	s, err = utils.GetStringFromResponse(r)
	tests.Assert(t, err == nil)
	tests.Assert(t, strings.Contains(s, "Zone cannot be zero"))

	// Make a request where the cluster id does not exist
	request = []byte(`{
		"cluster" : "123",
		"hostnames" : {
			"storage" : [ "storage.hostname.com" ],
			"manage" : [ "manage.hostname.com"  ]
		},
		"zone" : 10
    }`)

	// Check that it returns that the cluster id is not found
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusNotFound, r.StatusCode)
}

func TestPeerProbe(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// ClusterCreate JSON Request
	request := []byte(`{
    }`)

	// Post nothing
	r, err := http.Post(ts.URL+"/clusters", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusCreated)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read cluster information
	var clusterinfo api.ClusterInfoResponse
	err = utils.GetJsonFromResponse(r, &clusterinfo)
	tests.Assert(t, err == nil)

	// Override mock to check if the peer function was called
	probe_called := false
	app.xo.MockPeerProbe = func(exec_host, newnode string) error {
		probe_called = true
		return nil
	}

	// Create node on this cluster
	request = []byte(`{
		"cluster" : "` + clusterinfo.Id + `",
		"hostnames" : {
			"storage" : [ "storage0.hostname.com" ],
			"manage" : [ "manage0.hostname.com"  ]
		},
		"zone" : 1
    }`)

	// Create node
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusOK)
		if r.ContentLength <= 0 {
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			// Should have node information here
			tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")
			tests.Assert(t, err == nil)
			break
		}
	}
	tests.Assert(t, probe_called == false)

	// Now add another and check that probe was called
	request = []byte(`{
		"cluster" : "` + clusterinfo.Id + `",
		"hostnames" : {
			"storage" : [ "storage1.hostname.com" ],
			"manage" : [ "manage1.hostname.com"  ]
		},
		"zone" : 1
    }`)

	// Create node
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusOK)
		if r.ContentLength <= 0 {
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			// Should have node information here
			tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")
			tests.Assert(t, err == nil)
			break
		}
	}
	tests.Assert(t, probe_called == true)
}

func TestNodeAddDelete(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// ClusterCreate JSON Request
	request := []byte(`{
    }`)

	// Post nothing
	r, err := http.Post(ts.URL+"/clusters", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusCreated)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read cluster information
	var clusterinfo api.ClusterInfoResponse
	err = utils.GetJsonFromResponse(r, &clusterinfo)
	tests.Assert(t, err == nil)

	// Create node on this cluster
	request = []byte(`{
		"cluster" : "` + clusterinfo.Id + `",
		"hostnames" : {
			"storage" : [ "storage.hostname.com" ],
			"manage" : [ "manage.hostname.com"  ]
		},
		"zone" : 1
    }`)

	// Create node
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	var node api.NodeInfoResponse
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusOK)
		if r.ContentLength <= 0 {
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			// Should have node information here
			tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")
			err = utils.GetJsonFromResponse(r, &node)
			tests.Assert(t, err == nil)
			break
		}
	}
	tests.Assert(t, len(node.Id) > 0)
	tests.Assert(t, len(node.Hostnames.Manage) == 1)
	tests.Assert(t, len(node.Hostnames.Storage) == 1)
	tests.Assert(t, node.Hostnames.Manage[0] == "manage.hostname.com")
	tests.Assert(t, node.Hostnames.Storage[0] == "storage.hostname.com")
	tests.Assert(t, node.Zone == 1)
	tests.Assert(t, node.ClusterId == clusterinfo.Id)
	tests.Assert(t, len(node.DevicesInfo) == 0)

	// Check that the node has registered
	err = app.db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(BOLTDB_BUCKET_NODE))
		tests.Assert(t, b != nil)

		val := b.Get([]byte("STORAGE" + node.Hostnames.Storage[0]))
		tests.Assert(t, string(val) == node.Id)

		val = b.Get([]byte("MANAGE" + node.Hostnames.Manage[0]))
		tests.Assert(t, string(val) == node.Id)

		return nil
	})
	tests.Assert(t, err == nil)

	//---- OK, now it should have been registered
	// now let's add it again
	// It should return a conflict
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusConflict)

	// Check Cluster has node
	r, err = http.Get(ts.URL + "/clusters/" + clusterinfo.Id)
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	err = utils.GetJsonFromResponse(r, &clusterinfo)
	tests.Assert(t, len(clusterinfo.Nodes) == 1)
	tests.Assert(t, clusterinfo.Nodes[0] == node.Id)

	// Check the data is in the database correctly
	var entry *NodeEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err = NewNodeEntryFromId(tx, node.Id)
		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, entry != nil)
	tests.Assert(t, entry.Info.Id == node.Id)
	tests.Assert(t, len(entry.Info.Hostnames.Manage) == 1)
	tests.Assert(t, len(entry.Info.Hostnames.Storage) == 1)
	tests.Assert(t, entry.Info.Hostnames.Manage[0] == node.Hostnames.Manage[0])
	tests.Assert(t, entry.Info.Hostnames.Storage[0] == node.Hostnames.Storage[0])
	tests.Assert(t, len(entry.Devices) == 0)

	// Add some devices to check if delete conflict works
	err = app.db.Update(func(tx *bolt.Tx) error {
		entry, err = NewNodeEntryFromId(tx, node.Id)
		if err != nil {
			return err
		}

		entry.DeviceAdd("123")
		entry.DeviceAdd("456")
		return entry.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Now delete node and check for conflict
	req, err := http.NewRequest("DELETE", ts.URL+"/nodes/"+node.Id, nil)
	tests.Assert(t, err == nil)
	r, err = http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusConflict)
	tests.Assert(t, utils.GetErrorFromResponse(r).Error() == entry.ConflictString())

	// Check that nothing has changed in the db
	var cluster *ClusterEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		entry, err = NewNodeEntryFromId(tx, node.Id)
		if err != nil {
			return err
		}

		cluster, err = NewClusterEntryFromId(tx, entry.Info.ClusterId)
		if err != nil {
			return err
		}

		return nil
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, utils.SortedStringHas(cluster.Info.Nodes, node.Id))

	// Node delete the drives
	err = app.db.Update(func(tx *bolt.Tx) error {
		entry, err = NewNodeEntryFromId(tx, node.Id)
		if err != nil {
			return err
		}

		entry.DeviceDelete("123")
		entry.DeviceDelete("456")
		return entry.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Now delete node
	req, err = http.NewRequest("DELETE", ts.URL+"/nodes/"+node.Id, nil)
	tests.Assert(t, err == nil)
	r, err = http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Wait for deletion
	for {
		r, err := http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check db to make sure key is removed
	err = app.db.View(func(tx *bolt.Tx) error {
		_, err = NewNodeEntryFromId(tx, node.Id)
		return err
	})
	tests.Assert(t, err == ErrNotFound)

	// Check the cluster does not have this node id
	r, err = http.Get(ts.URL + "/clusters/" + clusterinfo.Id)
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	err = utils.GetJsonFromResponse(r, &clusterinfo)
	tests.Assert(t, len(clusterinfo.Nodes) == 0)

	// It should have deregistered the node
	// We should be able to add it again
	r, err = http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusOK)
		if r.ContentLength <= 0 {
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			// Should have node information here
			tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")
			err = utils.GetJsonFromResponse(r, &node)
			tests.Assert(t, err == nil)
			break
		}
	}
}

func TestNodeInfoIdNotFound(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Get unknown node id
	r, err := http.Get(ts.URL + "/nodes/123456789")
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusNotFound)

}

func TestNodeInfo(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create a node to save in the db
	node := NewNodeEntry()
	node.Info.Id = "abc"
	node.Info.ClusterId = "123"
	node.Info.Hostnames.Manage = sort.StringSlice{"manage.system"}
	node.Info.Hostnames.Storage = sort.StringSlice{"storage.system"}
	node.Info.Zone = 10

	// Save node in the db
	err := app.db.Update(func(tx *bolt.Tx) error {
		return node.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Get unknown node id
	r, err := http.Get(ts.URL + "/nodes/" + node.Info.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	var info api.NodeInfoResponse
	err = utils.GetJsonFromResponse(r, &info)
	tests.Assert(t, info.Id == node.Info.Id)
	tests.Assert(t, info.Hostnames.Manage[0] == node.Info.Hostnames.Manage[0])
	tests.Assert(t, len(info.Hostnames.Manage) == len(node.Info.Hostnames.Manage))
	tests.Assert(t, info.Hostnames.Storage[0] == node.Info.Hostnames.Storage[0])
	tests.Assert(t, len(info.Hostnames.Storage) == len(node.Info.Hostnames.Storage))
	tests.Assert(t, info.Zone == node.Info.Zone)

}

func TestNodeDeleteErrors(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create a node to save in the db
	node := NewNodeEntry()
	node.Info.Id = "abc"
	node.Info.ClusterId = "123"
	node.Info.Hostnames.Manage = sort.StringSlice{"manage.system"}
	node.Info.Hostnames.Storage = sort.StringSlice{"storage.system"}
	node.Info.Zone = 10

	// Save node in the db
	err := app.db.Update(func(tx *bolt.Tx) error {
		return node.Save(tx)
	})
	tests.Assert(t, err == nil)

	// Delete unknown id
	req, err := http.NewRequest("DELETE", ts.URL+"/nodes/123", nil)
	tests.Assert(t, err == nil)
	r, err := http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusNotFound)

}

func TestNodePeerProbeFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create a cluster.  We at least one
	// other node in the same cluster to execute a probe
	err := setupSampleDbWithTopology(app,
		1,     // clusters
		4,     // nodes_per_cluster
		4,     // devices_per_node,
		50*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Setup the mock peer probe to fail
	peerprobe_called := false
	peerprobe_calls := 0
	app.xo.MockPeerProbe = func(exec_host, newnode string) error {
		peerprobe_calls++
		peerprobe_called = true
		return errors.New("Mock")
	}

	// Get cluter id
	var clusterlist []string
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		clusterlist, err = ClusterList(tx)
		return err
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, len(clusterlist) == 1)
	clusterid := clusterlist[0]

	// Create a node
	storage_name := "host.hostname.com"
	manage_name := "host.hostname.com"
	request := []byte(`{
		"cluster" : "` + clusterid + `",
		"hostnames" : {
			"storage" : [ "` + storage_name + `" ],
			"manage" : [ "` + manage_name + `"  ]
		},
		"zone" : 1
    }`)

	// Create node
	r, err := http.Post(ts.URL+"/nodes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Since we forced the MockPeerProbe above to fail, the request should fail
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusInternalServerError)
			s, err := utils.GetStringFromResponse(r)
			tests.Assert(t, err == nil)
			tests.Assert(t, strings.TrimSpace(s) == "Mock")
			tests.Assert(t, peerprobe_called == true)
			tests.Assert(t, peerprobe_calls == 1)
			break
		}
	}

	// Check that the node has not been added to the db
	var nodelist []string
	var cluster *ClusterEntry
	err = app.db.View(func(tx *bolt.Tx) error {
		var err error
		cluster, err = NewClusterEntryFromId(tx, clusterid)
		if err != nil {
			return err
		}

		// Check that the node has not registered
		b := tx.Bucket([]byte(BOLTDB_BUCKET_NODE))
		tests.Assert(t, b != nil)

		val := b.Get([]byte("STORAGE" + storage_name))
		tests.Assert(t, val == nil)

		val = b.Get([]byte("MANAGE" + manage_name))
		tests.Assert(t, val == nil)

		// Set nodelist
		nodelist = EntryKeys(tx, BOLTDB_BUCKET_NODE)

		return nil
	})
	tests.Assert(t, err == nil)
	tests.Assert(t, len(nodelist) == 4)
	tests.Assert(t, len(cluster.Info.Nodes) == 4)
}

func TestNodePeerDetachFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create a cluster.  We do not want
	// any drives in the node so we can delete easily
	err := setupSampleDbWithTopology(app,
		1,     // clusters
		4,     // nodes_per_cluster
		0,     // devices_per_node,
		50*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Setup the mock peer probe to fail
	peer_called := false
	peer_calls := 0
	app.xo.MockPeerDetach = func(exec_host, newnode string) error {
		peer_calls++
		peer_called = true
		return errors.New("Mock")
	}

	// Get a node id
	var nodeid string
	err = app.db.View(func(tx *bolt.Tx) error {
		clusterlist, err := ClusterList(tx)
		if err != nil {
			return err
		}

		cluster, err := NewClusterEntryFromId(tx, clusterlist[0])
		if err != nil {
			return err
		}

		nodeid = cluster.Info.Nodes[0]

		return nil

	})
	tests.Assert(t, err == nil)
	tests.Assert(t, nodeid != "")

	// Delete node
	req, err := http.NewRequest("DELETE", ts.URL+"/nodes/"+nodeid, nil)
	tests.Assert(t, err == nil)
	r, err := http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Since we forced the MockPeerDetach above to fail, the request should fail
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusInternalServerError)
			s, err := utils.GetStringFromResponse(r)
			tests.Assert(t, err == nil)
			tests.Assert(t, strings.TrimSpace(s) == "Mock")
			tests.Assert(t, peer_called == true)
			tests.Assert(t, peer_calls == 1)
			break
		}
	}

	// Check that the node is still in the db
	err = app.db.View(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}

		cluster, err := NewClusterEntryFromId(tx, clusters[0])
		if err != nil {
			return err
		}
		tests.Assert(t, utils.SortedStringHas(cluster.Info.Nodes, nodeid))

		_, err = NewNodeEntryFromId(tx, nodeid)
		return err
	})
	tests.Assert(t, err == nil)
}

func TestNodePeerDetach(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create a cluster.  We do not want
	// any drives in the node so we can delete easily
	err := setupSampleDbWithTopology(app,
		1,     // clusters
		4,     // nodes_per_cluster
		0,     // devices_per_node,
		50*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Setup the mock peer probe to fail
	peer_called := false
	app.xo.MockPeerDetach = func(exec_host, newnode string) error {
		peer_called = true
		return nil
	}

	// get list of nodes
	var nodes []string
	err = app.db.View(func(tx *bolt.Tx) error {
		clusters, err := ClusterList(tx)
		if err != nil {
			return err
		}

		cluster, err := NewClusterEntryFromId(tx, clusters[0])
		if err != nil {
			return err
		}

		nodes = cluster.Info.Nodes
		return nil
	})
	tests.Assert(t, err == nil)

	// Delete nodes, peer detach should be called for each except the last one
	for index, node := range nodes {
		peer_called = false

		// Delete node
		req, err := http.NewRequest("DELETE", ts.URL+"/nodes/"+node, nil)
		tests.Assert(t, err == nil)
		r, err := http.DefaultClient.Do(req)
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusAccepted)
		location, err := r.Location()
		tests.Assert(t, err == nil)

		for {
			r, err = http.Get(location.String())
			tests.Assert(t, err == nil)
			if r.Header.Get("X-Pending") == "true" {
				tests.Assert(t, r.StatusCode == http.StatusOK)
				time.Sleep(time.Millisecond * 10)
			} else {
				tests.Assert(t, r.StatusCode == http.StatusNoContent)
				tests.Assert(t, err == nil)
				break
			}
		}

		// Check if detach was called
		if index == len(nodes)-1 {
			tests.Assert(t, peer_called == false)
		} else {
			tests.Assert(t, peer_called == true)
		}
	}
}

func TestNodeState(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Create mock allocator
	mockAllocator := NewMockAllocator(app.db)
	app.allocator = mockAllocator

	// Create a client
	c := client.NewClientNoAuth(ts.URL)
	tests.Assert(t, c != nil)

	// Create Cluster
	cluster, err := c.ClusterCreate()
	tests.Assert(t, err == nil)

	// Create Node
	nodeReq := &api.NodeAddRequest{
		Zone:      1,
		ClusterId: cluster.Id,
	}
	nodeReq.Hostnames.Manage = sort.StringSlice{"manage.host"}
	nodeReq.Hostnames.Storage = sort.StringSlice{"storage.host"}
	node, err := c.NodeAdd(nodeReq)
	tests.Assert(t, err == nil)

	// Add device
	deviceReq := &api.DeviceAddRequest{}
	deviceReq.Name = "/dev/fake1"
	deviceReq.NodeId = node.Id

	err = c.DeviceAdd(deviceReq)
	tests.Assert(t, err == nil)

	// Get node information again
	node, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.State == "online")

	// Get device information
	deviceId := node.DevicesInfo[0].Id
	device, err := c.DeviceInfo(deviceId)
	tests.Assert(t, err == nil)

	// Get info
	deviceInfo, err := c.DeviceInfo(device.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, deviceInfo.State == "online")

	// Check that the device is in the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[cluster.Id][0] == device.Id)

	// Set node offline
	request := []byte(`{
				"state" : "offline"
				}`)
	r, err := http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check it was removed from the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 0)

	// Get node info
	node, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.State == "offline")

	// Set offline again, should succeed
	request = []byte(`{
				"state" : "offline"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	// Check it was removed from the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 0)

	// Get node info
	node, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.State == "offline")

	// Set online again
	request = []byte(`{
				"state" : "online"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check that the device is in the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[cluster.Id][0] == device.Id)

	// Set online again, should succeed
	request = []byte(`{
				"state" : "online"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check that the device is in the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[cluster.Id][0] == device.Id)

	// Get node info
	node, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.State == "online")

	// Set unknown state
	request = []byte(`{
				"state" : "blah"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusInternalServerError)
			break
		}
	}

	// Check that the device is still in the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[cluster.Id][0] == device.Id)

	// Check node is still online
	node, err = c.NodeInfo(node.Id)
	tests.Assert(t, err == nil)
	tests.Assert(t, node.State == "online")

	// Set device offline
	request = []byte(`{
				"state" : "offline"
				}`)
	r, err = http.Post(ts.URL+"/devices/"+device.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Wait for deletion
	for {
		r, err := http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check it was removed from the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 0)

	// Set Node offline
	request = []byte(`{
				"state" : "offline"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check it was removed from the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 0)

	// Set Node online -- Device is still offline and should not be added
	request = []byte(`{
				"state" : "online"
				}`)
	r, err = http.Post(ts.URL+"/nodes/"+node.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)

	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	for {
		r, err = http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Check device is not in ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 0)

	// Now make device online
	request = []byte(`{
				"state" : "online"
				}`)
	r, err = http.Post(ts.URL+"/devices/"+device.Id+"/state",
		"application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err = r.Location()
	tests.Assert(t, err == nil)

	// Wait for deletion
	for {
		r, err := http.Get(location.String())
		tests.Assert(t, err == nil)
		if r.Header.Get("X-Pending") == "true" {
			tests.Assert(t, r.StatusCode == http.StatusOK)
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			break
		}
	}

	// Now it should be back in the ring
	tests.Assert(t, len(mockAllocator.clustermap[cluster.Id]) == 1)
	tests.Assert(t, mockAllocator.clustermap[cluster.Id][0] == device.Id)

}
