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
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/db"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func init() {
	// turn off logging
	logger.SetLevel(utils.LEVEL_NOLOG)
}

func TestVolumeCreateBadGid(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 100,
        "gid" : -1
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)

	body, err := utils.GetStringFromResponse(r)
	tests.Assert(t, err == nil)
	tests.Assert(t,
		strings.Contains(body, "Bad group id less than zero"))

	// VolumeCreate JSON Request
	request = []byte(`{
        "size" : 100,
        "gid" : ` + fmt.Sprintf("%v", math.MaxInt32) + `
    }`)

	// Send request
	r, err = http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)

	body, err = utils.GetStringFromResponse(r)
	tests.Assert(t, err == nil)
	tests.Assert(t,
		strings.Contains(body, "Bad group id equal or greater than 2**32"))

}

func TestVolumeCreateBadJson(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        asdfsdf
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == 422)
}

func TestVolumeCreateNoTopology(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 100
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
}

func TestVolumeCreateInvalidSize(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 0
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid volume size"))
}

func TestVolumeCreateSmallSize(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	os.Setenv("HEKETI_EXECUTOR", "mock")
	defer os.Unsetenv("HEKETI_EXECUTOR")

	data := []byte(`{
		"glusterfs" : {
			"db" : "` + tmpfile + `",
			"brick_min_size_gb" : 4
		}
	}`)

	bmin := BrickMinSize
	defer func() {
		BrickMinSize = bmin
	}()

	app := NewApp(bytes.NewReader(data))
	defer app.Close()

	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 2
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json",
		bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := utils.GetStringFromResponse(r)
	tests.Assert(t, err == nil)
	tests.Assert(t, strings.Contains(body, "Requested volume size (2 GB) "+
		"is smaller than the minimum supported volume size"), body)
}

func TestVolumeHeketiDbStorage(t *testing.T) {
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

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// VolumeCreate using default durability
	request := []byte(`{
        "size" : 100,
        "name" : "` + db.HeketiStorageVolumeName + `"
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	var info api.VolumeInfoResponse
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
			err = utils.GetJsonFromResponse(r, &info)
			tests.Assert(t, err == nil)
			break
		}
	}

	// Delete the volume
	req, err := http.NewRequest("DELETE", ts.URL+"/volumes/"+info.Id, nil)
	tests.Assert(t, err == nil)
	r, err = http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusConflict)
}

func TestVolumeCreateDurabilityTypeInvalid(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 100,
        "durability" : {
        	"type" : "bad type"
        }
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Unknown durability type"))
}

func TestVolumeCreateBadReplicaValues(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 100,
        "durability": {
        	"type": "replicate",
        	"replicate": {
            	"replica": 100
        	}
    	}
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid replica value"))

	// VolumeCreate JSON Request
	request = []byte(`{
        "size" : 100,
        "durability": {
        	"type": "replicate",
        	"replicate": {
            	"replica": 4
        	}
    	}
    }`)

	// Send request
	r, err = http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err = ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid replica value"))
}

func TestVolumeCreateBadDispersionValues(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 100,
        "durability": {
        	"type": "disperse",
        	"disperse": {
            	"data" : 8,
            	"redundancy" : 1
        	}
    	}
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid dispersion combination"))

	// VolumeCreate JSON Request
	request = []byte(`{
        "size" : 100,
        "durability": {
        	"type": "disperse",
        	"disperse": {
            	"data" : 4,
            	"redundancy" : 3
        	}
    	}
    }`)

	// Send request
	r, err = http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err = ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid dispersion combination"))
}

func TestVolumeCreateBadClusters(t *testing.T) {
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

	// Create a cluster
	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// VolumeCreate JSON Request
	request := []byte(`{
        "size" : 10,
        "clusters" : [
            "bad"
        ]
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Cluster id bad not found"))
}

func TestVolumeCreateBadSnapshotFactor(t *testing.T) {
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

	// Create JSON with missing factor
	request := []byte(`{
        "size" : 100,
        "snapshot" : {
            "enable" : true
        }
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid snapshot factor"))

	// Create JSON with large invalid factor
	request = []byte(`{
        "size" : 100,
        "snapshot" : {
            "enable" : true,
            "factor" : 101
        }
    }`)

	// Send request
	r, err = http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err = ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid snapshot factor"))

	// Create JSON with small invalid factor
	request = []byte(`{
        "size" : 100,
        "snapshot" : {
            "enable" : true,
            "factor" : 0.1
        }
    }`)

	// Send request
	r, err = http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err = ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid snapshot factor"))

}

func TestVolumeCreate(t *testing.T) {
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

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// VolumeCreate using default durability
	request := []byte(`{
        "size" : 100
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	var info api.VolumeInfoResponse
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
			err = utils.GetJsonFromResponse(r, &info)
			tests.Assert(t, err == nil)
			break
		}
	}
	tests.Assert(t, info.Id != "")
	tests.Assert(t, info.Cluster != "")
	tests.Assert(t, len(info.Bricks) == 1) // Only one 100GB brick needed
	tests.Assert(t, info.Bricks[0].Size == 100*GB)
	tests.Assert(t, info.Name == "vol_"+info.Id)
	tests.Assert(t, info.Snapshot.Enable == false)
	tests.Assert(t, info.Snapshot.Factor == 1)
	tests.Assert(t, info.Durability.Type == api.DurabilityDistributeOnly)
}

func TestVolumeInfoIdNotFound(t *testing.T) {
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

	// Now that we have some data in the database, we can
	// make a request for the clutser list
	r, err := http.Get(ts.URL + "/volumes/12345")
	tests.Assert(t, r.StatusCode == http.StatusNotFound)
	tests.Assert(t, err == nil)
}

func TestVolumeInfo(t *testing.T) {
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

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume
	req := &api.VolumeCreateRequest{}
	req.Size = 100
	req.Durability.Type = api.DurabilityEC
	v := NewVolumeEntryFromRequest(req)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Now that we have some data in the database, we can
	// make a request for the clutser list
	r, err := http.Get(ts.URL + "/volumes/" + v.Info.Id)
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read response
	var msg api.VolumeInfoResponse
	err = utils.GetJsonFromResponse(r, &msg)
	tests.Assert(t, err == nil)

	tests.Assert(t, msg.Id == v.Info.Id)
	tests.Assert(t, msg.Cluster == v.Info.Cluster)
	tests.Assert(t, msg.Name == v.Info.Name)
	tests.Assert(t, msg.Size == v.Info.Size)
	tests.Assert(t, reflect.DeepEqual(msg.Durability, v.Info.Durability))
	tests.Assert(t, reflect.DeepEqual(msg.Snapshot, v.Info.Snapshot))
	for _, brick := range msg.Bricks {
		tests.Assert(t, utils.SortedStringHas(v.Bricks, brick.Id))
	}
}

func TestVolumeListEmpty(t *testing.T) {
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

	// Get volumes, there should be none
	r, err := http.Get(ts.URL + "/volumes")
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read response
	var msg api.VolumeListResponse
	err = utils.GetJsonFromResponse(r, &msg)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(msg.Volumes) == 0)
}

func TestVolumeList(t *testing.T) {
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

	// Create some volumes
	numvolumes := 1000
	err := app.db.Update(func(tx *bolt.Tx) error {

		for i := 0; i < numvolumes; i++ {
			v := createSampleVolumeEntry(100)
			err := v.Save(tx)
			if err != nil {
				return err
			}
		}

		return nil

	})
	tests.Assert(t, err == nil)

	// Get volumes, there should be none
	r, err := http.Get(ts.URL + "/volumes")
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read response
	var msg api.VolumeListResponse
	err = utils.GetJsonFromResponse(r, &msg)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(msg.Volumes) == numvolumes)

	// Check that all the volumes are in the database
	err = app.db.View(func(tx *bolt.Tx) error {
		for _, id := range msg.Volumes {
			_, err := NewVolumeEntryFromId(tx, id)
			if err != nil {
				return err
			}
		}

		return nil
	})
	tests.Assert(t, err == nil)

}

func TestVolumeListReadOnlyDb(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)

	// Create some volumes
	numvolumes := 1000
	err := app.db.Update(func(tx *bolt.Tx) error {

		for i := 0; i < numvolumes; i++ {
			v := createSampleVolumeEntry(100)
			err := v.Save(tx)
			if err != nil {
				return err
			}
		}

		return nil

	})
	tests.Assert(t, err == nil)
	app.Close()

	// Open Db here to force read only mode
	db, err := bolt.Open(tmpfile, 0666, &bolt.Options{
		ReadOnly: true,
	})
	tests.Assert(t, err == nil, err)
	tests.Assert(t, db != nil)

	// Create the app
	app = NewTestApp(tmpfile)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Get volumes, there should be none
	r, err := http.Get(ts.URL + "/volumes")
	tests.Assert(t, r.StatusCode == http.StatusOK)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.Header.Get("Content-Type") == "application/json; charset=UTF-8")

	// Read response
	var msg api.VolumeListResponse
	err = utils.GetJsonFromResponse(r, &msg)
	tests.Assert(t, err == nil)
	tests.Assert(t, len(msg.Volumes) == numvolumes)

	// Check that all the volumes are in the database
	err = app.db.View(func(tx *bolt.Tx) error {
		for _, id := range msg.Volumes {
			_, err := NewVolumeEntryFromId(tx, id)
			if err != nil {
				return err
			}
		}

		return nil
	})
	tests.Assert(t, err == nil)

}

func TestVolumeDeleteIdNotFound(t *testing.T) {
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

	// Now that we have some data in the database, we can
	// make a request for the clutser list
	req, err := http.NewRequest("DELETE", ts.URL+"/volumes/12345", nil)
	tests.Assert(t, err == nil)
	r, err := http.DefaultClient.Do(req)
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusNotFound)
}

func TestVolumeDelete(t *testing.T) {
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

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume
	v := createSampleVolumeEntry(100)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Delete the volume
	req, err := http.NewRequest("DELETE", ts.URL+"/volumes/"+v.Info.Id, nil)
	tests.Assert(t, err == nil)
	r, err := http.DefaultClient.Do(req)
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
			continue
		} else {
			tests.Assert(t, r.StatusCode == http.StatusNoContent)
			tests.Assert(t, err == nil)
			break
		}
	}

	// Check it is not there
	r, err = http.Get(ts.URL + "/volumes/" + v.Info.Id)
	tests.Assert(t, r.StatusCode == http.StatusNotFound)
	tests.Assert(t, err == nil)
}

func TestVolumeExpandBadJson(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "asdfasd  0
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == 422)
}

func TestVolumeExpandIdNotFound(t *testing.T) {
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

	// JSON Request
	request := []byte(`{
        "expand_size" : 100
    }`)

	// Now that we have some data in the database, we can
	// make a request for the clutser list
	r, err := http.Post(ts.URL+"/volumes/12345/expand",
		"application/json",
		bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusNotFound, r.StatusCode)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Id not found"))
}

func TestVolumeExpandSizeTooSmall(t *testing.T) {
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

	// VolumeCreate JSON Request
	request := []byte(`{
        "expand_size" : 0
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusBadRequest)
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	tests.Assert(t, err == nil)
	r.Body.Close()
	tests.Assert(t, strings.Contains(string(body), "Invalid volume size"))
}

func TestVolumeExpand(t *testing.T) {
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

	// Create a cluster
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume
	v := createSampleVolumeEntry(100)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Keep a copy
	vc := &VolumeEntry{}
	*vc = *v

	// JSON Request
	request := []byte(`{
        "expand_size" : 1000
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes/"+v.Info.Id+"/expand",
		"application/json",
		bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	var info api.VolumeInfoResponse
	for {
		r, err := http.Get(location.String())
		tests.Assert(t, err == nil)
		tests.Assert(t, r.StatusCode == http.StatusOK)
		if r.Header.Get("X-Pending") == "true" {
			time.Sleep(time.Millisecond * 10)
			continue
		} else {
			err = utils.GetJsonFromResponse(r, &info)
			tests.Assert(t, err == nil)
			break
		}
	}

	tests.Assert(t, info.Size == 100+1000)
	tests.Assert(t, len(vc.Bricks) < len(info.Bricks))
}

func TestVolumeClusterResizeByAddingDevices(t *testing.T) {
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

	// Create a cluster
	err := setupSampleDbWithTopology(app,
		1,      // clusters
		2,      // nodes_per_cluster
		1,      // devices_per_node,
		500*GB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a volume which uses the entire storage
	v := createSampleVolumeEntry(495)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Try to create another volume, but this should fail
	v = createSampleVolumeEntry(495)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == ErrNoSpace)

	// Create a client
	c := client.NewClientNoAuth(ts.URL)
	tests.Assert(t, c != nil)

	// Get the cluster ID
	clusters, err := c.ClusterList()
	tests.Assert(t, len(clusters.Clusters) == 1)
	clusterId := clusters.Clusters[0]

	// Get Nodes
	clusterInfo, err := c.ClusterInfo(clusterId)
	tests.Assert(t, len(clusterInfo.Nodes) == 2)

	// Add two devices to the cluster
	for _, nodeId := range clusterInfo.Nodes {
		d := &api.DeviceAddRequest{}
		d.Name = "/fake/device"
		d.NodeId = nodeId
		err := c.DeviceAdd(d)
		tests.Assert(t, err == nil, err)
	}

	// Now add a volume, and it should work
	v = createSampleVolumeEntry(495)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == nil)

	// Try to create another volume, but this should fail
	v = createSampleVolumeEntry(495)
	tests.Assert(t, v != nil)
	err = v.Create(app.db, app.executor, app.allocator)
	tests.Assert(t, err == ErrNoSpace)
}

// Test for https://github.com/heketi/heketi/issues/382:
//
// A TopologyInfo request running concurrently to a
// VolumeCreate request failed with "Id not found" due
// to unsaved brick info.
func TestVolumeCreateVsTopologyInfo(t *testing.T) {
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

	// Create a cluster
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// Create a client
	c := client.NewClientNoAuth(ts.URL)
	tests.Assert(t, c != nil)

	// Start Several concurrent VolumeCreate and
	// TopologyInfo requests so that there is a
	// chance for a TopologyInfo request to hit
	// the race of unsaved // brick data.
	sg := utils.NewStatusGroup()
	for i := 0; i < 20; i++ {
		sg.Add(1)
		go func() {
			defer sg.Done()

			volumeReq := &api.VolumeCreateRequest{}
			volumeReq.Size = 10

			volume, err := c.VolumeCreate(volumeReq)
			sg.Err(err)
			if err != nil {
				return
			}

			if volume.Id == "" {
				sg.Err(errors.New("Empty volume Id."))
				return
			}

			if volume.Size != volumeReq.Size {
				sg.Err(fmt.Errorf("Unexpected Volume size "+
					"[%d] instead of [%d].",
					volume.Size, volumeReq.Size))
			}
		}()

		sg.Add(1)
		go func() {
			defer sg.Done()

			_, err := c.TopologyInfo()
			if err != nil {
				err = fmt.Errorf("TopologyInfo failed: %s", err)
			}
			sg.Err(err)
		}()
	}

	err = sg.Result()
	tests.Assert(t, err == nil, err)
}
func TestVolumeCreateWithOptions(t *testing.T) {
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

	// Setup database
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		10,   // nodes_per_cluster
		10,   // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)

	// VolumeCreate using a test option called "test-option"
	request := []byte(`{
        "size" : 100,
		"glustervolumeoptions" : [
			"test-option"
			]
    }`)

	// Send request
	r, err := http.Post(ts.URL+"/volumes", "application/json", bytes.NewBuffer(request))
	tests.Assert(t, err == nil)
	tests.Assert(t, r.StatusCode == http.StatusAccepted)
	location, err := r.Location()
	tests.Assert(t, err == nil)

	// Query queue until finished
	var info api.VolumeInfoResponse
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
			err = utils.GetJsonFromResponse(r, &info)
			tests.Assert(t, err == nil)
			break
		}
	}

	tests.Assert(t, info.Id != "")
	tests.Assert(t, info.Cluster != "")
	tests.Assert(t, len(info.Bricks) == 1) // Only one 100GB brick needed
	tests.Assert(t, info.Bricks[0].Size == 100*GB)
	tests.Assert(t, info.Name == "vol_"+info.Id)
	tests.Assert(t, info.Snapshot.Enable == false)
	tests.Assert(t, info.Snapshot.Factor == 1)
	tests.Assert(t, info.Durability.Type == api.DurabilityDistributeOnly)
	// GlusterVolumeOption should have the "test-option"
	tests.Assert(t, info.GlusterVolumeOptions[0] == "test-option")

}
