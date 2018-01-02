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
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func TestAppBadConfigData(t *testing.T) {
	data := []byte(`{ bad json }`)
	app := NewApp(bytes.NewBuffer(data))
	tests.Assert(t, app == nil)

	data = []byte(`{}`)
	app = NewApp(bytes.NewReader(data))
	tests.Assert(t, app == nil)

	data = []byte(`{
		"glusterfs" : {}
		}`)
	app = NewApp(bytes.NewReader(data))
	tests.Assert(t, app == nil)
}

func TestAppUnknownExecutorInConfig(t *testing.T) {
	data := []byte(`{
		"glusterfs" : {
			"executor" : "unknown value here"
		}
		}`)
	app := NewApp(bytes.NewReader(data))
	tests.Assert(t, app == nil)
}

func TestAppUnknownAllocatorInConfig(t *testing.T) {
	data := []byte(`{
		"glusterfs" : {
			"allocator" : "unknown value here"
		}
		}`)
	app := NewApp(bytes.NewReader(data))
	tests.Assert(t, app == nil)
}

func TestAppBadDbLocation(t *testing.T) {
	data := []byte(`{
		"glusterfs" : {
			"db" : "/badlocation"
		}
	}`)
	app := NewApp(bytes.NewReader(data))
	tests.Assert(t, app == nil)
}

func TestAppAdvsettings(t *testing.T) {

	dbfile := tests.Tempfile()
	defer os.Remove(dbfile)
	os.Setenv("HEKETI_EXECUTOR", "mock")
	defer os.Unsetenv("HEKETI_EXECUTOR")

	data := []byte(`{
		"glusterfs" : {
			"executor" : "crazyexec",
			"allocator" : "simple",
			"db" : "` + dbfile + `",
			"brick_max_size_gb" : 1024,
			"brick_min_size_gb" : 4,
			"max_bricks_per_volume" : 33
		}
	}`)

	bmax, bmin, bnum := BrickMaxSize, BrickMinSize, BrickMaxNum
	defer func() {
		BrickMaxSize, BrickMinSize, BrickMaxNum = bmax, bmin, bnum
	}()

	app := NewApp(bytes.NewReader(data))
	defer app.Close()
	tests.Assert(t, app != nil)
	tests.Assert(t, app.conf.Executor == "mock")
	tests.Assert(t, BrickMaxNum == 33)
	tests.Assert(t, BrickMaxSize == 1*TB)
	tests.Assert(t, BrickMinSize == 4*GB)
}

func TestAppLogLevel(t *testing.T) {
	dbfile := tests.Tempfile()
	defer os.Remove(dbfile)

	levels := []string{
		"none",
		"critical",
		"error",
		"warning",
		"info",
		"debug",
	}

	logger.SetLevel(utils.LEVEL_DEBUG)
	for _, level := range levels {
		data := []byte(`{
			"glusterfs" : {
				"executor" : "mock",
				"allocator" : "simple",
				"db" : "` + dbfile + `",
				"loglevel" : "` + level + `"
			}
		}`)

		app := NewApp(bytes.NewReader(data))
		tests.Assert(t, app != nil, level, string(data))

		switch level {
		case "none":
			tests.Assert(t, logger.Level() == utils.LEVEL_NOLOG)
		case "critical":
			tests.Assert(t, logger.Level() == utils.LEVEL_CRITICAL)
		case "error":
			tests.Assert(t, logger.Level() == utils.LEVEL_ERROR)
		case "warning":
			tests.Assert(t, logger.Level() == utils.LEVEL_WARNING)
		case "info":
			tests.Assert(t, logger.Level() == utils.LEVEL_INFO)
		case "debug":
			tests.Assert(t, logger.Level() == utils.LEVEL_DEBUG)
		}
		app.Close()
	}

	// Test that an unknown value does not change the loglevel
	logger.SetLevel(utils.LEVEL_NOLOG)
	data := []byte(`{
			"glusterfs" : {
				"executor" : "mock",
				"allocator" : "simple",
				"db" : "` + dbfile + `",
				"loglevel" : "blah"
			}
		}`)

	app := NewApp(bytes.NewReader(data))
	defer app.Close()
	tests.Assert(t, app != nil)
	tests.Assert(t, logger.Level() == utils.LEVEL_NOLOG)
}

func TestAppReadOnlyDb(t *testing.T) {

	dbfile := tests.Tempfile()
	defer os.Remove(dbfile)

	// First, create a db
	data := []byte(`{
		"glusterfs": {
			"executor" : "mock",
			"db" : "` + dbfile + `"
		}
	}`)
	app := NewApp(bytes.NewReader(data))
	tests.Assert(t, app != nil)
	tests.Assert(t, app.dbReadOnly == false)
	app.Close()

	// Now open it again here.  This will force NewApp()
	// to be unable to open RW.
	db, err := bolt.Open(dbfile, 0666, &bolt.Options{
		ReadOnly: true,
	})
	tests.Assert(t, err == nil, err)
	tests.Assert(t, db != nil)

	// Now open it again and notice how it opened
	app = NewApp(bytes.NewReader(data))
	defer app.Close()
	tests.Assert(t, app != nil)
	tests.Assert(t, app.dbReadOnly == true)
}

func TestAppPathNotFound(t *testing.T) {
	dbfile := tests.Tempfile()
	defer os.Remove(dbfile)

	app := NewTestApp(dbfile)
	tests.Assert(t, app != nil)
	defer app.Close()
	router := mux.NewRouter()
	app.SetRoutes(router)

	// Setup the server
	ts := httptest.NewServer(router)
	defer ts.Close()

	// Setup a new client
	c := client.NewClientNoAuth(ts.URL)

	// Test paths which do not match the hexadecimal id
	_, err := c.ClusterInfo("xxx")
	tests.Assert(t, strings.Contains(err.Error(), "Invalid path or request"))

	_, err = c.NodeInfo("xxx")
	tests.Assert(t, strings.Contains(err.Error(), "Invalid path or request"))

	_, err = c.VolumeInfo("xxx")
	tests.Assert(t, strings.Contains(err.Error(), "Invalid path or request"))
}
