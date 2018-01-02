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
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/boltdb/bolt"
	"github.com/gorilla/mux"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/executors/kubeexec"
	"github.com/heketi/heketi/executors/mockexec"
	"github.com/heketi/heketi/executors/sshexec"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/rest"
)

const (
	ASYNC_ROUTE           = "/queue"
	BOLTDB_BUCKET_CLUSTER = "CLUSTER"
	BOLTDB_BUCKET_NODE    = "NODE"
	BOLTDB_BUCKET_VOLUME  = "VOLUME"
	BOLTDB_BUCKET_DEVICE  = "DEVICE"
	BOLTDB_BUCKET_BRICK   = "BRICK"
)

var (
	logger     = utils.NewLogger("[heketi]", utils.LEVEL_INFO)
	dbfilename = "heketi.db"
)

type App struct {
	asyncManager *rest.AsyncHttpManager
	db           *bolt.DB
	dbReadOnly   bool
	executor     executors.Executor
	allocator    Allocator
	conf         *GlusterFSConfig

	// For testing only.  Keep access to the object
	// not through the interface
	xo *mockexec.MockExecutor
}

// Use for tests only
func NewApp(configIo io.Reader) *App {
	app := &App{}

	// Load configuration file
	app.conf = loadConfiguration(configIo)
	if app.conf == nil {
		return nil
	}

	// Setup loglevel
	app.setLogLevel(app.conf.Loglevel)

	// Setup asynchronous manager
	app.asyncManager = rest.NewAsyncHttpManager(ASYNC_ROUTE)

	// Setup executor
	var err error
	switch {
	case app.conf.Executor == "mock":
		app.xo, err = mockexec.NewMockExecutor()
		app.executor = app.xo
	case app.conf.Executor == "kube" || app.conf.Executor == "kubernetes":
		app.executor, err = kubeexec.NewKubeExecutor(&app.conf.KubeConfig)
	case app.conf.Executor == "ssh" || app.conf.Executor == "":
		app.executor, err = sshexec.NewSshExecutor(&app.conf.SshConfig)
	default:
		return nil
	}
	if err != nil {
		logger.Err(err)
		return nil
	}
	logger.Info("Loaded %v executor", app.conf.Executor)

	// Set db is set in the configuration file
	if app.conf.DBfile != "" {
		dbfilename = app.conf.DBfile
	}

	// Setup BoltDB database
	app.db, err = bolt.Open(dbfilename, 0600, &bolt.Options{Timeout: 3 * time.Second})
	if err != nil {
		logger.Warning("Unable to open database.  Retrying using read only mode")

		// Try opening as read-only
		app.db, err = bolt.Open(dbfilename, 0666, &bolt.Options{
			ReadOnly: true,
		})
		if err != nil {
			logger.LogError("Unable to open database: %v", err)
			return nil
		}
		app.dbReadOnly = true
	} else {
		err = app.db.Update(func(tx *bolt.Tx) error {
			// Create Cluster Bucket
			_, err := tx.CreateBucketIfNotExists([]byte(BOLTDB_BUCKET_CLUSTER))
			if err != nil {
				logger.LogError("Unable to create cluster bucket in DB")
				return err
			}

			// Create Node Bucket
			_, err = tx.CreateBucketIfNotExists([]byte(BOLTDB_BUCKET_NODE))
			if err != nil {
				logger.LogError("Unable to create node bucket in DB")
				return err
			}

			// Create Volume Bucket
			_, err = tx.CreateBucketIfNotExists([]byte(BOLTDB_BUCKET_VOLUME))
			if err != nil {
				logger.LogError("Unable to create volume bucket in DB")
				return err
			}

			// Create Device Bucket
			_, err = tx.CreateBucketIfNotExists([]byte(BOLTDB_BUCKET_DEVICE))
			if err != nil {
				logger.LogError("Unable to create device bucket in DB")
				return err
			}

			// Create Brick Bucket
			_, err = tx.CreateBucketIfNotExists([]byte(BOLTDB_BUCKET_BRICK))
			if err != nil {
				logger.LogError("Unable to create brick bucket in DB")
				return err
			}

			// Handle Upgrade Changes
			err = app.Upgrade(tx)
			if err != nil {
				logger.LogError("Unable to Upgrade Changes")
				return err
			}

			return nil

		})
		if err != nil {
			logger.Err(err)
			return nil
		}
	}

	// Set advanced settings
	app.setAdvSettings()

	// Setup allocator
	switch {
	case app.conf.Allocator == "mock":
		app.allocator = NewMockAllocator(app.db)
	case app.conf.Allocator == "simple" || app.conf.Allocator == "":
		app.conf.Allocator = "simple"
		app.allocator = NewSimpleAllocatorFromDb(app.db)
	default:
		return nil
	}
	logger.Info("Loaded %v allocator", app.conf.Allocator)

	// Show application has loaded
	logger.Info("GlusterFS Application Loaded")

	return app
}

func (a *App) setLogLevel(level string) {
	switch level {
	case "none":
		logger.SetLevel(utils.LEVEL_NOLOG)
	case "critical":
		logger.SetLevel(utils.LEVEL_CRITICAL)
	case "error":
		logger.SetLevel(utils.LEVEL_ERROR)
	case "warning":
		logger.SetLevel(utils.LEVEL_WARNING)
	case "info":
		logger.SetLevel(utils.LEVEL_INFO)
	case "debug":
		logger.SetLevel(utils.LEVEL_DEBUG)
	}
}

// Upgrade Path to update all the values for new API entries
func (a *App) Upgrade(tx *bolt.Tx) error {

	err := ClusterEntryUpgrade(tx)
	if err != nil {
		logger.LogError("Failed to upgrade db for cluster entries")
		return err
	}

	err = NodeEntryUpgrade(tx)
	if err != nil {
		logger.LogError("Failed to upgrade db for node entries")
		return err
	}

	err = VolumeEntryUpgrade(tx)
	if err != nil {
		logger.LogError("Failed to upgrade db for volume entries")
		return err
	}

	err = DeviceEntryUpgrade(tx)
	if err != nil {
		logger.LogError("Failed to upgrade db for device entries")
		return err
	}

	err = BrickEntryUpgrade(tx)
	if err != nil {
		logger.LogError("Failed to upgrade db for brick entries: %v", err)
		return err
	}

	return nil
}

func (a *App) setAdvSettings() {
	if a.conf.BrickMaxNum != 0 {
		logger.Info("Adv: Max bricks per volume set to %v", a.conf.BrickMaxNum)

		// From volume_entry.go
		BrickMaxNum = a.conf.BrickMaxNum
	}
	if a.conf.BrickMaxSize != 0 {
		logger.Info("Adv: Max brick size %v GB", a.conf.BrickMaxSize)

		// From volume_entry.go
		// Convert to KB
		BrickMaxSize = uint64(a.conf.BrickMaxSize) * 1024 * 1024
	}
	if a.conf.BrickMinSize != 0 {
		logger.Info("Adv: Min brick size %v GB", a.conf.BrickMinSize)

		// From volume_entry.go
		// Convert to KB
		BrickMinSize = uint64(a.conf.BrickMinSize) * 1024 * 1024
	}
}

// Register Routes
func (a *App) SetRoutes(router *mux.Router) error {

	routes := rest.Routes{

		// Asynchronous Manager
		rest.Route{
			Name:        "Async",
			Method:      "GET",
			Pattern:     ASYNC_ROUTE + "/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.asyncManager.HandlerStatus},

		// Cluster
		rest.Route{
			Name:        "ClusterCreate",
			Method:      "POST",
			Pattern:     "/clusters",
			HandlerFunc: a.ClusterCreate},
		rest.Route{
			Name:        "ClusterInfo",
			Method:      "GET",
			Pattern:     "/clusters/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.ClusterInfo},
		rest.Route{
			Name:        "ClusterList",
			Method:      "GET",
			Pattern:     "/clusters",
			HandlerFunc: a.ClusterList},
		rest.Route{
			Name:        "ClusterDelete",
			Method:      "DELETE",
			Pattern:     "/clusters/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.ClusterDelete},

		// Node
		rest.Route{
			Name:        "NodeAdd",
			Method:      "POST",
			Pattern:     "/nodes",
			HandlerFunc: a.NodeAdd},
		rest.Route{
			Name:        "NodeInfo",
			Method:      "GET",
			Pattern:     "/nodes/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.NodeInfo},
		rest.Route{
			Name:        "NodeDelete",
			Method:      "DELETE",
			Pattern:     "/nodes/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.NodeDelete},
		rest.Route{
			Name:        "NodeSetState",
			Method:      "POST",
			Pattern:     "/nodes/{id:[A-Fa-f0-9]+}/state",
			HandlerFunc: a.NodeSetState},

		// Devices
		rest.Route{
			Name:        "DeviceAdd",
			Method:      "POST",
			Pattern:     "/devices",
			HandlerFunc: a.DeviceAdd},
		rest.Route{
			Name:        "DeviceInfo",
			Method:      "GET",
			Pattern:     "/devices/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.DeviceInfo},
		rest.Route{
			Name:        "DeviceDelete",
			Method:      "DELETE",
			Pattern:     "/devices/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.DeviceDelete},
		rest.Route{
			Name:        "DeviceSetState",
			Method:      "POST",
			Pattern:     "/devices/{id:[A-Fa-f0-9]+}/state",
			HandlerFunc: a.DeviceSetState},

		// Volume
		rest.Route{
			Name:        "VolumeCreate",
			Method:      "POST",
			Pattern:     "/volumes",
			HandlerFunc: a.VolumeCreate},
		rest.Route{
			Name:        "VolumeInfo",
			Method:      "GET",
			Pattern:     "/volumes/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.VolumeInfo},
		rest.Route{
			Name:        "VolumeExpand",
			Method:      "POST",
			Pattern:     "/volumes/{id:[A-Fa-f0-9]+}/expand",
			HandlerFunc: a.VolumeExpand},
		rest.Route{
			Name:        "VolumeDelete",
			Method:      "DELETE",
			Pattern:     "/volumes/{id:[A-Fa-f0-9]+}",
			HandlerFunc: a.VolumeDelete},
		rest.Route{
			Name:        "VolumeList",
			Method:      "GET",
			Pattern:     "/volumes",
			HandlerFunc: a.VolumeList},

		// Backup
		rest.Route{
			Name:        "Backup",
			Method:      "GET",
			Pattern:     "/backup/db",
			HandlerFunc: a.Backup},
	}

	// Register all routes from the App
	for _, route := range routes {

		// Add routes from the table
		router.
			Methods(route.Method).
			Path(route.Pattern).
			Name(route.Name).
			Handler(route.HandlerFunc)

	}

	// Set default error handler
	router.NotFoundHandler = http.HandlerFunc(a.NotFoundHandler)

	return nil
}

func (a *App) Close() {

	// Close the DB
	a.db.Close()
	logger.Info("Closed")
}

func (a *App) Backup(w http.ResponseWriter, r *http.Request) {
	err := a.db.View(func(tx *bolt.Tx) error {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Disposition", `attachment; filename="heketi.db"`)
		w.Header().Set("Content-Length", strconv.Itoa(int(tx.Size())))
		_, err := tx.WriteTo(w)
		return err
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (a *App) NotFoundHandler(w http.ResponseWriter, r *http.Request) {
	logger.Warning("Invalid path or request %v", r.URL.Path)
	http.Error(w, "Invalid path or request", http.StatusNotFound)
}
