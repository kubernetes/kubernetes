//
// Copyright (c) 2017 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"bytes"
	"compress/gzip"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/pkg/utils"
	"github.com/heketi/tests"
)

func init() {
	logger.SetLevel(utils.LEVEL_NOLOG)
}

func TestBackupToKubeSecretMaxSize(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)

	// Create a Max cluster size when using Secrets
	// Max secret size is 1024M in Kubernetes
	// Max = 12000 Drives
	err := setupSampleDbWithTopology(app,
		1,    // clusters
		120,  // nodes_per_cluster
		100,  // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)
	app.Close()

	// Gzip database
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	dbData, err := ioutil.ReadFile(tmpfile)
	tests.Assert(t, err == nil)
	_, err = gz.Write(dbData)
	tests.Assert(t, err == nil)
	err = gz.Close()
	tests.Assert(t, err == nil)
	tests.Assert(t, b.Len() < 1024*MB)
}

func TestBackupToKubeSecretMaxSizeFailure(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)

	// Create a Max cluster size when using Secrets
	// Max secret size is 1024M in Kubernetes
	// Max = 12000 Drives
	// So here pick 24000 Drives in two clusters
	err := setupSampleDbWithTopology(app,
		2,    // clusters
		120,  // nodes_per_cluster
		100,  // devices_per_node,
		5*TB, // disksize)
	)
	tests.Assert(t, err == nil)
	app.Close()

	// Gzip database
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	dbData, err := ioutil.ReadFile(tmpfile)
	tests.Assert(t, err == nil)
	_, err = gz.Write(dbData)
	tests.Assert(t, err == nil)
	err = gz.Close()
	tests.Assert(t, err == nil)
	tests.Assert(t, b.Len() > 1024*MB)
}

func TestBackupToKubeSecretBackupOnNonGet(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	incluster_count := 0
	defer tests.Patch(&kubeBackupDbToSecret, func(db *bolt.DB) error {
		incluster_count++
		return nil
	}).Restore()

	// Backup on Post
	r, err := http.NewRequest(http.MethodPost, "http://mytest.com/hello", nil)
	tests.Assert(t, err == nil)
	w := httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	tests.Assert(t, incluster_count == 1)

	// Backup on PUT
	r, err = http.NewRequest(http.MethodPut, "http://mytest.com/hello", nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	tests.Assert(t, incluster_count == 2)

	// Backup on DELETE
	r, err = http.NewRequest(http.MethodDelete, "http://mytest.com/hello", nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	tests.Assert(t, incluster_count == 3)
}

func TestBackupToKubeSecretBackupOnGet(t *testing.T) {
	tmpfile := tests.Tempfile()
	defer os.Remove(tmpfile)

	// Create the app
	app := NewTestApp(tmpfile)
	defer app.Close()

	incluster_count := 0
	defer tests.Patch(&kubeBackupDbToSecret, func(db *bolt.DB) error {
		incluster_count++
		return nil
	}).Restore()

	// No backups on GET to non-/queue URLs
	r, err := http.NewRequest(http.MethodGet, "http://mytest.com/hello", nil)
	tests.Assert(t, err == nil)
	w := httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	tests.Assert(t, incluster_count == 0)

	// No backups on GET on /queue URL where the Status is still 200 (OK)
	// which means that the resource is pending
	r, err = http.NewRequest(http.MethodGet, "http://mytest.com"+ASYNC_ROUTE, nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	tests.Assert(t, incluster_count == 0)

	// No backups on GET on /queue URL where the Status is error
	r, err = http.NewRequest(http.MethodGet, "http://mytest.com"+ASYNC_ROUTE, nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	tests.Assert(t, incluster_count == 0)

	// Backup when a GET on /queue gets a Done
	r, err = http.NewRequest(http.MethodGet, "http://mytest.com"+ASYNC_ROUTE, nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	tests.Assert(t, incluster_count == 1)

	// Backup when a GET on /queue gets a See Other
	r, err = http.NewRequest(http.MethodGet, "http://mytest.com"+ASYNC_ROUTE, nil)
	tests.Assert(t, err == nil)
	w = httptest.NewRecorder()
	app.BackupToKubernetesSecret(w, r, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusSeeOther)
	})
	tests.Assert(t, incluster_count == 2)
}
