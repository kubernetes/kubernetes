/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/blang/semver"
)

var (
	latestVersion = semver.MustParse("3.1.12")
)

func TestExistingDataDirWithVersionFile(t *testing.T) {
	d, err := OpenOrCreateDataDirectory("testdata/datadir_with_version")
	if err != nil {
		t.Fatalf("Failed to open data dir: %v", err)
	}
	isEmpty, err := d.IsEmpty()
	if err != nil {
		t.Fatalf("Failed to check if data dir is empty: %v", err)
	}
	if isEmpty {
		t.Errorf("Expected non-empty data directory to exist")
	}
	exists, err := d.versionFile.Exists()
	if err != nil {
		t.Fatal(err)
	}
	if !exists {
		t.Fatalf("Expected version file %s to exist", d.versionFile.path)
	}
	vp, err := d.versionFile.Read()
	if err != nil {
		t.Fatalf("Failed to read version file %s: %v", d.versionFile.path, err)
	}
	expectedVersion := &EtcdVersionPair{&EtcdVersion{latestVersion}, storageEtcd3}
	if !vp.Equals(expectedVersion) {
		t.Errorf("Expected version file to contain %s, but got %s", expectedVersion, vp)
	}
}

func TestExistingDataDirWithoutVersionFile(t *testing.T) {
	targetVersion := &EtcdVersionPair{&EtcdVersion{latestVersion}, storageEtcd3}

	d, err := OpenOrCreateDataDirectory("testdata/datadir_without_version")
	if err != nil {
		t.Fatalf("Failed to open data dir: %v", err)
	}
	exists, err := d.versionFile.Exists()
	if err != nil {
		t.Fatal(err)
	}
	if exists {
		t.Errorf("Expected version file %s not to exist", d.versionFile.path)
	}
	err = d.Initialize(targetVersion)
	if err != nil {
		t.Fatalf("Failed initialize data directory %s: %v", d.path, err)
	}
	exists, err = d.versionFile.Exists()
	if err != nil {
		t.Fatal(err)
	}
	if exists {
		t.Fatalf("Expected version file %s not to exist after initializing non-empty data-dir", d.versionFile.path)
	}
}

func TestNonexistingDataDir(t *testing.T) {
	targetVersion := &EtcdVersionPair{&EtcdVersion{latestVersion}, storageEtcd3}
	path := newTestPath(t)
	d, err := OpenOrCreateDataDirectory(filepath.Join(path, "data-dir"))
	if err != nil {
		t.Fatalf("Failed to open data dir: %v", err)
	}
	isEmpty, err := d.IsEmpty()
	if err != nil {
		t.Fatalf("Failed to check if data dir is empty: %v", err)
	}
	if !isEmpty {
		t.Errorf("Expected empty data directory to exist")
	}
	err = d.Initialize(targetVersion)
	if err != nil {
		t.Fatalf("Failed initialize data directory %s: %v", d.path, err)
	}
	exists, err := d.versionFile.Exists()
	if err != nil {
		t.Fatal(err)
	}
	if !exists {
		t.Fatalf("Expected version file %s to exist", d.versionFile.path)
	}
	isEmpty, err = d.IsEmpty()
	if err != nil {
		t.Fatalf("Failed to check if data dir is empty: %v", err)
	}
	if isEmpty {
		t.Errorf("Expected non-empty data directory to exist after Initialize()")
	}
	vp, err := d.versionFile.Read()
	if err != nil {
		t.Fatalf("Failed to read version file %s: %v", d.versionFile.path, err)
	}
	if !vp.Equals(targetVersion) {
		t.Errorf("Expected version file to contain %s, but got %s", targetVersion, vp)
	}
}

func TestBackup(t *testing.T) {
	path := newTestPath(t)
	d, err := OpenOrCreateDataDirectory(filepath.Join(path, "data-dir"))
	if err != nil {
		t.Fatalf("Failed to open data dir: %v", err)
	}
	_, err = os.Create(filepath.Join(path, "data-dir", "empty.txt"))
	if err != nil {
		t.Fatal(err)
	}
	err = d.Backup()
	if err != nil {
		t.Fatalf("Failed to backup data directory %s: %v", d.path, err)
	}
	bak, err := OpenOrCreateDataDirectory(filepath.Join(path, "data-dir.bak"))
	if err != nil {
		t.Fatalf("Failed to open backup data dir: %v", err)
	}
	isEmpty, err := bak.IsEmpty()
	if err != nil {
		t.Fatal(err)
	}
	if isEmpty {
		t.Errorf("Expected non-empty backup directory to exist after Backup()")
	}
}

func newTestPath(t *testing.T) string {
	path, err := ioutil.TempDir("", "etcd-migrate-test-")
	if err != nil {
		t.Fatalf("Failed to create tmp dir for test: %v", err)
	}
	err = os.Chmod(path, 0777)
	if err != nil {
		t.Fatalf("Failed to granting permission to tmp dir for test: %v", err)
	}
	return path
}
