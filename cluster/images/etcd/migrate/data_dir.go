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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"
)

// DataDirectory provides utilities for initializing and backing up an
// etcd "data-dir" as well as managing a version.txt file to track the
// etcd server version and storage version of the etcd data in the
// directory.
type DataDirectory struct {
	path        string
	versionFile *VersionFile
}

// OpenOrCreateDataDirectory opens a data directory, creating the directory
// if it doesn't not already exist.
func OpenOrCreateDataDirectory(path string) (*DataDirectory, error) {
	exists, err := exists(path)
	if err != nil {
		return nil, err
	}
	if !exists {
		klog.Infof("data directory '%s' does not exist, creating it", path)
		err := os.MkdirAll(path, 0777)
		if err != nil {
			return nil, fmt.Errorf("failed to create data directory %s: %v", path, err)
		}
	}
	versionFile := &VersionFile{
		path: filepath.Join(path, versionFilename),
	}
	return &DataDirectory{path, versionFile}, nil
}

// Initialize set the version.txt to the target version if the data
// directory is empty. If the data directory is non-empty, no
// version.txt file will be written since the actual version of etcd
// used to create the data is unknown.
func (d *DataDirectory) Initialize(target *EtcdVersionPair) error {
	isEmpty, err := d.IsEmpty()
	if err != nil {
		return err
	}
	if isEmpty {
		klog.Infof("data directory '%s' is empty, writing target version '%s' to version.txt", d.path, target)
		err = d.versionFile.Write(target)
		if err != nil {
			return fmt.Errorf("failed to write version.txt to '%s': %v", d.path, err)
		}
		return nil
	}
	return nil
}

// Backup creates a backup copy of data directory.
func (d *DataDirectory) Backup() error {
	backupDir := fmt.Sprintf("%s.bak", d.path)
	err := os.RemoveAll(backupDir)
	if err != nil {
		return err
	}
	err = os.MkdirAll(backupDir, 0777)
	if err != nil {
		return err
	}
	err = copyDirectory(d.path, backupDir)
	if err != nil {
		return err
	}

	return nil
}

// IsEmpty returns true if the data directory is entirely empty.
func (d *DataDirectory) IsEmpty() (bool, error) {
	dir, err := os.Open(d.path)
	if err != nil {
		return false, fmt.Errorf("failed to open data directory %s: %v", d.path, err)
	}
	defer dir.Close()
	_, err = dir.Readdirnames(1)
	if err == io.EOF {
		return true, nil
	}
	return false, err
}

// String returns the data directory path.
func (d *DataDirectory) String() string {
	return d.path
}

// VersionFile provides utilities for reading and writing version.txt files
// to etcd "data-dir" for tracking the etcd server and storage versions
// of the data in the directory.
type VersionFile struct {
	path string
}

// Exists returns true if a version.txt file exists on the file system.
func (v *VersionFile) Exists() (bool, error) {
	return exists(v.path)
}

// Read parses the version.txt file and returns it's contents.
func (v *VersionFile) Read() (*EtcdVersionPair, error) {
	data, err := ioutil.ReadFile(v.path)
	if err != nil {
		return nil, fmt.Errorf("failed to read version file %s: %v", v.path, err)
	}
	txt := strings.TrimSpace(string(data))
	vp, err := ParseEtcdVersionPair(txt)
	if err != nil {
		return nil, fmt.Errorf("failed to parse etcd '<version>/<storage-version>' string from version.txt file contents '%s': %v", txt, err)
	}
	return vp, nil
}

// Write creates or overwrites the contents of the version.txt file with the given EtcdVersionPair.
func (v *VersionFile) Write(vp *EtcdVersionPair) error {
	data := []byte(fmt.Sprintf("%s/%s", vp.version, vp.storageVersion))
	return ioutil.WriteFile(v.path, data, 0666)
}

func exists(path string) (bool, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}
