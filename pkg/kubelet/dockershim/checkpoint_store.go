/*
Copyright 2017 The Kubernetes Authors.

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

package dockershim

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/errors"
)

const (
	tmpPrefix    = "."
	tmpSuffix    = ".tmp"
	keyMaxLength = 250
)

var keyRegex = regexp.MustCompile("^[a-zA-Z0-9]+$")

// CheckpointStore provides the interface for checkpoint storage backend.
// CheckpointStore must be thread-safe
type CheckpointStore interface {
	// key must contain one or more characters in [A-Za-z0-9]
	// Write persists a checkpoint with key
	Write(key string, data []byte) error
	// Read retrieves a checkpoint with key
	// Read must return CheckpointNotFoundError if checkpoint is not found
	Read(key string) ([]byte, error)
	// Delete deletes a checkpoint with key
	// Delete must not return error if checkpoint does not exist
	Delete(key string) error
	// List lists all keys of existing checkpoints
	List() ([]string, error)
}

// FileStore is an implementation of CheckpointStore interface which stores checkpoint in files.
type FileStore struct {
	// path to the base directory for storing checkpoint files
	path string
}

func (fstore *FileStore) Write(key string, data []byte) error {
	if err := validateKey(key); err != nil {
		return err
	}
	if _, err := os.Stat(fstore.path); err != nil {
		// if directory already exists, proceed
		if err = os.MkdirAll(fstore.path, 0755); err != nil && !os.IsExist(err) {
			return err
		}
	}
	tmpfile := filepath.Join(fstore.path, fmt.Sprintf("%s%s%s", tmpPrefix, key, tmpSuffix))
	if err := ioutil.WriteFile(tmpfile, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpfile, fstore.getCheckpointPath(key))
}

func (fstore *FileStore) Read(key string) ([]byte, error) {
	if err := validateKey(key); err != nil {
		return nil, err
	}
	bytes, err := ioutil.ReadFile(fstore.getCheckpointPath(key))
	if os.IsNotExist(err) {
		return bytes, errors.CheckpointNotFoundError
	}
	return bytes, err
}

func (fstore *FileStore) Delete(key string) error {
	if err := validateKey(key); err != nil {
		return err
	}
	if err := os.Remove(fstore.getCheckpointPath(key)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (fstore *FileStore) List() ([]string, error) {
	keys := make([]string, 0)
	files, err := ioutil.ReadDir(fstore.path)
	if err != nil {
		return keys, err
	}
	for _, f := range files {
		if !strings.HasPrefix(f.Name(), tmpPrefix) {
			keys = append(keys, f.Name())
		}
	}
	return keys, nil
}

func (fstore *FileStore) getCheckpointPath(key string) string {
	return filepath.Join(fstore.path, key)
}

func validateKey(key string) error {
	if len(key) <= keyMaxLength && keyRegex.MatchString(key) {
		return nil
	}
	return fmt.Errorf("checkpoint key %q is not valid.", key)
}
