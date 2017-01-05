/*
Copyright 2016 The Kubernetes Authors.

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
	"sync"
)

const (
	tmpPrefix    = "."
	tmpSuffix    = ".tmp"
	keyMaxLength = 250
	keyRegex     = "^[a-zA-Z0-9]+$"
)

// CheckpointStore provides the interface for checkpoint storage backend
type CheckpointStore interface {
	// key must contain one or more characters in [A-Za-z0-9]
	// Write persists a checkpoint with key
	Write(key string, data []byte) error
	// Read retrieves a checkpoint with key
	Read(key string) ([]byte, error)
	// Delete deletes a checkpoint with key
	Delete(key string) error
	// List lists all keys of existing checkpoints
	List() ([]string, error)
}

// FileStore is an implementation of CheckpointStore interface which stores checkpoint in file.
type FileStore struct {
	path string
}

func (fstore *FileStore) Write(key string, data []byte) error {
	if !isValidKey(key) {
		return fmt.Errorf("Checkpoint key %q is not valid.", key)
	}
	if _, err := os.Stat(fstore.path); os.IsNotExist(err) {
		if err = os.MkdirAll(fstore.path, 0755); err != nil && !os.IsExist(err) {
			return err
		}
	}
	tmpfile := filepath.Join(fstore.path, fmt.Sprintf("%s%s%s", tmpPrefix, key, tmpSuffix))
	if err := ioutil.WriteFile(tmpfile, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpfile, filepath.Join(fstore.path, key))
}

func (fstore *FileStore) Read(key string) ([]byte, error) {
	return ioutil.ReadFile(filepath.Join(fstore.path, key))
}

func (fstore *FileStore) Delete(key string) error {
	return os.Remove(filepath.Join(fstore.path, key))
}

func (fstore *FileStore) List() ([]string, error) {
	keys := make([]string, 0)
	files, err := ioutil.ReadDir(fstore.path)
	if err != nil {
		return keys, err
	}
	for _, f := range files {
		if !strings.HasSuffix(f.Name(), tmpSuffix) {
			keys = append(keys, f.Name())
		}
	}
	return keys, nil
}

// MemStore is an implementation of CheckpointStore interface which stores checkpoint in memory.
type MemStore struct {
	mem map[string][]byte
	mu  sync.Mutex
}

func NewMemStore() CheckpointStore {
	return &MemStore{mem: make(map[string][]byte)}
}

func (mstore *MemStore) Write(key string, data []byte) error {
	mstore.mu.Lock()
	defer mstore.mu.Unlock()
	mstore.mem[key] = data
	return nil
}

func (mstore *MemStore) Read(key string) ([]byte, error) {
	mstore.mu.Lock()
	defer mstore.mu.Unlock()
	data, ok := mstore.mem[key]
	if !ok {
		return nil, fmt.Errorf("Sandbox %q Checkpoint could not be found", key)
	}
	return data, nil
}

func (mstore *MemStore) Delete(key string) error {
	mstore.mu.Lock()
	defer mstore.mu.Unlock()
	delete(mstore.mem, key)
	return nil
}

func (mstore *MemStore) List() ([]string, error) {
	mstore.mu.Lock()
	defer mstore.mu.Unlock()
	keys := make([]string, 0)
	for key := range mstore.mem {
		keys = append(keys, key)
	}
	return keys, nil
}

var regex = regexp.MustCompile(keyRegex)

func isValidKey(key string) bool {
	return len(key) <= keyMaxLength && regex.MatchString(key)
}
