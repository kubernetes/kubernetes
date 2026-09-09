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

package testing

import (
	"fmt"
	"sync"
)

// MemStore is an implementation of CheckpointStore interface which stores checkpoint in memory.
type MemStore struct {
	mem map[string][]byte
	sync.Mutex
}

// NewMemStore returns an instance of MemStore
func NewMemStore() *MemStore {
	return &MemStore{mem: make(map[string][]byte)}
}

// Write writes the data to the store
func (mstore *MemStore) Write(key string, data []byte) error {
	mstore.Lock()
	defer mstore.Unlock()
	mstore.mem[key] = data
	return nil
}

// Read returns data read from store
func (mstore *MemStore) Read(key string) ([]byte, error) {
	mstore.Lock()
	defer mstore.Unlock()
	data, ok := mstore.mem[key]
	if !ok {
		return nil, fmt.Errorf("checkpoint is not found")
	}
	return data, nil
}

// Delete deletes data from the store
func (mstore *MemStore) Delete(key string) error {
	mstore.Lock()
	defer mstore.Unlock()
	delete(mstore.mem, key)
	return nil
}

// List returns all the keys from the store
func (mstore *MemStore) List() ([]string, error) {
	mstore.Lock()
	defer mstore.Unlock()
	keys := make([]string, 0)
	for key := range mstore.mem {
		keys = append(keys, key)
	}
	return keys, nil
}
