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
	"os"
	"reflect"
	"testing"
)

const (
	testPath = "/tmp/testFileStore"
)

func TestFileStore(t *testing.T) {
	cleanUpTestPath(t)
	defer cleanUpTestPath(t)
	testCheckpointStore(t, &FileStore{path: testPath})
}

func TestMemStore(t *testing.T) {
	testCheckpointStore(t, NewMemStore())
}

func testCheckpointStore(t *testing.T, store CheckpointStore) {
	var err error
	Checkpoints := []struct {
		key  string
		data string
	}{
		{
			"id1",
			"data1",
		},
		{
			"id2",
			"data2",
		},
	}

	// Test Add Checkpoint
	for _, c := range Checkpoints {
		if _, err = store.Read(c.key); err == nil {
			t.Errorf("Expect an error when retriving key %q before adding checkpoint", c.key)
		}
		if err = store.Write(c.key, []byte(c.data)); err != nil {
			t.Errorf("Failed to add checkpoint %v: %v", c, err)
		}
		data, err := store.Read(c.key)
		if err != nil {
			t.Errorf("Failed to get checkpoint %v: %v", c, err)
		}
		if !(string(data) == c.data) {
			t.Errorf("Expecting checkpoint data to be %q, but got %q", c.data, string(data))
		}
	}

	keys, err := store.List()
	if err != nil {
		t.Errorf("Failed to list checkpoints: %v", err)
	}
	expect := []string{"id1", "id2"}
	if !reflect.DeepEqual(keys, expect) {
		t.Errorf("Expecting list of checkpoints to be %v, but got %v", expect, keys)
	}

	// Test Delete Checkpoint
	for _, c := range Checkpoints {
		if err = store.Delete(c.key); err != nil {
			t.Errorf("Failed to delete checkpoint %v: %v", c, err)
		}
		if _, err = store.Read(c.key); err == nil {
			t.Errorf("Expect an error when retriving key %q after deleting checkpoint", c.key)
		}
	}
	keys, err = store.List()
	if err != nil {
		t.Errorf("Failed to list checkpoints: %v", err)
	}
	if len(keys) > 0 {
		t.Errorf("Expect there is no checkpoint existed, but got: %v", keys)
	}
}

func cleanUpTestPath(t *testing.T) {
	if _, err := os.Stat(testPath); !os.IsNotExist(err) {
		if err := os.RemoveAll(testPath); err != nil {
			t.Errorf("Failed to delete test directory: %v", err)
		}
	}
	return
}

func TestIsValidKey(t *testing.T) {
	testcases := []struct {
		key   string
		valid bool
	}{
		{
			"/foo/bar",
			false,
		},
		{
			".foo",
			false,
		},
		{
			"a78768279290d33d0b82eaea43cb8346f500057cb5bd250e88c97a5585385d66",
			true,
		},
	}

	for _, tc := range testcases {
		if tc.valid != isValidKey(tc.key) {
			t.Errorf("Expect isValidKey(%q) to be %v, but got %v", tc.key, tc.valid, isValidKey(tc.key))
		}
	}
}
