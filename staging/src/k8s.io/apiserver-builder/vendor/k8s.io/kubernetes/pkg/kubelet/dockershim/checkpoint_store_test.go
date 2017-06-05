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
	"io/ioutil"
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/errors"
)

func TestFileStore(t *testing.T) {
	path, err := ioutil.TempDir("", "FileStore")
	assert.NoError(t, err)
	defer cleanUpTestPath(t, path)
	store, err := NewFileStore(path)
	assert.NoError(t, err)

	Checkpoints := []struct {
		key       string
		data      string
		expectErr bool
	}{
		{
			"id1",
			"data1",
			false,
		},
		{
			"id2",
			"data2",
			false,
		},
		{
			"/id1",
			"data1",
			true,
		},
		{
			".id1",
			"data1",
			true,
		},
		{
			"   ",
			"data2",
			true,
		},
		{
			"___",
			"data2",
			true,
		},
	}

	// Test Add Checkpoint
	for _, c := range Checkpoints {
		_, err = store.Read(c.key)
		assert.Error(t, err)

		err = store.Write(c.key, []byte(c.data))
		if c.expectErr {
			assert.Error(t, err)
			continue
		} else {
			assert.NoError(t, err)
		}

		// Test Read Checkpoint
		data, err := store.Read(c.key)
		assert.NoError(t, err)
		assert.Equal(t, string(data), c.data)
	}

	// Test list checkpoints.
	keys, err := store.List()
	assert.NoError(t, err)
	sort.Strings(keys)
	assert.Equal(t, keys, []string{"id1", "id2"})

	// Test Delete Checkpoint
	for _, c := range Checkpoints {
		if c.expectErr {
			continue
		}

		err = store.Delete(c.key)
		assert.NoError(t, err)
		_, err = store.Read(c.key)
		assert.EqualValues(t, errors.CheckpointNotFoundError, err)
	}

	// Test delete non existed checkpoint
	err = store.Delete("id1")
	assert.NoError(t, err)

	// Test list checkpoints.
	keys, err = store.List()
	assert.NoError(t, err)
	assert.Equal(t, len(keys), 0)
}

func TestIsValidKey(t *testing.T) {
	testcases := []struct {
		key   string
		valid bool
	}{
		{
			"    ",
			false,
		},
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
		if tc.valid {
			assert.NoError(t, validateKey(tc.key))
		} else {
			assert.Error(t, validateKey(tc.key))
		}
	}
}

func cleanUpTestPath(t *testing.T, path string) {
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		if err := os.RemoveAll(path); err != nil {
			assert.NoError(t, err, "Failed to delete test directory: %v", err)
		}
	}
}
