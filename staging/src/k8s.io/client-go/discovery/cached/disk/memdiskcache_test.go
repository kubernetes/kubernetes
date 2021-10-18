/*
Copyright 2021 The Kubernetes Authors.

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

package disk

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/peterbourgon/diskv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemDiskCache_SetGetDeleteGetSetGet(t *testing.T) {
	for _, readonly := range []bool{false, true} {
		t.Run(fmt.Sprintf("readonly_%t", readonly), func(t *testing.T) {
			c := setupMemDiskCache(t, readonly)

			c.Set("key", []byte{1, 2, 3})

			val, found := c.Get("key")
			require.True(t, found)
			assert.Equal(t, []byte{1, 2, 3}, val)

			c.Delete("key")

			_, found = c.Get("key")
			assert.False(t, found)

			c.Set("key", []byte{3, 2, 1})

			val, found = c.Get("key")
			require.True(t, found)
			assert.Equal(t, []byte{3, 2, 1}, val)
		})
	}
}

func TestMemDiskCache_SetSetGet(t *testing.T) {
	for _, readonly := range []bool{false, true} {
		t.Run(fmt.Sprintf("readonly_%t", readonly), func(t *testing.T) {
			c := setupMemDiskCache(t, readonly)

			c.Set("key", []byte{1, 2, 3})
			c.Set("key", []byte{3, 2, 1})

			val, found := c.Get("key")
			require.True(t, found)
			assert.Equal(t, []byte{3, 2, 1}, val)
		})
	}
}

func setupMemDiskCache(t *testing.T, readonly bool) *memDiskCache {
	cacheDir := setupTempDir(t, readonly)
	return newMemDiskCache(diskv.New(diskv.Options{
		PathPerm: os.FileMode(0750),
		FilePerm: os.FileMode(0660),
		BasePath: cacheDir,
		TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
	}))
}

func setupTempDir(t *testing.T, readonly bool) string {
	cacheDir, err := ioutil.TempDir("", "")
	require.NoError(t, err)
	t.Cleanup(func() {
		if readonly {
			assert.NoError(t, os.Chmod(cacheDir, 0700))
		}
		assert.NoError(t, os.RemoveAll(cacheDir))
	})
	if readonly {
		assert.NoError(t, os.Chmod(cacheDir, 0000))
	}
	return cacheDir
}
