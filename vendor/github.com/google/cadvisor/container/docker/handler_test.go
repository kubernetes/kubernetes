// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Handler for Docker containers.
package docker

import (
	"io/ioutil"
	"os"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStorageDirDetectionWithOldVersions(t *testing.T) {
	as := assert.New(t)
	rwLayer, err := getRwLayerID("abcd", "/", aufsStorageDriver, []int{1, 9, 0})
	as.Nil(err)
	as.Equal(rwLayer, "abcd")
}

func TestStorageDirDetectionWithNewVersions(t *testing.T) {
	as := assert.New(t)
	testDir, err := ioutil.TempDir("", "")
	as.Nil(err)
	containerID := "abcd"
	randomizedID := "xyz"
	randomIDPath := path.Join(testDir, "image/aufs/layerdb/mounts/", containerID)
	as.Nil(os.MkdirAll(randomIDPath, os.ModePerm))
	as.Nil(ioutil.WriteFile(path.Join(randomIDPath, "mount-id"), []byte(randomizedID), os.ModePerm))
	rwLayer, err := getRwLayerID(containerID, testDir, "aufs", []int{1, 10, 0})
	as.Nil(err)
	as.Equal(rwLayer, randomizedID)
	rwLayer, err = getRwLayerID(containerID, testDir, "aufs", []int{1, 10, 0})
	as.Nil(err)
	as.Equal(rwLayer, randomizedID)

}
