// Copyright 2015 Google Inc. All Rights Reserved.
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

package libcontainer

import (
	"path/filepath"
	"testing"
)

func TestReadConfig(t *testing.T) {
	var (
		testdata    string = "testdata"
		containerID string = "1"
	)
	// Test with using the new config of docker v1.9.0
	dockerRoot := filepath.Join(testdata, "docker-v1.9.1")
	dockerRun := filepath.Join(testdata, "docker-v1.9.1")
	config, err := ReadConfig(dockerRoot, dockerRun, containerID)
	if err != nil {
		t.Error(err)
	}
	if config.Hostname != containerID {
		t.Errorf("Expected container hostname is %s, but got %s", containerID, config.Hostname)
	}

	// Test with using the pre config of docker v1.8.3
	dockerRoot = filepath.Join(testdata, "docker-v1.8.3")
	dockerRun = filepath.Join(testdata, "docker-v1.8.3")
	config, err = ReadConfig(dockerRoot, dockerRun, containerID)
	if err != nil {
		t.Error(err)
	}
	if config.Hostname != containerID {
		t.Errorf("Expected container hostname is %s, but got %s", containerID, config.Hostname)
	}

	// Test with using non-existed old config, return an error
	dockerRoot = filepath.Join(testdata, "docker-v1.8.0")
	dockerRun = filepath.Join(testdata, "docker-v1.8.0")
	config, err = ReadConfig(dockerRoot, dockerRun, containerID)
	if err == nil {
		t.Error("Expected an error, but got nil")
	}
}
