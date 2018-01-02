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

package common

import (
	"testing"
)

func TestGetContainerHintsFromFile(t *testing.T) {
	cHints, err := GetContainerHintsFromFile("test_resources/container_hints.json")

	if err != nil {
		t.Fatalf("Error in unmarshalling: %s", err)
	}

	if cHints.AllHosts[0].NetworkInterface.VethHost != "veth24031eth1" &&
		cHints.AllHosts[0].NetworkInterface.VethChild != "eth1" {
		t.Errorf("Cannot find network interface in %s", cHints)
	}

	correctMountDirs := [...]string{
		"/var/run/nm-sdc1",
		"/var/run/nm-sdb3",
		"/var/run/nm-sda3",
		"/var/run/netns/root",
		"/var/run/openvswitch/db.sock",
	}

	if len(cHints.AllHosts[0].Mounts) == 0 {
		t.Errorf("Cannot find any mounts")
	}

	for i, mountDir := range cHints.AllHosts[0].Mounts {
		if correctMountDirs[i] != mountDir.HostDir {
			t.Errorf("Cannot find mount %s in %s", mountDir.HostDir, cHints)
		}
	}
}

func TestFileNotExist(t *testing.T) {
	_, err := GetContainerHintsFromFile("/file_does_not_exist.json")
	if err != nil {
		t.Fatalf("GetContainerHintsFromFile must not error for blank file: %s", err)
	}
}
