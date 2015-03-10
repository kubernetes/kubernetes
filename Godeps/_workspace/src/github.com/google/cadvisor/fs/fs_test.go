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

package fs

import (
	"testing"
)

func TestGetDiskStatsMap(t *testing.T) {
	diskStatsMap, err := getDiskStatsMap("test_resources/diskstats")
	if err != nil {
		t.Errorf("Error calling getDiskStatMap %s", err)
	}
	if len(diskStatsMap) != 30 {
		t.Errorf("diskStatsMap %+v not valid", diskStatsMap)
	}
	keySet := map[string]string{
		"/dev/sda":  "/dev/sda",
		"/dev/sdb":  "/dev/sdb",
		"/dev/sdc":  "/dev/sdc",
		"/dev/sdd":  "/dev/sdd",
		"/dev/sde":  "/dev/sde",
		"/dev/sdf":  "/dev/sdf",
		"/dev/sdg":  "/dev/sdg",
		"/dev/sdh":  "/dev/sdh",
		"/dev/sdb1": "/dev/sdb1",
		"/dev/sdb2": "/dev/sdb2",
		"/dev/sda1": "/dev/sda1",
		"/dev/sda2": "/dev/sda2",
		"/dev/sdc1": "/dev/sdc1",
		"/dev/sdc2": "/dev/sdc2",
		"/dev/sdc3": "/dev/sdc3",
		"/dev/sdc4": "/dev/sdc4",
		"/dev/sdd1": "/dev/sdd1",
		"/dev/sdd2": "/dev/sdd2",
		"/dev/sdd3": "/dev/sdd3",
		"/dev/sdd4": "/dev/sdd4",
		"/dev/sde1": "/dev/sde1",
		"/dev/sde2": "/dev/sde2",
		"/dev/sdf1": "/dev/sdf1",
		"/dev/sdf2": "/dev/sdf2",
		"/dev/sdg1": "/dev/sdg1",
		"/dev/sdg2": "/dev/sdg2",
		"/dev/sdh1": "/dev/sdh1",
		"/dev/sdh2": "/dev/sdh2",
		"/dev/dm-0": "/dev/dm-0",
		"/dev/dm-1": "/dev/dm-1",
	}

	for device := range diskStatsMap {
		if _, ok := keySet[device]; !ok {
			t.Errorf("Cannot find device %s", device)
		}
		delete(keySet, device)
	}
	if len(keySet) != 0 {
		t.Errorf("diskStatsMap %+v contains illegal keys %+v", diskStatsMap, keySet)
	}
}

func TestFileNotExist(t *testing.T) {
	_, err := getDiskStatsMap("/file_does_not_exist")
	if err != nil {
		t.Fatalf("getDiskStatsMap must not error for absent file: %s", err)
	}
}
