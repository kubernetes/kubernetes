// Copyright 2016 Google Inc. All Rights Reserved.
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

// Handler for "raw" containers.
package raw

import (
	"reflect"
	"testing"

	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
)

func TestFsToFsStats(t *testing.T) {
	inodes := uint64(100)
	inodesFree := uint64(50)
	testCases := map[string]struct {
		fs       *fs.Fs
		expected info.FsStats
	}{
		"has_inodes": {
			fs: &fs.Fs{
				DeviceInfo: fs.DeviceInfo{Device: "123"},
				Type:       fs.VFS,
				Capacity:   uint64(1024 * 1024),
				Free:       uint64(1024),
				Available:  uint64(1024),
				Inodes:     &inodes,
				InodesFree: &inodesFree,
				DiskStats: fs.DiskStats{
					ReadsCompleted:  uint64(100),
					ReadsMerged:     uint64(100),
					SectorsRead:     uint64(100),
					ReadTime:        uint64(100),
					WritesCompleted: uint64(100),
					WritesMerged:    uint64(100),
					SectorsWritten:  uint64(100),
					WriteTime:       uint64(100),
					IoInProgress:    uint64(100),
					IoTime:          uint64(100),
					WeightedIoTime:  uint64(100),
				},
			},
			expected: info.FsStats{
				Device:          "123",
				Type:            fs.VFS.String(),
				Limit:           uint64(1024 * 1024),
				Usage:           uint64(1024*1024) - uint64(1024),
				HasInodes:       true,
				Inodes:          inodes,
				InodesFree:      inodesFree,
				Available:       uint64(1024),
				ReadsCompleted:  uint64(100),
				ReadsMerged:     uint64(100),
				SectorsRead:     uint64(100),
				ReadTime:        uint64(100),
				WritesCompleted: uint64(100),
				WritesMerged:    uint64(100),
				SectorsWritten:  uint64(100),
				WriteTime:       uint64(100),
				IoInProgress:    uint64(100),
				IoTime:          uint64(100),
				WeightedIoTime:  uint64(100),
			},
		},
		"has_no_inodes": {
			fs: &fs.Fs{
				DeviceInfo: fs.DeviceInfo{Device: "123"},
				Type:       fs.DeviceMapper,
				Capacity:   uint64(1024 * 1024),
				Free:       uint64(1024),
				Available:  uint64(1024),
				DiskStats: fs.DiskStats{
					ReadsCompleted:  uint64(100),
					ReadsMerged:     uint64(100),
					SectorsRead:     uint64(100),
					ReadTime:        uint64(100),
					WritesCompleted: uint64(100),
					WritesMerged:    uint64(100),
					SectorsWritten:  uint64(100),
					WriteTime:       uint64(100),
					IoInProgress:    uint64(100),
					IoTime:          uint64(100),
					WeightedIoTime:  uint64(100),
				},
			},
			expected: info.FsStats{
				Device:          "123",
				Type:            fs.DeviceMapper.String(),
				Limit:           uint64(1024 * 1024),
				Usage:           uint64(1024*1024) - uint64(1024),
				HasInodes:       false,
				Available:       uint64(1024),
				ReadsCompleted:  uint64(100),
				ReadsMerged:     uint64(100),
				SectorsRead:     uint64(100),
				ReadTime:        uint64(100),
				WritesCompleted: uint64(100),
				WritesMerged:    uint64(100),
				SectorsWritten:  uint64(100),
				WriteTime:       uint64(100),
				IoInProgress:    uint64(100),
				IoTime:          uint64(100),
				WeightedIoTime:  uint64(100),
			},
		},
	}
	for testName, testCase := range testCases {
		actual := fsToFsStats(testCase.fs)
		if !reflect.DeepEqual(testCase.expected, actual) {
			t.Errorf("test case=%v, expected=%v, actual=%v", testName, testCase.expected, actual)
		}
	}
}
