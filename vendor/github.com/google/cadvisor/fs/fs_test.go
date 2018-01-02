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
	"errors"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/docker/docker/pkg/mount"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestDirDiskUsage(t *testing.T) {
	as := assert.New(t)
	fsInfo, err := NewFsInfo(Context{})
	as.NoError(err)
	dir, err := ioutil.TempDir(os.TempDir(), "")
	as.NoError(err)
	defer os.RemoveAll(dir)
	dataSize := 1024 * 100 //100 KB
	b := make([]byte, dataSize)
	f, err := ioutil.TempFile(dir, "")
	as.NoError(err)
	as.NoError(ioutil.WriteFile(f.Name(), b, 0700))
	fi, err := f.Stat()
	as.NoError(err)
	expectedSize := uint64(fi.Size())
	size, err := fsInfo.GetDirDiskUsage(dir, time.Minute)
	as.NoError(err)
	as.True(expectedSize <= size, "expected dir size to be at-least %d; got size: %d", expectedSize, size)
}

func TestDirInodeUsage(t *testing.T) {
	as := assert.New(t)
	fsInfo, err := NewFsInfo(Context{})
	as.NoError(err)
	dir, err := ioutil.TempDir(os.TempDir(), "")
	as.NoError(err)
	defer os.RemoveAll(dir)
	numFiles := 1000
	for i := 0; i < numFiles; i++ {
		_, err := ioutil.TempFile(dir, "")
		require.NoError(t, err)
	}
	inodes, err := fsInfo.GetDirInodeUsage(dir, time.Minute)
	as.NoError(err)
	// We sould get numFiles+1 inodes, since we get 1 inode for each file, plus 1 for the directory
	as.True(uint64(numFiles+1) == inodes, "expected inodes in dir to be %d; got inodes: %d", numFiles+1, inodes)
}

var dmStatusTests = []struct {
	dmStatus    string
	used        uint64
	total       uint64
	errExpected bool
}{
	{`0 409534464 thin-pool 64085 3705/4161600 88106/3199488 - rw no_discard_passdown queue_if_no_space -`, 88106, 3199488, false},
	{`0 209715200 thin-pool 707 1215/524288 30282/1638400 - rw discard_passdown`, 30282, 1638400, false},
	{`Invalid status line`, 0, 0, false},
}

func TestParseDMStatus(t *testing.T) {
	for _, tt := range dmStatusTests {
		used, total, err := parseDMStatus(tt.dmStatus)
		if tt.errExpected && err != nil {
			t.Errorf("parseDMStatus(%q) expected error", tt.dmStatus)
		}
		if used != tt.used {
			t.Errorf("parseDMStatus(%q) wrong used value => %q, want %q", tt.dmStatus, used, tt.used)
		}
		if total != tt.total {
			t.Errorf("parseDMStatus(%q) wrong total value => %q, want %q", tt.dmStatus, total, tt.total)
		}
	}
}

var dmTableTests = []struct {
	dmTable     string
	major       uint
	minor       uint
	dataBlkSize uint
	errExpected bool
}{
	{`0 409534464 thin-pool 253:6 253:7 128 32768 1 skip_block_zeroing`, 253, 7, 128, false},
	{`0 409534464 thin-pool 253:6 258:9 512 32768 1 skip_block_zeroing otherstuff`, 258, 9, 512, false},
	{`Invalid status line`, 0, 0, 0, false},
}

func TestParseDMTable(t *testing.T) {
	for _, tt := range dmTableTests {
		major, minor, dataBlkSize, err := parseDMTable(tt.dmTable)
		if tt.errExpected && err != nil {
			t.Errorf("parseDMTable(%q) expected error", tt.dmTable)
		}
		if major != tt.major {
			t.Errorf("parseDMTable(%q) wrong major value => %q, want %q", tt.dmTable, major, tt.major)
		}
		if minor != tt.minor {
			t.Errorf("parseDMTable(%q) wrong minor value => %q, want %q", tt.dmTable, minor, tt.minor)
		}
		if dataBlkSize != tt.dataBlkSize {
			t.Errorf("parseDMTable(%q) wrong dataBlkSize value => %q, want %q", tt.dmTable, dataBlkSize, tt.dataBlkSize)
		}
	}
}

func TestAddSystemRootLabel(t *testing.T) {
	tests := []struct {
		mounts   []*mount.Info
		expected string
	}{
		{
			mounts: []*mount.Info{
				{Source: "/dev/sda1", Mountpoint: "/foo"},
				{Source: "/dev/sdb1", Mountpoint: "/"},
			},
			expected: "/dev/sdb1",
		},
	}

	for i, tt := range tests {
		fsInfo := &RealFsInfo{
			labels:     map[string]string{},
			partitions: map[string]partition{},
		}
		fsInfo.addSystemRootLabel(tt.mounts)

		if source, ok := fsInfo.labels[LabelSystemRoot]; !ok || source != tt.expected {
			t.Errorf("case %d: expected mount source '%s', got '%s'", i, tt.expected, source)
		}
	}
}

type testDmsetup struct {
	data []byte
	err  error
}

func (*testDmsetup) Message(deviceName string, sector int, message string) ([]byte, error) {
	return nil, nil
}

func (*testDmsetup) Status(deviceName string) ([]byte, error) {
	return nil, nil
}

func (t *testDmsetup) Table(poolName string) ([]byte, error) {
	return t.data, t.err
}

func TestGetDockerDeviceMapperInfo(t *testing.T) {
	tests := []struct {
		name              string
		driver            string
		driverStatus      map[string]string
		dmsetupTable      string
		dmsetupTableError error
		expectedDevice    string
		expectedPartition *partition
		expectedError     bool
	}{
		{
			name:              "not devicemapper",
			driver:            "btrfs",
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     false,
		},
		{
			name:              "nil driver status",
			driver:            "devicemapper",
			driverStatus:      nil,
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     true,
		},
		{
			name:              "loopback",
			driver:            "devicemapper",
			driverStatus:      map[string]string{"Data loop file": "/var/lib/docker/devicemapper/devicemapper/data"},
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     false,
		},
		{
			name:              "missing pool name",
			driver:            "devicemapper",
			driverStatus:      map[string]string{},
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     true,
		},
		{
			name:              "error invoking dmsetup",
			driver:            "devicemapper",
			driverStatus:      map[string]string{"Pool Name": "vg_vagrant-docker--pool"},
			dmsetupTableError: errors.New("foo"),
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     true,
		},
		{
			name:              "unable to parse dmsetup table",
			driver:            "devicemapper",
			driverStatus:      map[string]string{"Pool Name": "vg_vagrant-docker--pool"},
			dmsetupTable:      "no data here!",
			expectedDevice:    "",
			expectedPartition: nil,
			expectedError:     true,
		},
		{
			name:           "happy path",
			driver:         "devicemapper",
			driverStatus:   map[string]string{"Pool Name": "vg_vagrant-docker--pool"},
			dmsetupTable:   "0 53870592 thin-pool 253:2 253:3 1024 0 1 skip_block_zeroing",
			expectedDevice: "vg_vagrant-docker--pool",
			expectedPartition: &partition{
				fsType:    "devicemapper",
				major:     253,
				minor:     3,
				blockSize: 1024,
			},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		fsInfo := &RealFsInfo{
			dmsetup: &testDmsetup{
				data: []byte(tt.dmsetupTable),
			},
		}

		dockerCtx := DockerContext{
			Driver:       tt.driver,
			DriverStatus: tt.driverStatus,
		}

		device, partition, err := fsInfo.getDockerDeviceMapperInfo(dockerCtx)

		if tt.expectedError && err == nil {
			t.Errorf("%s: expected error but got nil", tt.name)
			continue
		}
		if !tt.expectedError && err != nil {
			t.Errorf("%s: unexpected error: %v", tt.name, err)
			continue
		}

		if e, a := tt.expectedDevice, device; e != a {
			t.Errorf("%s: device: expected %q, got %q", tt.name, e, a)
		}

		if e, a := tt.expectedPartition, partition; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: partition: expected %#v, got %#v", tt.name, e, a)
		}
	}
}

func TestAddDockerImagesLabel(t *testing.T) {
	tests := []struct {
		name                           string
		driver                         string
		driverStatus                   map[string]string
		dmsetupTable                   string
		getDockerDeviceMapperInfoError error
		mounts                         []*mount.Info
		expectedDockerDevice           string
		expectedPartition              *partition
	}{
		{
			name:         "devicemapper, not loopback",
			driver:       "devicemapper",
			driverStatus: map[string]string{"Pool Name": "vg_vagrant-docker--pool"},
			dmsetupTable: "0 53870592 thin-pool 253:2 253:3 1024 0 1 skip_block_zeroing",
			mounts: []*mount.Info{
				{
					Source:     "/dev/mapper/vg_vagrant-lv_root",
					Mountpoint: "/",
					Fstype:     "devicemapper",
				},
			},
			expectedDockerDevice: "vg_vagrant-docker--pool",
			expectedPartition: &partition{
				fsType:    "devicemapper",
				major:     253,
				minor:     3,
				blockSize: 1024,
			},
		},
		{
			name:         "devicemapper, loopback on non-root partition",
			driver:       "devicemapper",
			driverStatus: map[string]string{"Data loop file": "/var/lib/docker/devicemapper/devicemapper/data"},
			mounts: []*mount.Info{
				{
					Source:     "/dev/mapper/vg_vagrant-lv_root",
					Mountpoint: "/",
					Fstype:     "devicemapper",
				},
				{
					Source:     "/dev/sdb1",
					Mountpoint: "/var/lib/docker/devicemapper",
				},
			},
			expectedDockerDevice: "/dev/sdb1",
		},
		{
			name: "multiple mounts - innermost check",
			mounts: []*mount.Info{
				{
					Source:     "/dev/sda1",
					Mountpoint: "/",
					Fstype:     "ext4",
				},
				{
					Source:     "/dev/sdb1",
					Mountpoint: "/var/lib/docker",
					Fstype:     "ext4",
				},
				{
					Source:     "/dev/sdb2",
					Mountpoint: "/var/lib/docker/btrfs",
					Fstype:     "btrfs",
				},
			},
			expectedDockerDevice: "/dev/sdb2",
		},
		{
			name: "root fs inside container, docker-images bindmount",
			mounts: []*mount.Info{
				{
					Source:     "overlay",
					Mountpoint: "/",
					Fstype:     "overlay",
				},
				{
					Source:     "/dev/sda1",
					Mountpoint: "/var/lib/docker",
					Fstype:     "ext4",
				},
			},
			expectedDockerDevice: "/dev/sda1",
		},
		{
			name: "[overlay2] root fs inside container - /var/lib/docker bindmount",
			mounts: []*mount.Info{
				{
					Source:     "overlay",
					Mountpoint: "/",
					Fstype:     "overlay",
				},
				{
					Source:     "/dev/sdb1",
					Mountpoint: "/var/lib/docker",
					Fstype:     "ext4",
				},
				{
					Source:     "/dev/sdb2",
					Mountpoint: "/var/lib/docker/overlay2",
					Fstype:     "ext4",
				},
			},
			expectedDockerDevice: "/dev/sdb2",
		},
	}

	for _, tt := range tests {
		fsInfo := &RealFsInfo{
			labels:     map[string]string{},
			partitions: map[string]partition{},
			dmsetup: &testDmsetup{
				data: []byte(tt.dmsetupTable),
			},
		}

		context := Context{
			Docker: DockerContext{
				Root:         "/var/lib/docker",
				Driver:       tt.driver,
				DriverStatus: tt.driverStatus,
			},
		}

		fsInfo.addDockerImagesLabel(context, tt.mounts)

		if e, a := tt.expectedDockerDevice, fsInfo.labels[LabelDockerImages]; e != a {
			t.Errorf("%s: docker device: expected %q, got %q", tt.name, e, a)
		}

		if tt.expectedPartition == nil {
			continue
		}
		if e, a := *tt.expectedPartition, fsInfo.partitions[tt.expectedDockerDevice]; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: docker partition: expected %#v, got %#v", tt.name, e, a)
		}
	}
}

func TestProcessMounts(t *testing.T) {
	tests := []struct {
		name             string
		mounts           []*mount.Info
		excludedPrefixes []string
		expected         map[string]partition
	}{
		{
			name: "unsupported fs types",
			mounts: []*mount.Info{
				{Fstype: "overlay"},
				{Fstype: "somethingelse"},
			},
			expected: map[string]partition{},
		},
		{
			name: "avoid bind mounts",
			mounts: []*mount.Info{
				{Root: "/", Mountpoint: "/", Source: "/dev/sda1", Fstype: "xfs", Major: 253, Minor: 0},
				{Root: "/foo", Mountpoint: "/bar", Source: "/dev/sda1", Fstype: "xfs", Major: 253, Minor: 0},
			},
			expected: map[string]partition{
				"/dev/sda1": {fsType: "xfs", mountpoint: "/", major: 253, minor: 0},
			},
		},
		{
			name: "exclude prefixes",
			mounts: []*mount.Info{
				{Root: "/", Mountpoint: "/someother", Source: "/dev/sda1", Fstype: "xfs", Major: 253, Minor: 2},
				{Root: "/", Mountpoint: "/", Source: "/dev/sda2", Fstype: "xfs", Major: 253, Minor: 0},
				{Root: "/", Mountpoint: "/excludeme", Source: "/dev/sda3", Fstype: "xfs", Major: 253, Minor: 1},
			},
			excludedPrefixes: []string{"/exclude", "/some"},
			expected: map[string]partition{
				"/dev/sda2": {fsType: "xfs", mountpoint: "/", major: 253, minor: 0},
			},
		},
		{
			name: "supported fs types",
			mounts: []*mount.Info{
				{Root: "/", Mountpoint: "/a", Source: "/dev/sda", Fstype: "ext3", Major: 253, Minor: 0},
				{Root: "/", Mountpoint: "/b", Source: "/dev/sdb", Fstype: "ext4", Major: 253, Minor: 1},
				{Root: "/", Mountpoint: "/c", Source: "/dev/sdc", Fstype: "btrfs", Major: 253, Minor: 2},
				{Root: "/", Mountpoint: "/d", Source: "/dev/sdd", Fstype: "xfs", Major: 253, Minor: 3},
				{Root: "/", Mountpoint: "/e", Source: "/dev/sde", Fstype: "zfs", Major: 253, Minor: 4},
			},
			expected: map[string]partition{
				"/dev/sda": {fsType: "ext3", mountpoint: "/a", major: 253, minor: 0},
				"/dev/sdb": {fsType: "ext4", mountpoint: "/b", major: 253, minor: 1},
				"/dev/sdc": {fsType: "btrfs", mountpoint: "/c", major: 253, minor: 2},
				"/dev/sdd": {fsType: "xfs", mountpoint: "/d", major: 253, minor: 3},
				"/dev/sde": {fsType: "zfs", mountpoint: "/e", major: 253, minor: 4},
			},
		},
	}

	for _, test := range tests {
		actual := processMounts(test.mounts, test.excludedPrefixes)
		if !reflect.DeepEqual(test.expected, actual) {
			t.Errorf("%s: expected %#v, got %#v", test.name, test.expected, actual)
		}
	}
}
