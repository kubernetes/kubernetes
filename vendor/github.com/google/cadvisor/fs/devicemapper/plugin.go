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

package devicemapper

import (
	"github.com/google/cadvisor/fs"

	mount "github.com/moby/sys/mountinfo"
	"k8s.io/klog/v2"
)

type dmPlugin struct{}

// NewPlugin creates a new DeviceMapper filesystem plugin.
func NewPlugin() fs.FsPlugin {
	return &dmPlugin{}
}

func (p *dmPlugin) Name() string {
	return "devicemapper"
}

// CanHandle returns true if the filesystem type is devicemapper.
func (p *dmPlugin) CanHandle(fsType string) bool {
	return fsType == "devicemapper"
}

// Priority returns 100 - DeviceMapper has higher priority than VFS.
func (p *dmPlugin) Priority() int {
	return 100
}

// GetStats returns filesystem statistics for DeviceMapper thin provisioning.
func (p *dmPlugin) GetStats(device string, partition fs.PartitionInfo) (*fs.FsStats, error) {
	capacity, free, avail, err := GetDMStats(device, partition.BlockSize)
	if err != nil {
		return nil, err
	}

	klog.V(5).Infof("got devicemapper fs capacity stats: capacity: %v free: %v available: %v", capacity, free, avail)

	return &fs.FsStats{
		Capacity:  capacity,
		Free:      free,
		Available: avail,
		Type:      fs.DeviceMapper,
	}, nil
}

// ProcessMount handles DeviceMapper mount processing.
// For DeviceMapper, no special processing is needed.
func (p *dmPlugin) ProcessMount(mnt *mount.Info) (bool, *mount.Info, error) {
	return true, mnt, nil
}
