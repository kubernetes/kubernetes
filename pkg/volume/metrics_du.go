/*
Copyright 2014 The Kubernetes Authors.

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

package volume

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume/util/fs"
)

var _ StatsProvider = &metricsDu{}

// metricsDu represents a StatsProvider that calculates the used and
// available Volume space by calling fs.DiskUsage() and gathering
// filesystem info for the Volume path.
type metricsDu struct {
	// the directory path the volume is mounted to.
	path string
}

// NewMetricsDu creates a new metricsDu with the Volume path.
func NewMetricsDu(path string) StatsProvider {
	return &metricsDu{path}
}

// GetStats calculates the volume usage and device free space by executing "du"
// and gathering filesystem info for the Volume path.
// See StatsProvider.GetStats
func (md *metricsDu) GetStats() (*Stats, error) {
	stats := &Stats{Time: metav1.Now()}
	if md.path == "" {
		return stats, NewNoPathDefinedError()
	}

	err := md.runDiskUsage(stats)
	if err != nil {
		return stats, err
	}

	err = md.runFind(stats)
	if err != nil {
		return stats, err
	}

	err = md.getFsInfo(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

// runDiskUsage gets disk usage of md.path and writes the results to stats.Used
func (md *metricsDu) runDiskUsage(stats *Stats) error {
	used, err := fs.DiskUsage(md.path)
	if err != nil {
		return err
	}
	stats.Used = used
	return nil
}

// runFind executes the "find" command and writes the results to stats.InodesUsed
func (md *metricsDu) runFind(stats *Stats) error {
	inodesUsed, err := fs.Find(md.path)
	if err != nil {
		return err
	}
	stats.InodesUsed = resource.NewQuantity(inodesUsed, resource.BinarySI)
	return nil
}

// getFsInfo writes stats.Capacity and stats.Available from the filesystem
// info
func (md *metricsDu) getFsInfo(stats *Stats) error {
	available, capacity, _, inodes, inodesFree, _, err := fs.Info(md.path)
	if err != nil {
		return NewFsInfoFailedError(err)
	}
	stats.Available = resource.NewQuantity(available, resource.BinarySI)
	stats.Capacity = resource.NewQuantity(capacity, resource.BinarySI)
	stats.Inodes = resource.NewQuantity(inodes, resource.BinarySI)
	stats.InodesFree = resource.NewQuantity(inodesFree, resource.BinarySI)
	return nil
}
