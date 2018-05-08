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
	"time"

	cadvisor "github.com/google/cadvisor/fs"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume/util/fs"
)

const timeout = time.Minute

var _ MetricsProvider = &metricsDu{}

// metricsDu represents a MetricsProvider that calculates the used and
// available Volume space by executing the "du" command and gathering
// filesystem info for the Volume path.
type metricsDu struct {
	// the directory path the volume is mounted to.
	path string
}

// NewMetricsDu creates a new metricsDu with the Volume path.
func NewMetricsDu(path string) MetricsProvider {
	return &metricsDu{path}
}

// GetMetrics calculates the volume usage and device free space by executing "du"
// and gathering filesystem info for the Volume path.
// See MetricsProvider.GetMetrics
func (md *metricsDu) GetMetrics() (*Metrics, error) {
	metrics := &Metrics{Time: metav1.Now()}
	if md.path == "" {
		return metrics, NewNoPathDefinedError()
	}

	used, err := cadvisor.GetDirDiskUsage(md.path, timeout)
	if err != nil {
		return metrics, err
	}

	inodesUsed, err := cadvisor.GetDirInodeUsage(md.path, timeout)
	if err != nil {
		return metrics, err
	}

	available, capacity, _, inodes, inodesFree, _, err := fs.FsInfo(md.path)
	if err != nil {
		return metrics, NewFsInfoFailedError(err)
	}

	metrics.Available = resource.NewQuantity(available, resource.BinarySI)
	metrics.Capacity = resource.NewQuantity(capacity, resource.BinarySI)
	metrics.Used = resource.NewQuantity(int64(used), resource.BinarySI)
	metrics.Inodes = resource.NewQuantity(inodes, resource.BinarySI)
	metrics.InodesFree = resource.NewQuantity(inodesFree, resource.BinarySI)
	metrics.InodesUsed = resource.NewQuantity(int64(inodesUsed), resource.BinarySI)

	return metrics, nil
}
