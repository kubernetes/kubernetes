/*
Copyright 2016 The Kubernetes Authors.

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

package gce_pd

import (
	"github.com/golang/glog"
	"google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/volume"
)

var _ volume.SnapshottableVolumePlugin = &gcePersistentDiskPlugin{}

// CreateSnapshot issues a command to the GCE cloud provider to create a
// snapshot with the given name.
// Callers are responsible for retrying on failure.
// Callers are responsible for thread safety between concurrent snapshot operations.
func (plugin *gcePersistentDiskPlugin) CreateSnapshot(spec *Spec, snapshotName string) (string, error) {
	gceCloud, err := getCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return "", err
	}
	snapshot := compute.Snapshot{
		Name: snapshotName,
	}

	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	pdName := volumeSource.PDName

	if err := gceCloud.CreateSnapshot(pdName, &snapshot); err != nil {
		glog.Errorf("Error creating snapshot (name: %q) of PD %q: %+v", snapshotName, pdName, err)
		return "", err
	}

	return snapshot.CreationTimestamp, nil
}
