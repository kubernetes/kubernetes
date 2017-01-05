/*
Copyright 2017 The Kubernetes Authors.

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

package flexvolume

import (
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
)

type flexVolume struct {
	// driverName is the name of the plugin driverName.
	driverName string
	// Driver executable used to setup the volume.
	execPath string
	// mounter provides the interface that is used to mount the actual
	// block device.
	mounter mount.Interface
	// podName is the name of the pod, if available.
	podName string
	// podUID is the UID of the pod.
	podUID types.UID
	// podNamespace is the namespace of the pod, if available.
	podNamespace string
	// podServiceAccountName is the service account name of the pod, if available.
	podServiceAccountName string
	// volName is the name of the pod's volume.
	volName string
	// the underlying plugin
	plugin *flexVolumePlugin
}

// volume.Volume interface

func (f *flexVolume) GetPath() string {
	name := f.driverName
	return f.plugin.host.GetPodVolumeDir(f.podUID, utilstrings.EscapeQualifiedNameForDisk(name), f.volName)
}
