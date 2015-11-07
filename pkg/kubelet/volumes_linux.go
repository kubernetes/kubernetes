// +build linux

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"os"
	"path/filepath"
	"syscall"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

// Bitmask to OR with current ownership of volumes that allow ownership management by the Kubelet
const managedOwnershipBitmask = os.FileMode(0660)

// manageVolumeOwnership modifies the given volume to be owned by fsGroup.
func (kl *Kubelet) manageVolumeOwnership(pod *api.Pod, volSpec *volume.Spec, builder volume.Builder, fsGroup int64) error {
	return filepath.Walk(builder.GetPath(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		stat, ok := info.Sys().(*syscall.Stat_t)
		if !ok {
			return nil
		}

		if stat == nil {
			glog.Errorf("Got nil stat_t for path %v while managing ownership of volume %v for pod %s/%s", path, volSpec.Name, pod.Namespace, pod.Name)
			return nil
		}

		err = kl.chownRunner.Chown(path, int(stat.Uid), int(fsGroup))
		if err != nil {
			glog.Errorf("Chown failed on %v: %v", path, err)
		}

		err = kl.chmodRunner.Chmod(path, info.Mode()|managedOwnershipBitmask|os.ModeSetgid)
		if err != nil {
			glog.Errorf("Chmod failed on %v: %v", path, err)
		}

		return nil
	})
}
