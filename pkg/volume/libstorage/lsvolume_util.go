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

package libstorage

import (
	"fmt"
	"path"

	"github.com/golang/glog"
	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// returns peristent disk path used for mounts
func (m *lsVolume) makePDPath(service, volName string) string {
	return path.Join(
		m.plugin.host.GetPluginDir(lsPluginName),
		"mounts",
		fmt.Sprintf("%s/%s", service, volName))
}

func (m *lsVolume) verifyDevicePath(path string) (string, error) {
	if pathExists, err := volutil.PathExists(path); err != nil {
		glog.Errorf("libStorage: failed to verify device path: %v", err)
		return "", err
	} else if pathExists {
		return path, nil
	}
	return "", nil
}

func getLibStorageSource(spec *volume.Spec) (*api.LibStorageVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.LibStorage != nil {
		return spec.Volume.LibStorage, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.LibStorage != nil {
		return spec.PersistentVolume.Spec.LibStorage, nil
	}

	return nil, fmt.Errorf("LibStorage is not found in spec")
}
