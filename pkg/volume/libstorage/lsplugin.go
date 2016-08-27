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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	lsPluginName = "kubernetes.io/libstorage"
)

type lsPlugin struct {
	host volume.VolumeHost
}

// func ProbeVolumePlugins(opts string) []volume.VolumePlugin {
// 	p := &lsPlugin{
// 		host: nil,
// 	}
// 	return []volume.VolumePlugin{p}
// }

func (p *lsPlugin) Init(host volume.VolumeHost) error {
	p.host = host
	return nil
}

func (p *lsPlugin) GetPluginName() string {
	return lsPluginName
}

func (p *lsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	source, err := p.getLibStorageSource(spec)
	if err != nil {
		return "", err
	}
	return source.VolumeName, nil
}

func (p *lsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.LibStorage != nil) ||
		(spec.Volume != nil && spec.Volume.LibStorage != nil)
}

func (plugin *lsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	lsVolumeSpec := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			LibStorage: &api.LibStorageVolumeSource{
				VolumeName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(lsVolumeSpec), nil
}

// func (p *lsPlugin) NewMounter(
// 	spec *volume.Spec,
// 	pod *api.Pod,
// 	_ volume.VolumeOptions) (volume.Mounter, error) {
// 	lsVol, err := p.getLibStorageSource(spec)
// 	if err != nil {
// 		return nil, err
// 	}
// 	volName := lsVol.VolumeName
//
// 	return &lsVolume{
// 		volName:  lsVol.VolumeName,
// 		podUID:   pod.UID,
// 		mounter:  mount.New(),
// 		plugin:   p,
// 		readOnly: spec.ReadOnly,
// 		k8mtx:    keymutex.NewKeyMutex(),
// 	}, nil
// }

// Helper methods

func (p *lsPlugin) getLibStorageSource(spec *volume.Spec) (*api.LibStorageVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.LibStorage != nil {
		return spec.Volume.LibStorage, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.LibStorage != nil {
		return spec.PersistentVolume.Spec.LibStorage, nil
	}

	return nil, fmt.Errorf("LibStorage is not found in spec")
}
