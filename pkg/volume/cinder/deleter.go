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

package cinder

import (
	"fmt"

	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

type cinderVolumeDeleter struct {
	*cinderVolume
}

func newDeleter(spec *volume.Spec, plugin *cinderPlugin, manager cinderManager) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Cinder == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Cinder is nil")
	}
	return &cinderVolumeDeleter{
		&cinderVolume{
			volName:   spec.Name(),
			pdName:    spec.PersistentVolume.Spec.Cinder.VolumeID,
			secretRef: spec.PersistentVolume.Spec.Cinder.SecretRef,
			manager:   manager,
			plugin:    plugin,
		},
	}, nil
}

func (r *cinderVolumeDeleter) GetPath() string {
	name := cinderVolumePluginName
	return r.plugin.host.GetPodVolumeDir(r.podUID, kstrings.EscapeQualifiedNameForDisk(name), r.volName)
}

func (r *cinderVolumeDeleter) Delete() error {
	if err := r.manager.DeleteVolume(r); err != nil {
		return err
	}
	return nil
}
