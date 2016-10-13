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
	"errors"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

func getManager(plugin *cinderPlugin, spec *volume.Spec, options *volume.VolumeOptions) (cinderManager, error) {
	if spec != nil {
		cinder, _, err := getVolumeSource(spec)
		if err != nil {
			return nil, err
		}
		if cinder.SecretRef != nil && cinder.SecretRef.Name != "" {
			return &cdManagerLocal{plugin: plugin}, nil
		}
	}

	if options != nil {
		for k := range options.Parameters {
			if strings.ToLower(k) == "secretref" {
				return &cdManagerLocal{plugin: plugin}, nil
			}
		}
	}

	return &cdManagerCloud{plugin: plugin}, nil
}

func getVolumeSource(spec *volume.Spec) (*v1.CinderVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Cinder != nil {
		return spec.Volume.Cinder, spec.Volume.Cinder.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Cinder != nil {
		return spec.PersistentVolume.Spec.Cinder, spec.ReadOnly, nil
	}

	return nil, false, errors.New("Spec does not reference a Cinder volume type")
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(cinderVolumePluginName), mount.MountsInGlobalPDPath, devName)
}
