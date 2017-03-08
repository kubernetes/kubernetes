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

package validation

import (
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/probeplugins"
)

var volumePlugins []volume.VolumePlugin

func init() {
	volumePlugins = probeplugins.ProbeVolumePlugins("")
}

// ValidateForMountOptions validations mount options
func ValidateForMountOptions(pv *api.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}

	volumePlugin := findPluginBySpec(volumePlugins, pv)
	mountOptions := volume.MountOptionFromApiPV(pv)
	metaField := field.NewPath("metadata")
	if volumePlugin == nil && len(mountOptions) > 0 {
		allErrs = append(allErrs, field.Forbidden(metaField.Child("annotations", volume.MountOptionAnnotation), "may not specify mount options for this volume type"))
	}

	if volumePlugin != nil {
		if !volumePlugin.SupportsMountOption() && len(mountOptions) > 0 {
			allErrs = append(allErrs, field.Forbidden(metaField.Child("annotations", volume.MountOptionAnnotation), "may not specify mount options for this volume type"))
		}
	}
	return allErrs
}

func findPluginBySpec(volumePlugins []volume.VolumePlugin, pv *api.PersistentVolume) volume.VolumePlugin {
	matches := []volume.VolumePlugin{}
	v1Pv := &v1.PersistentVolume{}
	err := v1.Convert_api_PersistentVolume_To_v1_PersistentVolume(pv, v1Pv, nil)
	if err != nil {
		glog.Errorf("Error converting to v1.PersistentVolume: %v", err)
		return nil
	}
	volumeSpec := &volume.Spec{PersistentVolume: v1Pv}
	for _, plugin := range volumePlugins {
		if plugin.CanSupport(volumeSpec) {
			matches = append(matches, plugin)
		}
	}

	if len(matches) == 0 {
		glog.V(5).Infof("No matching plugin found for : %s", pv.Name)
		return nil
	}

	if len(matches) > 1 {
		glog.V(3).Infof("multiple volume plugins matched for : %s ", pv.Name)
		return nil
	}

	return matches[0]
}
