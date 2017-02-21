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
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/volume"
)

type pluginDefaults flexVolumePlugin

func logPrefix(plugin *flexVolumePlugin) string {
	return "flexVolume driver " + plugin.driverName + ": "
}

func (plugin *pluginDefaults) GetVolumeName(spec *volume.Spec) (string, error) {
	glog.Warning(logPrefix((*flexVolumePlugin)(plugin)), "using default GetVolumeName for volume ", spec.Name)
	return spec.Name(), nil
}
