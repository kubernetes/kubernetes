/*
Copyright 2024 The Kubernetes Authors.

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

package image

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	err := pluginMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, nil /* host */)
	if err != nil {
		t.Fatalf("Failed to init plugins: %v", err)
	}

	plugin, err := pluginMgr.FindPluginByName(pluginName)
	if err != nil {
		t.Fatal("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != pluginName {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: ""}}}}) {
		t.Errorf("Expected true")
	}
	if plugin.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
}
