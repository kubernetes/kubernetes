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

package app

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/klog/v2/ktesting"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
	"k8s.io/kubernetes/pkg/volume"
)

func checkPlugins(t *testing.T, got []volume.VolumePlugin, expected []string) {
	pluginNames := make([]string, len(got))
	for i, p := range got {
		pluginNames[i] = p.GetPluginName()
	}
	sort.Strings(pluginNames)
	sort.Strings(expected)
	if !reflect.DeepEqual(pluginNames, expected) {
		t.Errorf("Expected %+v, got %+v", expected, pluginNames)
	}
}

func TestProbeAttachableVolumePlugins(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	plugins, err := ProbeAttachableVolumePlugins(logger, getConfig())
	if err != nil {
		t.Fatalf("ProbeAttachableVolumePlugins failed: %s", err)
	}
	checkPlugins(t, plugins, []string{"kubernetes.io/csi", "kubernetes.io/fc", "kubernetes.io/iscsi"})
}

func TestProbeExpandableVolumePlugins(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	plugins, err := ProbeExpandableVolumePlugins(logger, getConfig())
	if err != nil {
		t.Fatalf("TestProbeExpandableVolumePlugins failed: %s", err)
	}
	checkPlugins(t, plugins, []string{"kubernetes.io/portworx-volume"})
}

func TestProbeControllerVolumePlugins(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	plugins, err := ProbeProvisionableRecyclableVolumePlugins(logger, getConfig())
	if err != nil {
		t.Fatalf("ProbeControllerVolumePlugins failed: %s", err)
	}
	checkPlugins(t, plugins, []string{"kubernetes.io/host-path", "kubernetes.io/nfs", "kubernetes.io/portworx-volume"})
}

func getConfig() persistentvolumeconfig.VolumeConfiguration {
	return persistentvolumeconfig.VolumeConfiguration{
		EnableHostPathProvisioning: true,
		EnableDynamicProvisioning:  true,
		PersistentVolumeRecyclerConfiguration: persistentvolumeconfig.PersistentVolumeRecyclerConfiguration{
			MaximumRetry:                5,
			MinimumTimeoutNFS:           30,
			PodTemplateFilePathNFS:      "",
			IncrementTimeoutNFS:         10,
			PodTemplateFilePathHostPath: "",
			MinimumTimeoutHostPath:      30,
			IncrementTimeoutHostPath:    10,
		},
		FlexVolumePluginDir: "",
	}
}
