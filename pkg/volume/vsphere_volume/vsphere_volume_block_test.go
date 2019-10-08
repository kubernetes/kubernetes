// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere_volume

import (
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

var (
	testVolumePath = "volPath1"
	testGlobalPath = "plugins/kubernetes.io/vsphere-volume/volumeDevices/volPath1"
	testPodPath    = "pods/poduid/volumeDevices/kubernetes.io~vsphere-volume"
)

func TestGetVolumeSpecFromGlobalMapPath(t *testing.T) {
	// make our test path for fake GlobalMapPath
	// /tmp symbolized our pluginDir
	// /tmp/testGlobalPathXXXXX/plugins/kubernetes.io/vsphere-volume/volumeDevices/
	tmpVDir, err := utiltesting.MkTmpdir("vsphereBlockVolume")
	if err != nil {
		t.Fatalf("cant' make a temp dir: %s", err)
	}
	// deferred clean up
	defer os.RemoveAll(tmpVDir)

	expectedGlobalPath := filepath.Join(tmpVDir, testGlobalPath)

	// Bad Path
	badspec, err := getVolumeSpecFromGlobalMapPath("", "")
	if badspec != nil || err == nil {
		t.Errorf("Expected not to get spec from GlobalMapPath but did")
	}

	// Good Path
	spec, err := getVolumeSpecFromGlobalMapPath("myVolume", expectedGlobalPath)
	if spec == nil || err != nil {
		t.Fatalf("Failed to get spec from GlobalMapPath: %s", err)
	}
	if spec.PersistentVolume.Name != "myVolume" {
		t.Errorf("Invalid PV name from GlobalMapPath spec: %s", spec.PersistentVolume.Name)
	}
	if spec.PersistentVolume.Spec.VsphereVolume.VolumePath != testVolumePath {
		t.Fatalf("Invalid volumePath from GlobalMapPath spec: %s", spec.PersistentVolume.Spec.VsphereVolume.VolumePath)
	}
	block := v1.PersistentVolumeBlock
	specMode := spec.PersistentVolume.Spec.VolumeMode
	if &specMode == nil {
		t.Errorf("Invalid volumeMode from GlobalMapPath spec: %v expected: %v", &specMode, block)
	}
	if *specMode != block {
		t.Errorf("Invalid volumeMode from GlobalMapPath spec: %v expected: %v", *specMode, block)
	}
}

func TestGetPodAndPluginMapPaths(t *testing.T) {
	tmpVDir, err := utiltesting.MkTmpdir("vsphereBlockVolume")
	if err != nil {
		t.Fatalf("cant' make a temp dir: %s", err)
	}
	// deferred clean up
	defer os.RemoveAll(tmpVDir)

	expectedGlobalPath := filepath.Join(tmpVDir, testGlobalPath)
	expectedPodPath := filepath.Join(tmpVDir, testPodPath)

	spec := getTestVolume(true) // block volume
	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), nil, volumetest.NewFakeVolumeHost(tmpVDir, nil, nil))
	plugin, err := pluginMgr.FindMapperPluginByName(vsphereVolumePluginName)
	if err != nil {
		os.RemoveAll(tmpVDir)
		t.Fatalf("Can't find the plugin by name: %q", vsphereVolumePluginName)
	}
	if plugin.GetPluginName() != vsphereVolumePluginName {
		t.Fatalf("Wrong name: %s", plugin.GetPluginName())
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID("poduid"),
		},
	}
	mapper, err := plugin.NewBlockVolumeMapper(spec, pod, volume.VolumeOptions{})
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}
	if mapper == nil {
		t.Fatalf("Got a nil Mounter")
	}

	// GetGlobalMapPath
	globalMapPath, err := mapper.GetGlobalMapPath(spec)
	if err != nil || len(globalMapPath) == 0 {
		t.Fatalf("Invalid GlobalMapPath from spec: %s", spec.PersistentVolume.Spec.VsphereVolume.VolumePath)
	}
	if globalMapPath != expectedGlobalPath {
		t.Errorf("Failed to get GlobalMapPath: %s %s", globalMapPath, expectedGlobalPath)
	}

	// GetPodDeviceMapPath
	devicePath, volumeName := mapper.GetPodDeviceMapPath()
	if devicePath != expectedPodPath {
		t.Errorf("Got unexpected pod path: %s, expected %s", devicePath, expectedPodPath)
	}
	if volumeName != testVolumePath {
		t.Errorf("Got unexpected volNamne: %s, expected %s", volumeName, testVolumePath)
	}
}

func getTestVolume(isBlock bool) *volume.Spec {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: testVolumePath,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
					VolumePath: testVolumePath,
				},
			},
		},
	}
	if isBlock {
		blockMode := v1.PersistentVolumeBlock
		pv.Spec.VolumeMode = &blockMode
	}
	return volume.NewSpecFromPersistentVolume(pv, true)
}
