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

package csi

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"testing"

	api "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestBlockMapperGetGlobalMapPath(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	// TODO (vladimirvivien) specName with slashes will not work
	testCases := []struct {
		name           string
		specVolumeName string
		path           string
	}{
		{
			name:           "simple specName",
			specVolumeName: "spec-0",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/%s/%s", "spec-0", "dev")),
		},
		{
			name:           "specName with dots",
			specVolumeName: "test.spec.1",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/%s/%s", "test.spec.1", "dev")),
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		pv := makeTestPV(tc.specVolumeName, 10, testDriver, testVol)
		spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
		mapper, err := plug.NewBlockVolumeMapper(
			spec,
			&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
			volume.VolumeOptions{},
		)
		if err != nil {
			t.Fatalf("Failed to make a new Mapper: %v", err)
		}
		csiMapper := mapper.(*csiBlockMapper)

		path, err := csiMapper.GetGlobalMapPath(spec)
		if err != nil {
			t.Errorf("mapper GetGlobalMapPath failed: %v", err)
		}

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}
	}
}

func TestBlockMapperSetupDevice(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithNodeName(
		tmpDir,
		fakeClient,
		nil,
		"fakeNode",
	)
	plug.host = host
	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	pvName := pv.GetName()
	nodeName := string(plug.host.GetNodeName())
	spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

	// MapDevice
	mapper, err := plug.NewBlockVolumeMapper(
		spec,
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("failed to create new mapper: %v", err)
	}
	csiMapper := mapper.(*csiBlockMapper)
	csiMapper.csiClient = setupClient(t, true)

	attachID := getAttachmentName(csiMapper.volumeID, csiMapper.driverName, string(nodeName))
	attachment := makeTestAttachment(attachID, nodeName, pvName)
	attachment.Status.Attached = true
	_, err = csiMapper.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
	if err != nil {
		t.Fatalf("failed to setup VolumeAttachment: %v", err)
	}
	t.Log("created attachement ", attachID)

	devicePath, err := csiMapper.SetUpDevice()
	if err != nil {
		t.Fatalf("mapper failed to SetupDevice: %v", err)
	}

	globalMapPath, err := csiMapper.GetGlobalMapPath(spec)
	if err != nil {
		t.Fatalf("mapper failed to GetGlobalMapPath: %v", err)
	}

	if devicePath != globalMapPath {
		t.Fatalf("mapper.SetupDevice returned unexpected path %s instead of %v", devicePath, globalMapPath)
	}

	vols := csiMapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodeStagedVolumes()
	if vols[csiMapper.volumeID] != devicePath {
		t.Error("csi server may not have received NodePublishVolume call")
	}
}

func TestBlockMapperMapDevice(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithNodeName(
		tmpDir,
		fakeClient,
		nil,
		"fakeNode",
	)
	plug.host = host
	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	pvName := pv.GetName()
	nodeName := string(plug.host.GetNodeName())
	spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

	// MapDevice
	mapper, err := plug.NewBlockVolumeMapper(
		spec,
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("failed to create new mapper: %v", err)
	}
	csiMapper := mapper.(*csiBlockMapper)
	csiMapper.csiClient = setupClient(t, true)

	attachID := getAttachmentName(csiMapper.volumeID, csiMapper.driverName, string(nodeName))
	attachment := makeTestAttachment(attachID, nodeName, pvName)
	attachment.Status.Attached = true
	_, err = csiMapper.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
	if err != nil {
		t.Fatalf("failed to setup VolumeAttachment: %v", err)
	}
	t.Log("created attachement ", attachID)

	devicePath, err := csiMapper.SetUpDevice()
	if err != nil {
		t.Fatalf("mapper failed to SetupDevice: %v", err)
	}
	globalMapPath, err := csiMapper.GetGlobalMapPath(csiMapper.spec)
	if err != nil {
		t.Fatalf("mapper failed to GetGlobalMapPath: %v", err)
	}

	// Map device to global and pod device map path
	volumeMapPath, volName := csiMapper.GetPodDeviceMapPath()
	err = csiMapper.MapDevice(devicePath, globalMapPath, volumeMapPath, volName, csiMapper.podUID)
	if err != nil {
		t.Fatalf("mapper failed to GetGlobalMapPath: %v", err)
	}

	if _, err := os.Stat(filepath.Join(volumeMapPath, volName)); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("mapper.MapDevice failed, volume path not created: %s", volumeMapPath)
		} else {
			t.Errorf("mapper.MapDevice failed: %v", err)
		}
	}

	pubs := csiMapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
	if pubs[csiMapper.volumeID] != volumeMapPath {
		t.Error("csi server may not have received NodePublishVolume call")
	}
}

func TestBlockMapperTearDownDevice(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithNodeName(
		tmpDir,
		fakeClient,
		nil,
		"fakeNode",
	)
	plug.host = host
	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

	// save volume data
	dir := getVolumeDeviceDataDir(pv.ObjectMeta.Name, plug.host)
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		dir,
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	unmapper, err := plug.NewBlockVolumeUnmapper(pv.ObjectMeta.Name, testPodUID)
	if err != nil {
		t.Fatalf("failed to make a new Unmapper: %v", err)
	}

	csiUnmapper := unmapper.(*csiBlockMapper)
	csiUnmapper.csiClient = setupClient(t, true)

	globalMapPath, err := csiUnmapper.GetGlobalMapPath(spec)
	if err != nil {
		t.Fatalf("unmapper failed to GetGlobalMapPath: %v", err)
	}

	err = csiUnmapper.TearDownDevice(globalMapPath, "/dev/test")
	if err != nil {
		t.Fatal(err)
	}

	// ensure csi client call and node unstaged
	vols := csiUnmapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodeStagedVolumes()
	if _, ok := vols[csiUnmapper.volumeID]; ok {
		t.Error("csi server may not have received NodeUnstageVolume call")
	}

	// ensure csi client call and node unpblished
	pubs := csiUnmapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
	if _, ok := pubs[csiUnmapper.volumeID]; ok {
		t.Error("csi server may not have received NodeUnpublishVolume call")
	}
}
