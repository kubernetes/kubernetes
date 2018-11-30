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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func prepareBlockMapperTest(plug *csiPlugin, specVolumeName string, t *testing.T) (*csiBlockMapper, *volume.Spec, *api.PersistentVolume, error) {
	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV(specVolumeName, 10, testDriver, testVol)
	spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
	mapper, err := plug.NewBlockVolumeMapper(
		spec,
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("Failed to make a new Mapper: %v", err)
	}
	csiMapper := mapper.(*csiBlockMapper)
	return csiMapper, spec, pv, nil
}

func TestBlockMapperGetGlobalMapPath(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
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
		csiMapper, spec, _, err := prepareBlockMapperTest(plug, tc.specVolumeName, t)
		if err != nil {
			t.Fatalf("Failed to make a new Mapper: %v", err)
		}

		path, err := csiMapper.GetGlobalMapPath(spec)
		if err != nil {
			t.Errorf("mapper GetGlobalMapPath failed: %v", err)
		}

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}
	}
}

func TestBlockMapperGetStagingPath(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name           string
		specVolumeName string
		path           string
	}{
		{
			name:           "simple specName",
			specVolumeName: "spec-0",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/staging/%s", "spec-0")),
		},
		{
			name:           "specName with dots",
			specVolumeName: "test.spec.1",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/staging/%s", "test.spec.1")),
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		csiMapper, _, _, err := prepareBlockMapperTest(plug, tc.specVolumeName, t)
		if err != nil {
			t.Fatalf("Failed to make a new Mapper: %v", err)
		}

		path := csiMapper.getStagingPath()

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}
	}
}

func TestBlockMapperGetPublishPath(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name           string
		specVolumeName string
		path           string
	}{
		{
			name:           "simple specName",
			specVolumeName: "spec-0",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/publish/%s", "spec-0")),
		},
		{
			name:           "specName with dots",
			specVolumeName: "test.spec.1",
			path:           path.Join(tmpDir, fmt.Sprintf("plugins/kubernetes.io/csi/volumeDevices/publish/%s", "test.spec.1")),
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		csiMapper, _, _, err := prepareBlockMapperTest(plug, tc.specVolumeName, t)
		if err != nil {
			t.Fatalf("Failed to make a new Mapper: %v", err)
		}

		path := csiMapper.getPublishPath()

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}
	}
}

func TestBlockMapperGetDeviceMapPath(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name           string
		specVolumeName string
		path           string
	}{
		{
			name:           "simple specName",
			specVolumeName: "spec-0",
			path:           path.Join(tmpDir, fmt.Sprintf("pods/%s/volumeDevices/kubernetes.io~csi", testPodUID)),
		},
		{
			name:           "specName with dots",
			specVolumeName: "test.spec.1",
			path:           path.Join(tmpDir, fmt.Sprintf("pods/%s/volumeDevices/kubernetes.io~csi", testPodUID)),
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		csiMapper, _, _, err := prepareBlockMapperTest(plug, tc.specVolumeName, t)
		if err != nil {
			t.Fatalf("Failed to make a new Mapper: %v", err)
		}

		path, volName := csiMapper.GetPodDeviceMapPath()

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}

		if tc.specVolumeName != volName {
			t.Errorf("expecting volName %s, got %s", tc.specVolumeName, volName)
		}
	}
}

func TestBlockMapperSetupDevice(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		fakeClient,
		nil,
		nil,
		"fakeNode",
	)
	plug.host = host

	csiMapper, _, pv, err := prepareBlockMapperTest(plug, "test-pv", t)
	if err != nil {
		t.Fatalf("Failed to make a new Mapper: %v", err)
	}

	pvName := pv.GetName()
	nodeName := string(plug.host.GetNodeName())

	csiMapper.csiClient = setupClient(t, true)

	attachID := getAttachmentName(csiMapper.volumeID, string(csiMapper.driverName), string(nodeName))
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

	// Check if SetUpDevice returns the right path
	publishPath := csiMapper.getPublishPath()
	if devicePath != publishPath {
		t.Fatalf("mapper.SetupDevice returned unexpected path %s instead of %v", devicePath, publishPath)
	}

	// Check if NodeStageVolume staged to the right path
	stagingPath := csiMapper.getStagingPath()
	svols := csiMapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodeStagedVolumes()
	svol, ok := svols[csiMapper.volumeID]
	if !ok {
		t.Error("csi server may not have received NodeStageVolume call")
	}
	if svol.Path != stagingPath {
		t.Errorf("csi server expected device path %s, got %s", stagingPath, svol.Path)
	}

	// Check if NodePublishVolume published to the right path
	pvols := csiMapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
	pvol, ok := pvols[csiMapper.volumeID]
	if !ok {
		t.Error("csi server may not have received NodePublishVolume call")
	}
	if pvol.Path != publishPath {
		t.Errorf("csi server expected path %s, got %s", publishPath, pvol.Path)
	}
}

func TestBlockMapperMapDevice(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		fakeClient,
		nil,
		nil,
		"fakeNode",
	)
	plug.host = host

	csiMapper, _, pv, err := prepareBlockMapperTest(plug, "test-pv", t)
	if err != nil {
		t.Fatalf("Failed to make a new Mapper: %v", err)
	}

	pvName := pv.GetName()
	nodeName := string(plug.host.GetNodeName())

	csiMapper.csiClient = setupClient(t, true)

	attachID := getAttachmentName(csiMapper.volumeID, string(csiMapper.driverName), string(nodeName))
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

	// Actual SetupDevice should create a symlink to or a bind mout of device in devicePath.
	// Create dummy file there before calling MapDevice to test it properly.
	fd, err := os.Create(devicePath)
	if err != nil {
		t.Fatalf("mapper failed to create dummy file in devicePath: %v", err)
	}
	if err := fd.Close(); err != nil {
		t.Fatalf("mapper failed to close dummy file in devicePath: %v", err)
	}

	// Map device to global and pod device map path
	volumeMapPath, volName := csiMapper.GetPodDeviceMapPath()
	err = csiMapper.MapDevice(devicePath, globalMapPath, volumeMapPath, volName, csiMapper.podUID)
	if err != nil {
		t.Fatalf("mapper failed to GetGlobalMapPath: %v", err)
	}

	// Check if symlink {globalMapPath}/{podUID} exists
	globalMapFilePath := filepath.Join(globalMapPath, string(csiMapper.podUID))
	if _, err := os.Stat(globalMapFilePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("mapper.MapDevice failed, symlink in globalMapPath not created: %v", err)
			t.Errorf("mapper.MapDevice devicePath:%v, globalMapPath: %v, globalMapFilePath: %v",
				devicePath, globalMapPath, globalMapFilePath)
		} else {
			t.Errorf("mapper.MapDevice failed: %v", err)
		}
	}

	// Check if symlink {volumeMapPath}/{volName} exists
	volumeMapFilePath := filepath.Join(volumeMapPath, volName)
	if _, err := os.Stat(volumeMapFilePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("mapper.MapDevice failed, symlink in volumeMapPath not created: %v", err)
		} else {
			t.Errorf("mapper.MapDevice failed: %v", err)
		}
	}
}

func TestBlockMapperTearDownDevice(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIBlockVolume, true)()

	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	fakeClient := fakeclient.NewSimpleClientset()
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		fakeClient,
		nil,
		nil,
		"fakeNode",
	)
	plug.host = host

	_, spec, pv, err := prepareBlockMapperTest(plug, "test-pv", t)
	if err != nil {
		t.Fatalf("Failed to make a new Mapper: %v", err)
	}

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

	// ensure csi client call and node unpblished
	pubs := csiUnmapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
	if _, ok := pubs[csiUnmapper.volumeID]; ok {
		t.Error("csi server may not have received NodeUnpublishVolume call")
	}

	// ensure csi client call and node unstaged
	vols := csiUnmapper.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodeStagedVolumes()
	if _, ok := vols[csiUnmapper.volumeID]; ok {
		t.Error("csi server may not have received NodeUnstageVolume call")
	}
}
