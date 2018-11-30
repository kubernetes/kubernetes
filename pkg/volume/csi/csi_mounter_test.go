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

package csi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"testing"

	"reflect"

	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	csiapi "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	fakecsi "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

var (
	testDriver  = "test-driver"
	testVol     = "vol-123"
	testns      = "test-ns"
	testPod     = "test-pod"
	testPodUID  = types.UID("test-pod")
	testAccount = "test-service-account"
)

func TestMounterGetPath(t *testing.T) {
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
			path:           path.Join(tmpDir, fmt.Sprintf("pods/%s/volumes/kubernetes.io~csi/%s/%s", testPodUID, "spec-0", "/mount")),
		},
		{
			name:           "specName with dots",
			specVolumeName: "test.spec.1",
			path:           path.Join(tmpDir, fmt.Sprintf("pods/%s/volumes/kubernetes.io~csi/%s/%s", testPodUID, "test.spec.1", "/mount")),
		},
	}
	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
		pv := makeTestPV(tc.specVolumeName, 10, testDriver, testVol)
		spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
		mounter, err := plug.NewMounter(
			spec,
			&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
			volume.VolumeOptions{},
		)
		if err != nil {
			t.Fatalf("Failed to make a new Mounter: %v", err)
		}
		csiMounter := mounter.(*csiMountMgr)

		path := csiMounter.GetPath()

		if tc.path != path {
			t.Errorf("expecting path %s, got %s", tc.path, path)
		}
	}
}

func MounterSetUpTests(t *testing.T, podInfoEnabled bool) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIDriverRegistry, podInfoEnabled)()
	tests := []struct {
		name                  string
		driver                string
		volumeContext         map[string]string
		expectedVolumeContext map[string]string
	}{
		{
			name:                  "no pod info",
			driver:                "no-info",
			volumeContext:         nil,
			expectedVolumeContext: nil,
		},
		{
			name:                  "no CSIDriver -> no pod info",
			driver:                "unknown-driver",
			volumeContext:         nil,
			expectedVolumeContext: nil,
		},
		{
			name:                  "CSIDriver with PodInfoRequiredOnMount=nil -> no pod info",
			driver:                "nil",
			volumeContext:         nil,
			expectedVolumeContext: nil,
		},
		{
			name:                  "no pod info -> keep existing volumeContext",
			driver:                "no-info",
			volumeContext:         map[string]string{"foo": "bar"},
			expectedVolumeContext: map[string]string{"foo": "bar"},
		},
		{
			name:                  "add pod info",
			driver:                "info",
			volumeContext:         nil,
			expectedVolumeContext: map[string]string{"csi.storage.k8s.io/pod.uid": "test-pod", "csi.storage.k8s.io/serviceAccount.name": "test-service-account", "csi.storage.k8s.io/pod.name": "test-pod", "csi.storage.k8s.io/pod.namespace": "test-ns"},
		},
		{
			name:                  "add pod info -> keep existing volumeContext",
			driver:                "info",
			volumeContext:         map[string]string{"foo": "bar"},
			expectedVolumeContext: map[string]string{"foo": "bar", "csi.storage.k8s.io/pod.uid": "test-pod", "csi.storage.k8s.io/serviceAccount.name": "test-service-account", "csi.storage.k8s.io/pod.name": "test-pod", "csi.storage.k8s.io/pod.namespace": "test-ns"},
		},
	}

	emptyPodMountInfoVersion := ""
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			klog.Infof("Starting test %s", test.name)
			fakeClient := fakeclient.NewSimpleClientset()
			fakeCSIClient := fakecsi.NewSimpleClientset(
				getCSIDriver("no-info", &emptyPodMountInfoVersion, nil),
				getCSIDriver("info", &currentPodInfoMountVersion, nil),
				getCSIDriver("nil", nil, nil),
			)
			plug, tmpDir := newTestPlugin(t, fakeClient, fakeCSIClient)
			defer os.RemoveAll(tmpDir)

			if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
				// Wait until the informer in CSI volume plugin has all CSIDrivers.
				wait.PollImmediate(testInformerSyncPeriod, testInformerSyncTimeout, func() (bool, error) {
					return plug.csiDriverInformer.Informer().HasSynced(), nil
				})
			}

			registerFakePlugin(test.driver, "endpoint", []string{"1.0.0"}, t)
			pv := makeTestPV("test-pv", 10, test.driver, testVol)
			pv.Spec.CSI.VolumeAttributes = test.volumeContext
			pv.Spec.MountOptions = []string{"foo=bar", "baz=qux"}
			pvName := pv.GetName()

			mounter, err := plug.NewMounter(
				volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
				&api.Pod{
					ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns, Name: testPod},
					Spec: api.PodSpec{
						ServiceAccountName: testAccount,
					},
				},
				volume.VolumeOptions{},
			)
			if err != nil {
				t.Fatalf("failed to make a new Mounter: %v", err)
			}

			if mounter == nil {
				t.Fatal("failed to create CSI mounter")
			}

			csiMounter := mounter.(*csiMountMgr)
			csiMounter.csiClient = setupClient(t, true)

			attachID := getAttachmentName(csiMounter.volumeID, string(csiMounter.driverName), string(plug.host.GetNodeName()))

			attachment := &storage.VolumeAttachment{
				ObjectMeta: meta.ObjectMeta{
					Name: attachID,
				},
				Spec: storage.VolumeAttachmentSpec{
					NodeName: "test-node",
					Attacher: csiPluginName,
					Source: storage.VolumeAttachmentSource{
						PersistentVolumeName: &pvName,
					},
				},
				Status: storage.VolumeAttachmentStatus{
					Attached:    false,
					AttachError: nil,
					DetachError: nil,
				},
			}
			_, err = csiMounter.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
			if err != nil {
				t.Fatalf("failed to setup VolumeAttachment: %v", err)
			}

			// Mounter.SetUp()
			fsGroup := int64(2000)
			if err := csiMounter.SetUp(&fsGroup); err != nil {
				t.Fatalf("mounter.Setup failed: %v", err)
			}

			//Test the default value of file system type is not overridden
			if len(csiMounter.spec.PersistentVolume.Spec.CSI.FSType) != 0 {
				t.Errorf("default value of file system type was overridden by type %s", csiMounter.spec.PersistentVolume.Spec.CSI.FSType)
			}

			path := csiMounter.GetPath()
			if _, err := os.Stat(path); err != nil {
				if os.IsNotExist(err) {
					t.Errorf("SetUp() failed, volume path not created: %s", path)
				} else {
					t.Errorf("SetUp() failed: %v", err)
				}
			}

			// ensure call went all the way
			pubs := csiMounter.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
			vol, ok := pubs[csiMounter.volumeID]
			if !ok {
				t.Error("csi server may not have received NodePublishVolume call")
			}
			if vol.Path != csiMounter.GetPath() {
				t.Errorf("csi server expected path %s, got %s", csiMounter.GetPath(), vol.Path)
			}
			if !reflect.DeepEqual(vol.MountFlags, pv.Spec.MountOptions) {
				t.Errorf("csi server expected mount options %v, got %v", pv.Spec.MountOptions, vol.MountFlags)
			}
			if podInfoEnabled {
				if !reflect.DeepEqual(vol.VolumeContext, test.expectedVolumeContext) {
					t.Errorf("csi server expected volumeContext %+v, got %+v", test.expectedVolumeContext, vol.VolumeContext)
				}
			} else {
				// CSIPodInfo feature is disabled, we expect no modifications to volumeContext.
				if !reflect.DeepEqual(vol.VolumeContext, test.volumeContext) {
					t.Errorf("csi server expected volumeContext %+v, got %+v", test.volumeContext, vol.VolumeContext)
				}
			}
		})
	}
}

func TestMounterSetUp(t *testing.T) {
	t.Run("WithCSIPodInfo", func(t *testing.T) {
		MounterSetUpTests(t, true)
	})
	t.Run("WithoutCSIPodInfo", func(t *testing.T) {
		MounterSetUpTests(t, false)
	})
}
func TestMounterSetUpWithFSGroup(t *testing.T) {
	fakeClient := fakeclient.NewSimpleClientset()
	plug, tmpDir := newTestPlugin(t, fakeClient, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name        string
		accessModes []api.PersistentVolumeAccessMode
		readOnly    bool
		fsType      string
		setFsGroup  bool
		fsGroup     int64
	}{
		{
			name: "default fstype, with no fsgroup (should not apply fsgroup)",
			accessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			readOnly: false,
			fsType:   "",
		},
		{
			name: "default fstype  with fsgroup (should not apply fsgroup)",
			accessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			readOnly:   false,
			fsType:     "",
			setFsGroup: true,
			fsGroup:    3000,
		},
		{
			name: "fstype, fsgroup, RWM, ROM provided (should not apply fsgroup)",
			accessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteMany,
				api.ReadOnlyMany,
			},
			fsType:     "ext4",
			setFsGroup: true,
			fsGroup:    3000,
		},
		{
			name: "fstype, fsgroup, RWO, but readOnly (should not apply fsgroup)",
			accessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			readOnly:   true,
			fsType:     "ext4",
			setFsGroup: true,
			fsGroup:    3000,
		},
		{
			name: "fstype, fsgroup, RWO provided (should apply fsgroup)",
			accessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			fsType:     "ext4",
			setFsGroup: true,
			fsGroup:    3000,
		},
	}

	for i, tc := range testCases {
		t.Logf("Running test %s", tc.name)

		volName := fmt.Sprintf("test-vol-%d", i)
		registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
		pv := makeTestPV("test-pv", 10, testDriver, volName)
		pv.Spec.AccessModes = tc.accessModes
		pvName := pv.GetName()

		spec := volume.NewSpecFromPersistentVolume(pv, tc.readOnly)

		if tc.fsType != "" {
			spec.PersistentVolume.Spec.CSI.FSType = tc.fsType
		}

		mounter, err := plug.NewMounter(
			spec,
			&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
			volume.VolumeOptions{},
		)
		if err != nil {
			t.Fatalf("Failed to make a new Mounter: %v", err)
		}

		if mounter == nil {
			t.Fatal("failed to create CSI mounter")
		}

		csiMounter := mounter.(*csiMountMgr)
		csiMounter.csiClient = setupClient(t, true)

		attachID := getAttachmentName(csiMounter.volumeID, string(csiMounter.driverName), string(plug.host.GetNodeName()))
		attachment := makeTestAttachment(attachID, "test-node", pvName)

		_, err = csiMounter.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
		if err != nil {
			t.Errorf("failed to setup VolumeAttachment: %v", err)
			continue
		}

		// Mounter.SetUp()
		var fsGroupPtr *int64
		if tc.setFsGroup {
			fsGroup := tc.fsGroup
			fsGroupPtr = &fsGroup
		}
		if err := csiMounter.SetUp(fsGroupPtr); err != nil {
			t.Fatalf("mounter.Setup failed: %v", err)
		}

		//Test the default value of file system type is not overridden
		if len(csiMounter.spec.PersistentVolume.Spec.CSI.FSType) != len(tc.fsType) {
			t.Errorf("file system type was overridden by type %s", csiMounter.spec.PersistentVolume.Spec.CSI.FSType)
		}

		// ensure call went all the way
		pubs := csiMounter.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
		if pubs[csiMounter.volumeID].Path != csiMounter.GetPath() {
			t.Error("csi server may not have received NodePublishVolume call")
		}
	}
}

func TestUnmounterTeardown(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	// save the data file prior to unmount
	dir := path.Join(getTargetPath(testPodUID, pv.ObjectMeta.Name, plug.host), "/mount")
	if err := os.MkdirAll(dir, 0755); err != nil && !os.IsNotExist(err) {
		t.Errorf("failed to create dir [%s]: %v", dir, err)
	}

	// do a fake local mount
	diskMounter := util.NewSafeFormatAndMountFromHost(plug.GetPluginName(), plug.host)
	if err := diskMounter.FormatAndMount("/fake/device", dir, "testfs", nil); err != nil {
		t.Errorf("failed to mount dir [%s]: %v", dir, err)
	}

	if err := saveVolumeData(
		path.Dir(dir),
		volDataFileName,
		map[string]string{
			volDataKey.specVolID:  pv.ObjectMeta.Name,
			volDataKey.driverName: testDriver,
			volDataKey.volHandle:  testVol,
		},
	); err != nil {
		t.Fatalf("failed to save volume data: %v", err)
	}

	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	if err != nil {
		t.Fatalf("failed to make a new Unmounter: %v", err)
	}

	csiUnmounter := unmounter.(*csiMountMgr)
	csiUnmounter.csiClient = setupClient(t, true)
	err = csiUnmounter.TearDownAt(dir)
	if err != nil {
		t.Fatal(err)
	}

	// ensure csi client call
	pubs := csiUnmounter.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
	if _, ok := pubs[csiUnmounter.volumeID]; ok {
		t.Error("csi server may not have received NodeUnpublishVolume call")
	}

}

func TestSaveVolumeData(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		data       map[string]string
		shouldFail bool
	}{
		{name: "test with data ok", data: map[string]string{"key0": "val0", "_key1": "val1", "key2": "val2"}},
		{name: "test with data ok 2 ", data: map[string]string{"_key0_": "val0", "&key1": "val1", "key2": "val2"}},
	}

	for i, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		specVolID := fmt.Sprintf("spec-volid-%d", i)
		mountDir := path.Join(getTargetPath(testPodUID, specVolID, plug.host), "/mount")
		if err := os.MkdirAll(mountDir, 0755); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to create dir [%s]: %v", mountDir, err)
		}

		err := saveVolumeData(path.Dir(mountDir), volDataFileName, tc.data)

		if !tc.shouldFail && err != nil {
			t.Errorf("unexpected failure: %v", err)
		}
		// did file get created
		dataDir := getTargetPath(testPodUID, specVolID, plug.host)
		file := path.Join(dataDir, volDataFileName)
		if _, err := os.Stat(file); err != nil {
			t.Errorf("failed to create data dir: %v", err)
		}

		// validate content
		data, err := ioutil.ReadFile(file)
		if !tc.shouldFail && err != nil {
			t.Errorf("failed to read data file: %v", err)
		}

		jsonData := new(bytes.Buffer)
		if err := json.NewEncoder(jsonData).Encode(tc.data); err != nil {
			t.Errorf("failed to encode json: %v", err)
		}
		if string(data) != jsonData.String() {
			t.Errorf("expecting encoded data %v, got %v", string(data), jsonData)
		}
	}
}

func getCSIDriver(name string, podInfoMountVersion *string, attachable *bool) *csiapi.CSIDriver {
	return &csiapi.CSIDriver{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
		},
		Spec: csiapi.CSIDriverSpec{
			PodInfoOnMountVersion: podInfoMountVersion,
			AttachRequired:        attachable,
		},
	}
}
