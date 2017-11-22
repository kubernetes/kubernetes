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
	"fmt"
	"os"
	"path"
	"testing"

	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1alpha1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
)

var (
	testDriver = "test-driver"
	testVol    = "vol-123"
	testns     = "test-ns"
	testPodUID = types.UID("test-pod")
)

func TestMounterGetPath(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	mounter, err := plug.NewMounter(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
		&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}},
		volume.VolumeOptions{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}
	csiMounter := mounter.(*csiMountMgr)
	expectedPath := path.Join(tmpDir, fmt.Sprintf(
		"pods/%s/volumes/kubernetes.io~csi/%s/%s",
		testPodUID,
		csiMounter.driverName,
		csiMounter.volumeID,
	))
	mountPath := csiMounter.GetPath()
	if mountPath != expectedPath {
		t.Errorf("Got unexpected path: %s", mountPath)
	}

}

func TestMounterSetUp(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)
	pvName := pv.GetName()

	mounter, err := plug.NewMounter(
		volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly),
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
	csiMounter.csiClient = setupClient(t)

	attachment := &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: "pv-1234556775313",
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
	_, err = csiMounter.k8s.StorageV1alpha1().VolumeAttachments().Create(attachment)
	if err != nil {
		t.Fatalf("failed to setup VolumeAttachment: %v", err)
	}

	// Mounter.SetUp()
	if err := csiMounter.SetUp(nil); err != nil {
		t.Fatalf("mounter.Setup failed: %v", err)
	}

	// ensure call went all the way
	pubs := csiMounter.csiClient.(*csiDriverClient).nodeClient.(*fake.NodeClient).GetNodePublishedVolumes()
	if pubs[csiMounter.volumeID] != csiMounter.GetPath() {
		t.Error("csi server may not have received NodePublishVolume call")
	}
}

func TestUnmounterTeardown(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	pv := makeTestPV("test-pv", 10, testDriver, testVol)

	unmounter, err := plug.NewUnmounter(pv.ObjectMeta.Name, testPodUID)
	if err != nil {
		t.Fatalf("failed to make a new Unmounter: %v", err)
	}

	csiUnmounter := unmounter.(*csiMountMgr)
	csiUnmounter.csiClient = setupClient(t)

	dir := csiUnmounter.GetPath()

	err = csiUnmounter.TearDownAt(dir)
	if err != nil {
		t.Fatal(err)
	}

	// ensure csi client call
	pubs := csiUnmounter.csiClient.(*csiDriverClient).nodeClient.(*fake.NodeClient).GetNodePublishedVolumes()
	if _, ok := pubs[csiUnmounter.volumeID]; ok {
		t.Error("csi server may not have received NodeUnpublishVolume call")
	}

}
