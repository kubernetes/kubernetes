/*
Copyright 2014 The Kubernetes Authors.

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

package rbd

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/rbd" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

type fakeDiskManager struct {
	// Make sure we can run tests in parallel.
	mutex sync.RWMutex
	// Key format: "<pool>/<image>"
	rbdImageLocks map[string]bool
	rbdMapIndex   int
	rbdDevices    map[string]bool
}

func NewFakeDiskManager() *fakeDiskManager {
	return &fakeDiskManager{
		rbdImageLocks: make(map[string]bool),
		rbdMapIndex:   0,
		rbdDevices:    make(map[string]bool),
	}
}

func (fake *fakeDiskManager) MakeGlobalPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}

func (fake *fakeDiskManager) AttachDisk(b rbdMounter) (string, error) {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	fake.rbdMapIndex += 1
	devicePath := fmt.Sprintf("/dev/rbd%d", fake.rbdMapIndex)
	fake.rbdDevices[devicePath] = true
	return devicePath, nil
}

func (fake *fakeDiskManager) DetachDisk(r *rbdPlugin, deviceMountPath string, device string) error {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	ok := fake.rbdDevices[device]
	if !ok {
		return fmt.Errorf("rbd: failed to detach device %s, it does not exist", device)
	}
	delete(fake.rbdDevices, device)
	return nil
}

func (fake *fakeDiskManager) CreateImage(provisioner *rbdVolumeProvisioner) (r *v1.RBDPersistentVolumeSource, volumeSizeGB int, err error) {
	return nil, 0, fmt.Errorf("not implemented")
}

func (fake *fakeDiskManager) DeleteImage(deleter *rbdVolumeDeleter) error {
	return fmt.Errorf("not implemented")
}

func (fake *fakeDiskManager) Fencing(r rbdMounter, nodeName string) error {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	key := fmt.Sprintf("%s/%s", r.Pool, r.Image)
	isLocked, ok := fake.rbdImageLocks[key]
	if ok && isLocked {
		// not expected in testing
		return fmt.Errorf("%s is already locked", key)
	}
	fake.rbdImageLocks[key] = true
	return nil
}

func (fake *fakeDiskManager) Defencing(r rbdMounter, nodeName string) error {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	key := fmt.Sprintf("%s/%s", r.Pool, r.Image)
	isLocked, ok := fake.rbdImageLocks[key]
	if !ok || !isLocked {
		// not expected in testing
		return fmt.Errorf("%s is not locked", key)
	}
	delete(fake.rbdImageLocks, key)
	return nil
}

func (fake *fakeDiskManager) IsLocked(r rbdMounter, nodeName string) (bool, error) {
	fake.mutex.RLock()
	defer fake.mutex.RUnlock()
	key := fmt.Sprintf("%s/%s", r.Pool, r.Image)
	isLocked, ok := fake.rbdImageLocks[key]
	return ok && isLocked, nil
}

// checkMounterLog checks fakeMounter must have expected logs, and the last action msut equal to expectedAction.
func checkMounterLog(t *testing.T, fakeMounter *mount.FakeMounter, expected int, expectedAction mount.FakeAction) {
	if len(fakeMounter.Log) != expected {
		t.Fatalf("fakeMounter should have %d logs, actual: %d", expected, len(fakeMounter.Log))
	}
	lastIndex := len(fakeMounter.Log) - 1
	lastAction := fakeMounter.Log[lastIndex]
	if !reflect.DeepEqual(expectedAction, lastAction) {
		t.Fatalf("fakeMounter.Log[%d] should be %v, not: %v", lastIndex, expectedAction, lastAction)
	}
}

func doTestPlugin(t *testing.T, c *testcase) {
	fakeVolumeHost := volumetest.NewFakeVolumeHost(c.root, nil, nil)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, fakeVolumeHost)
	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeMounter := fakeVolumeHost.GetMounter(plug.GetPluginName()).(*mount.FakeMounter)
	fakeNodeName := types.NodeName("localhost")
	fdm := NewFakeDiskManager()

	// attacher
	attacher, err := plug.(*rbdPlugin).newAttacherInternal(fdm)
	if err != nil {
		t.Errorf("Failed to make a new Attacher: %v", err)
	}
	deviceAttachPath, err := attacher.Attach(c.spec, fakeNodeName)
	if err != nil {
		t.Fatal(err)
	}
	devicePath, err := attacher.WaitForAttach(c.spec, deviceAttachPath, c.pod, time.Second*10)
	if err != nil {
		t.Fatal(err)
	}
	if devicePath != c.expectedDevicePath {
		t.Errorf("Unexpected path, expected %q, not: %q", c.expectedDevicePath, devicePath)
	}
	deviceMountPath, err := attacher.GetDeviceMountPath(c.spec)
	if err != nil {
		t.Fatal(err)
	}
	if deviceMountPath != c.expectedDeviceMountPath {
		t.Errorf("Unexpected mount path, expected %q, not: %q", c.expectedDeviceMountPath, deviceMountPath)
	}
	err = attacher.MountDevice(c.spec, devicePath, deviceMountPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(deviceMountPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("Attacher.MountDevice() failed, device mount path not created: %s", deviceMountPath)
		} else {
			t.Errorf("Attacher.MountDevice() failed: %v", err)
		}
	}
	checkMounterLog(t, fakeMounter, 1, mount.FakeAction{Action: "mount", Target: c.expectedDeviceMountPath, Source: devicePath, FSType: "ext4"})

	// mounter
	mounter, err := plug.(*rbdPlugin).newMounterInternal(c.spec, c.pod.UID, fdm, "secrets")
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Error("Got a nil Mounter")
	}
	path := mounter.GetPath()
	if path != c.expectedPodMountPath {
		t.Errorf("Unexpected path, expected %q, got: %q", c.expectedPodMountPath, path)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	checkMounterLog(t, fakeMounter, 2, mount.FakeAction{Action: "mount", Target: c.expectedPodMountPath, Source: devicePath, FSType: ""})

	// unmounter
	unmounter, err := plug.(*rbdPlugin).newUnmounterInternal(c.spec.Name(), c.pod.UID, fdm)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Error("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}
	checkMounterLog(t, fakeMounter, 3, mount.FakeAction{Action: "unmount", Target: c.expectedPodMountPath, Source: "", FSType: ""})

	// detacher
	detacher, err := plug.(*rbdPlugin).newDetacherInternal(fdm)
	if err != nil {
		t.Errorf("Failed to make a new Attacher: %v", err)
	}
	err = detacher.UnmountDevice(deviceMountPath)
	if err != nil {
		t.Fatalf("Detacher.UnmountDevice failed to unmount %s", deviceMountPath)
	}
	checkMounterLog(t, fakeMounter, 4, mount.FakeAction{Action: "unmount", Target: c.expectedDeviceMountPath, Source: "", FSType: ""})
	err = detacher.Detach(deviceMountPath, fakeNodeName)
	if err != nil {
		t.Fatalf("Detacher.Detach failed to detach %s from %s", deviceMountPath, fakeNodeName)
	}
}

type testcase struct {
	spec                    *volume.Spec
	root                    string
	pod                     *v1.Pod
	expectedDevicePath      string
	expectedDeviceMountPath string
	expectedPodMountPath    string
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	podUID := uuid.NewUUID()
	var cases []*testcase
	cases = append(cases, &testcase{
		spec: volume.NewSpecFromVolume(&v1.Volume{
			Name: "vol1",
			VolumeSource: v1.VolumeSource{
				RBD: &v1.RBDVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDPool:      "pool1",
					RBDImage:     "image1",
					FSType:       "ext4",
				},
			},
		}),
		root: tmpDir,
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "testpod",
				Namespace: "testns",
				UID:       podUID,
			},
		},
		expectedDevicePath:      "/dev/rbd1",
		expectedDeviceMountPath: fmt.Sprintf("%s/plugins/kubernetes.io/rbd/rbd/pool1-image-image1", tmpDir),
		expectedPodMountPath:    fmt.Sprintf("%s/pods/%s/volumes/kubernetes.io~rbd/vol1", tmpDir, podUID),
	})
	cases = append(cases, &testcase{
		spec: volume.NewSpecFromPersistentVolume(&v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "vol2",
			},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					RBD: &v1.RBDPersistentVolumeSource{
						CephMonitors: []string{"a", "b"},
						RBDPool:      "pool2",
						RBDImage:     "image2",
						FSType:       "ext4",
					},
				},
			},
		}, false),
		root: tmpDir,
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "testpod",
				Namespace: "testns",
				UID:       podUID,
			},
		},
		expectedDevicePath:      "/dev/rbd1",
		expectedDeviceMountPath: fmt.Sprintf("%s/plugins/kubernetes.io/rbd/rbd/pool2-image-image2", tmpDir),
		expectedPodMountPath:    fmt.Sprintf("%s/pods/%s/volumes/kubernetes.io~rbd/vol2", tmpDir, podUID),
	})

	for i := 0; i < len(cases); i++ {
		doTestPlugin(t, cases[i])
	}
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				RBD: &v1.RBDPersistentVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDImage:     "bar",
					FSType:       "ext4",
				},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(rbdPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}

func TestGetSecretNameAndNamespace(t *testing.T) {
	secretName := "test-secret-name"
	secretNamespace := "test-secret-namespace"

	volSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					RBD: &v1.RBDPersistentVolumeSource{
						CephMonitors: []string{"a", "b"},
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}

	secretRef := new(v1.SecretReference)
	secretRef.Name = secretName
	secretRef.Namespace = secretNamespace
	volSpec.PersistentVolume.Spec.PersistentVolumeSource.RBD.SecretRef = secretRef

	foundSecretName, foundSecretNamespace, err := getSecretNameAndNamespace(volSpec, "default")
	if err != nil {
		t.Errorf("getSecretNameAndNamespace failed to get Secret's name and namespace: %v", err)
	}
	if strings.Compare(secretName, foundSecretName) != 0 || strings.Compare(secretNamespace, foundSecretNamespace) != 0 {
		t.Errorf("getSecretNameAndNamespace returned incorrect values, expected %s and %s but got %s and %s", secretName, secretNamespace, foundSecretName, foundSecretNamespace)
	}
}
