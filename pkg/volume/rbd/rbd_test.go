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
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	testVolName    = "vol-1234"
	testRBDImage   = "volume-a4b47414-a675-47dc-a9cc-c223f13439b0"
	testRBDPool    = "volumes"
	testGlobalPath = "plugins/kubernetes.io/rbd/volumeDevices/volumes-image-volume-a4b47414-a675-47dc-a9cc-c223f13439b0"
)

func TestGetVolumeSpecFromGlobalMapPath(t *testing.T) {
	// make our test path for fake GlobalMapPath
	// /tmp symbolized our pluginDir
	// /tmp/testGlobalPathXXXXX/plugins/kubernetes.io/rbd/volumeDevices/pdVol1
	tmpVDir, err := utiltesting.MkTmpdir("rbdBlockTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	//deferred clean up
	defer os.RemoveAll(tmpVDir)

	expectedGlobalPath := filepath.Join(tmpVDir, testGlobalPath)

	//Bad Path
	badspec, err := getVolumeSpecFromGlobalMapPath("", testVolName)
	if badspec != nil || err == nil {
		t.Fatalf("Expected not to get spec from GlobalMapPath but did")
	}

	// Good Path
	spec, err := getVolumeSpecFromGlobalMapPath(expectedGlobalPath, testVolName)
	if spec == nil || err != nil {
		t.Fatalf("Failed to get spec from GlobalMapPath: %v", err)
	}

	if spec.PersistentVolume.Name != testVolName {
		t.Errorf("Invalid spec name for GlobalMapPath spec: %s", spec.PersistentVolume.Name)
	}

	if spec.PersistentVolume.Spec.RBD.RBDPool != testRBDPool {
		t.Errorf("Invalid RBDPool from GlobalMapPath spec: %s", spec.PersistentVolume.Spec.RBD.RBDPool)
	}

	if spec.PersistentVolume.Spec.RBD.RBDImage != testRBDImage {
		t.Errorf("Invalid RBDImage from GlobalMapPath spec: %s", spec.PersistentVolume.Spec.RBD.RBDImage)
	}

	block := v1.PersistentVolumeBlock
	specMode := spec.PersistentVolume.Spec.VolumeMode
	if &specMode == nil {
		t.Errorf("Invalid volumeMode from GlobalMapPath spec: %v - %v", &specMode, block)
	}
	if *specMode != block {
		t.Errorf("Invalid volumeMode from GlobalMapPath spec: %v - %v", *specMode, block)
	}
}

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/rbd" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{RBD: &v1.RBDVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{}}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{RBD: &v1.RBDPersistentVolumeSource{}}}}}) {
		t.Errorf("Expected true")
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

func (fake *fakeDiskManager) MakeGlobalVDPDName(rbd rbd) string {
	return makePDNameInternal(rbd.plugin.host, rbd.Pool, rbd.Image)
}

func (fake *fakeDiskManager) AttachDisk(b rbdMounter) (string, error) {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	fake.rbdMapIndex++
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

func (fake *fakeDiskManager) DetachBlockDisk(r rbdDiskUnmapper, device string) error {
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

func (fake *fakeDiskManager) ExpandImage(rbdExpander *rbdVolumeExpander, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	return resource.Quantity{}, fmt.Errorf("not implemented")
}

// checkMounterLog checks fakeMounter must have expected logs, and the last action msut equal to expectedAction.
func checkMounterLog(t *testing.T, fakeMounter *mount.FakeMounter, expected int, expectedAction mount.FakeAction) {
	log := fakeMounter.GetLog()
	if len(log) != expected {
		t.Fatalf("fakeMounter should have %d logs, actual: %d", expected, len(log))
	}
	lastIndex := len(log) - 1
	lastAction := log[lastIndex]
	if !reflect.DeepEqual(expectedAction, lastAction) {
		t.Fatalf("fakeMounter.Log[%d] should be %#v, not: %#v", lastIndex, expectedAction, lastAction)
	}
}

func doTestPlugin(t *testing.T, c *testcase) {
	fakeVolumeHost := volumetest.NewFakeVolumeHost(t, c.root, nil, nil)
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

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
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
	tmpDir, err = filepath.EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

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
					ReadOnly:     true,
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
		expectedDeviceMountPath: fmt.Sprintf("%s/plugins/kubernetes.io/rbd/mounts/pool1-image-image1", tmpDir),
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
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
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
		expectedDeviceMountPath: fmt.Sprintf("%s/plugins/kubernetes.io/rbd/mounts/pool2-image-image2", tmpDir),
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, client, nil))
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

// https://github.com/kubernetes/kubernetes/issues/57744
func TestGetDeviceMountPath(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	fakeVolumeHost := volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, fakeVolumeHost)
	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fdm := NewFakeDiskManager()

	// attacher
	attacher, err := plug.(*rbdPlugin).newAttacherInternal(fdm)
	if err != nil {
		t.Errorf("Failed to make a new Attacher: %v", err)
	}

	pool, image := "pool", "image"
	spec := volume.NewSpecFromVolume(&v1.Volume{
		Name: "vol",
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				CephMonitors: []string{"a", "b"},
				RBDPool:      pool,
				RBDImage:     image,
				FSType:       "ext4",
			},
		},
	})

	deprecatedDir := fmt.Sprintf("%s/plugins/kubernetes.io/rbd/rbd/%s-image-%s", tmpDir, pool, image)
	canonicalDir := fmt.Sprintf("%s/plugins/kubernetes.io/rbd/mounts/%s-image-%s", tmpDir, pool, image)

	type testCase struct {
		deprecated bool
		targetPath string
	}
	for _, c := range []testCase{
		{false, canonicalDir},
		{true, deprecatedDir},
	} {
		if c.deprecated {
			// This is a deprecated device mount path, we create it,
			// and hope attacher.GetDeviceMountPath return c.targetPath.
			if err := os.MkdirAll(c.targetPath, 0700); err != nil {
				t.Fatalf("Create deprecated mount path failed: %v", err)
			}
		}
		mountPath, err := attacher.GetDeviceMountPath(spec)
		if err != nil {
			t.Fatalf("GetDeviceMountPath failed: %v", err)
		}
		if mountPath != c.targetPath {
			t.Errorf("Mismatch device mount path: wanted %s, got %s", c.targetPath, mountPath)
		}
	}
}

// https://github.com/kubernetes/kubernetes/issues/57744
func TestConstructVolumeSpec(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skipf("TestConstructVolumeSpec is not supported on GOOS=%s", runtime.GOOS)
	}
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	fakeVolumeHost := volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, fakeVolumeHost)
	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeMounter := fakeVolumeHost.GetMounter(plug.GetPluginName()).(*mount.FakeMounter)

	pool, image, volumeName := "pool", "image", "vol"
	podMountPath := fmt.Sprintf("%s/pods/pod123/volumes/kubernetes.io~rbd/%s", tmpDir, volumeName)
	deprecatedDir := fmt.Sprintf("%s/plugins/kubernetes.io/rbd/rbd/%s-image-%s", tmpDir, pool, image)
	canonicalDir := fmt.Sprintf("%s/plugins/kubernetes.io/rbd/mounts/%s-image-%s", tmpDir, pool, image)

	type testCase struct {
		volumeName string
		targetPath string
	}

	for _, c := range []testCase{
		{"vol", canonicalDir},
		{"vol", deprecatedDir},
	} {
		if err := os.MkdirAll(c.targetPath, 0700); err != nil {
			t.Fatalf("Create mount path %s failed: %v", c.targetPath, err)
		}
		if err = fakeMounter.Mount("/dev/rbd0", c.targetPath, "fake", nil); err != nil {
			t.Fatalf("Mount %s to %s failed: %v", c.targetPath, podMountPath, err)
		}
		if err = fakeMounter.Mount(c.targetPath, podMountPath, "fake", []string{"bind"}); err != nil {
			t.Fatalf("Mount %s to %s failed: %v", c.targetPath, podMountPath, err)
		}
		spec, err := plug.ConstructVolumeSpec(c.volumeName, podMountPath)
		if err != nil {
			t.Errorf("ConstructVolumeSpec failed: %v", err)
		} else {
			if spec.Volume.RBD.RBDPool != pool {
				t.Errorf("Mismatch rbd pool: wanted %s, got %s", pool, spec.Volume.RBD.RBDPool)
			}
			if spec.Volume.RBD.RBDImage != image {
				t.Fatalf("Mismatch rbd image: wanted %s, got %s", image, spec.Volume.RBD.RBDImage)
			}
		}
		if err = fakeMounter.Unmount(podMountPath); err != nil {
			t.Fatalf("Unmount pod path %s failed: %v", podMountPath, err)
		}
		if err = fakeMounter.Unmount(c.targetPath); err != nil {
			t.Fatalf("Unmount device path %s failed: %v", c.targetPath, err)
		}
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	modes := plug.GetAccessModes()
	for _, v := range modes {
		if !volumetest.ContainsAccessMode(modes, v) {
			t.Errorf("Expected AccessModeTypes: %s", v)
		}
	}
}

func TestRequiresRemount(t *testing.T) {
	tmpDir, _ := utiltesting.MkTmpdir("rbd_test")
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))
	plug, _ := plugMgr.FindPluginByName("kubernetes.io/rbd")
	has := plug.RequiresRemount()
	if has {
		t.Errorf("Exepcted RequiresRemount to be false, got %t", has)
	}
}

func TestGetRbdImageSize(t *testing.T) {
	for i, c := range []struct {
		Output     string
		TargetSize int
	}{
		{
			Output:     `{"name":"kubernetes-dynamic-pvc-18e7a4d9-050d-11e9-b905-548998f3478f","size":10737418240,"objects":2560,"order":22,"object_size":4194304,"block_name_prefix":"rbd_data.9f4ff7238e1f29","format":2}`,
			TargetSize: 10240,
		},
		{
			Output:     `{"name":"kubernetes-dynamic-pvc-070635bf-e33f-11e8-aab7-548998f3478f","size":1073741824,"objects":256,"order":22,"object_size":4194304,"block_name_prefix":"rbd_data.670ac4238e1f29","format":2}`,
			TargetSize: 1024,
		},
	} {
		size, err := getRbdImageSize([]byte(c.Output))
		if err != nil {
			t.Errorf("Case %d: getRbdImageSize failed: %v", i, err)
			continue
		}
		if size != c.TargetSize {
			t.Errorf("Case %d: unexpected size, wanted %d, got %d", i, c.TargetSize, size)
		}
	}
}

func TestGetRbdImageInfo(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	for i, c := range []struct {
		DeviceMountPath    string
		TargetRbdImageInfo *rbdImageInfo
	}{
		{
			DeviceMountPath:    fmt.Sprintf("%s/plugins/kubernetes.io/rbd/rbd/pool1-image-image1", tmpDir),
			TargetRbdImageInfo: &rbdImageInfo{pool: "pool1", name: "image1"},
		},
		{
			DeviceMountPath:    fmt.Sprintf("%s/plugins/kubernetes.io/rbd/mounts/pool2-image-image2", tmpDir),
			TargetRbdImageInfo: &rbdImageInfo{pool: "pool2", name: "image2"},
		},
	} {
		rbdImageInfo, err := getRbdImageInfo(c.DeviceMountPath)
		if err != nil {
			t.Errorf("Case %d: getRbdImageInfo failed: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(rbdImageInfo, c.TargetRbdImageInfo) {
			t.Errorf("Case %d: unexpected RbdImageInfo, wanted %v, got %v", i, c.TargetRbdImageInfo, rbdImageInfo)
		}
	}
}
