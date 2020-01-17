/*
Copyright 2015 The Kubernetes Authors.

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

package fc

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"k8s.io/utils/exec/testing"
	"k8s.io/utils/mount"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("fc_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/fc")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/fc" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{FC: &v1.FCVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{}}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{FC: &v1.FCVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("fc_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/fc")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) || !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", v1.ReadWriteOnce, v1.ReadOnlyMany)
	}
}

type fakeDiskManager struct {
	tmpDir       string
	attachCalled bool
	detachCalled bool
}

func newFakeDiskManager() *fakeDiskManager {
	return &fakeDiskManager{
		tmpDir: utiltesting.MkTmpdirOrDie("fc_test"),
	}
}

func (fake *fakeDiskManager) Cleanup() {
	os.RemoveAll(fake.tmpDir)
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk fcDisk) string {
	return fake.tmpDir
}

func (fake *fakeDiskManager) MakeGlobalVDPDName(disk fcDisk) string {
	return fake.tmpDir
}

func (fake *fakeDiskManager) AttachDisk(b fcDiskMounter) (string, error) {
	globalPath := b.manager.MakeGlobalPDName(*b.fcDisk)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return "", err
	}
	// Simulate the global mount so that the fakeMounter returns the
	// expected number of mounts for the attached disk.
	b.mounter.Mount(globalPath, globalPath, b.fsType, nil)

	fake.attachCalled = true
	return "", nil
}

func (fake *fakeDiskManager) DetachDisk(c fcDiskUnmounter, mntPath string) error {
	globalPath := c.manager.MakeGlobalPDName(*c.fcDisk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	fake.detachCalled = true
	return nil
}

func (fake *fakeDiskManager) DetachBlockFCDisk(c fcDiskUnmapper, mapPath, devicePath string) error {
	err := os.RemoveAll(mapPath)
	if err != nil {
		return err
	}
	fake.detachCalled = true
	return nil
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("fc_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/fc")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeManager := newFakeDiskManager()
	defer fakeManager.Cleanup()
	fakeMounter := mount.NewFakeMounter(nil)
	fakeExec := &testingexec.FakeExec{}
	mounter, err := plug.(*fcPlugin).newMounterInternal(spec, types.UID("poduid"), fakeManager, fakeMounter, fakeExec)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter: %v", err)
	}

	path := mounter.GetPath()
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~fc/vol1", tmpDir)
	if path != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, path)
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

	fakeManager2 := newFakeDiskManager()
	defer fakeManager2.Cleanup()
	unmounter, err := plug.(*fcPlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager2, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter: %v", err)
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("TearDown() failed: %v", err)
	}
}

func doTestPluginNilMounter(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("fc_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/fc")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeManager := newFakeDiskManager()
	defer fakeManager.Cleanup()
	fakeMounter := mount.NewFakeMounter(nil)
	fakeExec := &testingexec.FakeExec{}
	mounter, err := plug.(*fcPlugin).newMounterInternal(spec, types.UID("poduid"), fakeManager, fakeMounter, fakeExec)
	if err == nil {
		t.Errorf("Error failed to make a new Mounter is expected: %v", err)
	}
	if mounter != nil {
		t.Errorf("A nil Mounter is expected: %v", err)
	}
}

func TestPluginVolume(t *testing.T) {
	lun := int32(0)
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			FC: &v1.FCVolumeSource{
				TargetWWNs: []string{"500a0981891b8dc5"},
				FSType:     "ext4",
				Lun:        &lun,
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolume(t *testing.T) {
	lun := int32(0)
	fs := v1.PersistentVolumeFilesystem
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				FC: &v1.FCVolumeSource{
					TargetWWNs: []string{"500a0981891b8dc5"},
					FSType:     "ext4",
					Lun:        &lun,
				},
			},
			VolumeMode: &fs,
		},
	}
	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPluginVolumeWWIDs(t *testing.T) {
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			FC: &v1.FCVolumeSource{
				WWIDs:  []string{"3600508b400105e210000900000490000"},
				FSType: "ext4",
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolumeWWIDs(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				FC: &v1.FCVolumeSource{
					WWIDs:  []string{"3600508b400105e21 000900000490000"},
					FSType: "ext4",
				},
			},
			VolumeMode: &fs,
		},
	}
	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPluginVolumeNoDiskInfo(t *testing.T) {
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			FC: &v1.FCVolumeSource{
				FSType: "ext4",
			},
		},
	}
	doTestPluginNilMounter(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolumeNoDiskInfo(t *testing.T) {
	fs := v1.PersistentVolumeFilesystem
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				FC: &v1.FCVolumeSource{
					FSType: "ext4",
				},
			},
			VolumeMode: &fs,
		},
	}
	doTestPluginNilMounter(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("fc_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	lun := int32(0)
	fs := v1.PersistentVolumeFilesystem
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				FC: &v1.FCVolumeSource{
					TargetWWNs: []string{"some_wwn"},
					FSType:     "ext4",
					Lun:        &lun,
				},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
			VolumeMode: &fs,
		},
	}

	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
			VolumeMode: &fs,
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, client, nil))
	plug, _ := plugMgr.FindPluginByName(fcPluginName)

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

func Test_getWwnsLun(t *testing.T) {
	num := int32(0)
	fc := &v1.FCVolumeSource{
		TargetWWNs: []string{"500a0981891b8dc5"},
		FSType:     "ext4",
		Lun:        &num,
	}
	wwn, lun, _, err := getWwnsLunWwids(fc)
	// if no wwn and lun, exit
	if (len(wwn) == 0 && lun != "0") || err != nil {
		t.Errorf("no fc disk found")
	}
}

func Test_getWwids(t *testing.T) {
	fc := &v1.FCVolumeSource{
		FSType: "ext4",
		WWIDs:  []string{"3600508b400105e210000900000490000"},
	}
	_, _, wwid, err := getWwnsLunWwids(fc)
	// if no wwn and lun, exit
	if len(wwid) == 0 || err != nil {
		t.Errorf("no fc disk found")
	}
}

func Test_getWwnsLunWwidsError(t *testing.T) {
	fc := &v1.FCVolumeSource{
		FSType: "ext4",
	}
	wwn, lun, wwid, err := getWwnsLunWwids(fc)
	// expected no wwn and lun and wwid
	if (len(wwn) != 0 && lun != "" && len(wwid) != 0) || err == nil {
		t.Errorf("unexpected fc disk found")
	}
}

func Test_ConstructVolumeSpec(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skipf("Test_ConstructVolumeSpec is not supported on GOOS=%s", runtime.GOOS)
	}
	fm := mount.NewFakeMounter(
		[]mount.MountPoint{
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod1"},
			{Device: "/dev/sdb", Path: "/var/lib/kubelet/plugins/kubernetes.io/fc/50060e801049cfd1-lun-0"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod2"},
			{Device: "/dev/sdc", Path: "/var/lib/kubelet/plugins/kubernetes.io/fc/volumeDevices/3600508b400105e210000900000490000"},
		})
	mountPaths := []string{
		"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod1",
		"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod2",
	}
	for _, path := range mountPaths {
		refs, err := fm.GetMountRefs(path)
		if err != nil {
			t.Errorf("couldn't get mountrefs. err: %v", err)
		}
		var globalPDPath string
		for _, ref := range refs {
			if strings.Contains(ref, "kubernetes.io/fc") {
				globalPDPath = ref
				break
			}
		}
		if len(globalPDPath) == 0 {
			t.Errorf("couldn't fetch mountrefs")
		}
		arr := strings.Split(globalPDPath, "/")
		if len(arr) < 1 {
			t.Errorf("failed to retrieve volume plugin information from globalPDPath: %v", globalPDPath)
		}
		volumeInfo := arr[len(arr)-1]
		if strings.Contains(volumeInfo, "-lun-") {
			wwnLun := strings.Split(volumeInfo, "-lun-")
			if len(wwnLun) < 2 {
				t.Errorf("failed to retrieve TargetWWN and Lun. volumeInfo is invalid: %v", volumeInfo)
			}
			lun, _ := strconv.Atoi(wwnLun[1])
			lun32 := int32(lun)
			if wwnLun[0] != "50060e801049cfd1" || lun32 != 0 {
				t.Errorf("failed to retrieve TargetWWN and Lun")
			}
		} else {
			if volumeInfo != "3600508b400105e210000900000490000" {
				t.Errorf("failed to retrieve WWIDs")
			}
		}
	}
}

func Test_ConstructVolumeSpecNoRefs(t *testing.T) {
	fm := mount.NewFakeMounter(
		[]mount.MountPoint{
			{Device: "/dev/sdd", Path: "/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod1"},
		})
	mountPaths := []string{
		"/var/lib/kubelet/pods/some-pod/volumes/kubernetes.io~fc/fc-in-pod1",
	}
	for _, path := range mountPaths {
		refs, _ := fm.GetMountRefs(path)
		var globalPDPath string
		for _, ref := range refs {
			if strings.Contains(ref, "kubernetes.io/fc") {
				globalPDPath = ref
				break
			}
		}
		if len(globalPDPath) != 0 {
			t.Errorf("invalid globalPDPath")
		}
	}
}
