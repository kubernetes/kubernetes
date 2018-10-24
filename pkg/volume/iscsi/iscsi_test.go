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

package iscsi

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("iscsi_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/iscsi" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{ISCSI: &v1.ISCSIVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{}}}) {
		t.Errorf("Expected false")
	}
	if plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{}}}}) {
		t.Errorf("Expected false")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{ISCSI: &v1.ISCSIPersistentVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("iscsi_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/iscsi")
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

func NewFakeDiskManager() *fakeDiskManager {
	return &fakeDiskManager{
		tmpDir: utiltesting.MkTmpdirOrDie("iscsi_test"),
	}
}

func (fake *fakeDiskManager) Cleanup() {
	os.RemoveAll(fake.tmpDir)
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk iscsiDisk) string {
	return fake.tmpDir
}

func (fake *fakeDiskManager) MakeGlobalVDPDName(disk iscsiDisk) string {
	return fake.tmpDir
}

func (fake *fakeDiskManager) AttachDisk(b iscsiDiskMounter) (string, error) {
	globalPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return "", err
	}
	// Simulate the global mount so that the fakeMounter returns the
	// expected number of mounts for the attached disk.
	b.mounter.Mount(globalPath, globalPath, b.fsType, nil)

	return "/dev/sdb", nil
}

func (fake *fakeDiskManager) DetachDisk(c iscsiDiskUnmounter, mntPath string) error {
	globalPath := c.manager.MakeGlobalPDName(*c.iscsiDisk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) DetachBlockISCSIDisk(c iscsiDiskUnmapper, mntPath string) error {
	globalPath := c.manager.MakeGlobalVDPDName(*c.iscsiDisk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("iscsi_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeManager := NewFakeDiskManager()
	defer fakeManager.Cleanup()
	fakeMounter := &mount.FakeMounter{}
	fakeExec := mount.NewFakeExec(nil)
	mounter, err := plug.(*iscsiPlugin).newMounterInternal(spec, types.UID("poduid"), fakeManager, fakeMounter, fakeExec, nil)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Error("Got a nil Mounter")
	}

	path := mounter.GetPath()
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~iscsi/vol1", tmpDir)
	if path != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, path)
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

	fakeManager2 := NewFakeDiskManager()
	defer fakeManager2.Cleanup()
	unmounter, err := plug.(*iscsiPlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager2, fakeMounter, fakeExec)
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
}

func TestPluginVolume(t *testing.T) {
	vol := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal: "127.0.0.1:3260",
				IQN:          "iqn.2014-12.server:storage.target01",
				FSType:       "ext4",
				Lun:          0,
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}

func TestPluginPersistentVolume(t *testing.T) {
	vol := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "vol1",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				ISCSI: &v1.ISCSIPersistentVolumeSource{
					TargetPortal: "127.0.0.1:3260",
					IQN:          "iqn.2014-12.server:storage.target01",
					FSType:       "ext4",
					Lun:          0,
				},
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("iscsi_test")
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
				ISCSI: &v1.ISCSIPersistentVolumeSource{
					TargetPortal: "127.0.0.1:3260",
					IQN:          "iqn.2014-12.server:storage.target01",
					FSType:       "ext4",
					Lun:          0,
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
	plug, _ := plugMgr.FindPluginByName(iscsiPluginName)

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

func TestPortalMounter(t *testing.T) {
	if portal := portalMounter("127.0.0.1"); portal != "127.0.0.1:3260" {
		t.Errorf("wrong portal: %s", portal)
	}
	if portal := portalMounter("127.0.0.1:3260"); portal != "127.0.0.1:3260" {
		t.Errorf("wrong portal: %s", portal)
	}
}

type testcase struct {
	name      string
	defaultNs string
	spec      *volume.Spec
	// Expected return of the test
	expectedName          string
	expectedNs            string
	expectedIface         string
	expectedError         error
	expectedDiscoveryCHAP bool
	expectedSessionCHAP   bool
}

func TestGetSecretNameAndNamespaceForPV(t *testing.T) {
	tests := []testcase{
		{
			name:      "persistent volume source",
			defaultNs: "default",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							ISCSI: &v1.ISCSIPersistentVolumeSource{
								TargetPortal: "127.0.0.1:3260",
								IQN:          "iqn.2014-12.server:storage.target01",
								FSType:       "ext4",
								Lun:          0,
								SecretRef: &v1.SecretReference{
									Name:      "name",
									Namespace: "ns",
								},
							},
						},
					},
				},
			},
			expectedName:  "name",
			expectedNs:    "ns",
			expectedError: nil,
		},
		{
			name:      "persistent volume source without namespace",
			defaultNs: "default",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							ISCSI: &v1.ISCSIPersistentVolumeSource{
								TargetPortal: "127.0.0.1:3260",
								IQN:          "iqn.2014-12.server:storage.target01",
								FSType:       "ext4",
								Lun:          0,
								SecretRef: &v1.SecretReference{
									Name: "name",
								},
							},
						},
					},
				},
			},
			expectedName:  "name",
			expectedNs:    "default",
			expectedError: nil,
		},
		{
			name:      "pod volume source",
			defaultNs: "default",
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						ISCSI: &v1.ISCSIVolumeSource{
							TargetPortal: "127.0.0.1:3260",
							IQN:          "iqn.2014-12.server:storage.target01",
							FSType:       "ext4",
							Lun:          0,
						},
					},
				},
			},
			expectedName:  "",
			expectedNs:    "",
			expectedError: nil,
		},
	}
	for _, testcase := range tests {
		resultName, resultNs, err := getISCSISecretNameAndNamespace(testcase.spec, testcase.defaultNs)
		if err != testcase.expectedError || resultName != testcase.expectedName || resultNs != testcase.expectedNs {
			t.Errorf("%s failed: expected err=%v ns=%q name=%q, got %v/%q/%q", testcase.name, testcase.expectedError, testcase.expectedNs, testcase.expectedName,
				err, resultNs, resultName)
		}
	}

}

func TestGetISCSIInitiatorInfo(t *testing.T) {
	tests := []testcase{
		{
			name: "persistent volume source",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							ISCSI: &v1.ISCSIPersistentVolumeSource{
								TargetPortal:   "127.0.0.1:3260",
								IQN:            "iqn.2014-12.server:storage.target01",
								FSType:         "ext4",
								Lun:            0,
								ISCSIInterface: "tcp",
							},
						},
					},
				},
			},
			expectedIface: "tcp",
			expectedError: nil,
		},
		{
			name: "pod volume source",
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						ISCSI: &v1.ISCSIVolumeSource{
							TargetPortal:   "127.0.0.1:3260",
							IQN:            "iqn.2014-12.server:storage.target01",
							FSType:         "ext4",
							Lun:            0,
							ISCSIInterface: "tcp",
						},
					},
				},
			},
			expectedIface: "tcp",
			expectedError: nil,
		},
	}
	for _, testcase := range tests {
		resultIface, _, err := getISCSIInitiatorInfo(testcase.spec)
		if err != testcase.expectedError || resultIface != testcase.expectedIface {
			t.Errorf("%s failed: expected err=%v iface=%s, got %v/%s", testcase.name, testcase.expectedError, testcase.expectedIface,
				err, resultIface)
		}
	}
}

func TestGetISCSICHAP(t *testing.T) {
	tests := []testcase{
		{
			name: "persistent volume source",
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							ISCSI: &v1.ISCSIPersistentVolumeSource{
								DiscoveryCHAPAuth: true,
								SessionCHAPAuth:   true,
							},
						},
					},
				},
			},
			expectedDiscoveryCHAP: true,
			expectedSessionCHAP:   true,
			expectedError:         nil,
		},
		{
			name: "pod volume source",
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						ISCSI: &v1.ISCSIVolumeSource{
							DiscoveryCHAPAuth: true,
							SessionCHAPAuth:   true,
						},
					},
				},
			},
			expectedDiscoveryCHAP: true,
			expectedSessionCHAP:   true,
			expectedError:         nil,
		},
		{
			name:                  "no volume",
			spec:                  &volume.Spec{},
			expectedDiscoveryCHAP: false,
			expectedSessionCHAP:   false,
			expectedError:         fmt.Errorf("Spec does not reference an ISCSI volume type"),
		},
	}
	for _, testcase := range tests {
		resultDiscoveryCHAP, err := getISCSIDiscoveryCHAPInfo(testcase.spec)
		resultSessionCHAP, err := getISCSISessionCHAPInfo(testcase.spec)
		switch testcase.name {
		case "no volume":
			if err.Error() != testcase.expectedError.Error() || resultDiscoveryCHAP != testcase.expectedDiscoveryCHAP || resultSessionCHAP != testcase.expectedSessionCHAP {
				t.Errorf("%s failed: expected err=%v DiscoveryCHAP=%v SessionCHAP=%v, got %v/%v/%v",
					testcase.name, testcase.expectedError, testcase.expectedDiscoveryCHAP, testcase.expectedSessionCHAP,
					err, resultDiscoveryCHAP, resultSessionCHAP)
			}
		default:
			if err != testcase.expectedError || resultDiscoveryCHAP != testcase.expectedDiscoveryCHAP || resultSessionCHAP != testcase.expectedSessionCHAP {
				t.Errorf("%s failed: expected err=%v DiscoveryCHAP=%v SessionCHAP=%v, got %v/%v/%v", testcase.name, testcase.expectedError, testcase.expectedDiscoveryCHAP, testcase.expectedSessionCHAP,
					err, resultDiscoveryCHAP, resultSessionCHAP)
			}
		}
	}
}

func TestGetVolumeSpec(t *testing.T) {
	path := "plugins/kubernetes.io/iscsi/volumeDevices/iface-default/127.0.0.1:3260-iqn.2014-12.server:storage.target01-lun-0"
	spec, _ := getVolumeSpecFromGlobalMapPath("test", path)

	portal := spec.PersistentVolume.Spec.PersistentVolumeSource.ISCSI.TargetPortal
	if portal != "127.0.0.1:3260" {
		t.Errorf("wrong portal: %v", portal)
	}
	iqn := spec.PersistentVolume.Spec.PersistentVolumeSource.ISCSI.IQN
	if iqn != "iqn.2014-12.server:storage.target01" {
		t.Errorf("wrong iqn: %v", iqn)
	}
	lun := spec.PersistentVolume.Spec.PersistentVolumeSource.ISCSI.Lun
	if lun != 0 {
		t.Errorf("wrong lun: %v", lun)
	}
	iface := spec.PersistentVolume.Spec.PersistentVolumeSource.ISCSI.ISCSIInterface
	if iface != "default" {
		t.Errorf("wrong ISCSIInterface: %v", iface)
	}
}

func TestGetVolumeSpec_no_lun(t *testing.T) {
	path := "plugins/kubernetes.io/iscsi/volumeDevices/iface-default/127.0.0.1:3260-iqn.2014-12.server:storage.target01"
	_, err := getVolumeSpecFromGlobalMapPath("test", path)
	if !strings.Contains(err.Error(), "malformatted mnt path") {
		t.Errorf("should get error: malformatted mnt path")
	}
}

func TestGetVolumeSpec_no_iface(t *testing.T) {
	path := "plugins/kubernetes.io/iscsi/volumeDevices/default/127.0.0.1:3260-iqn.2014-12.server:storage.target01-lun-0"
	_, err := getVolumeSpecFromGlobalMapPath("test", path)
	if !strings.Contains(err.Error(), "failed to retrieve iface") {
		t.Errorf("should get error: failed to retrieve iface")
	}
}
