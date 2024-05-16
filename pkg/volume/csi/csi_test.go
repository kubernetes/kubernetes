/*
Copyright 2019 The Kubernetes Authors.

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
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// TestCSI_VolumeAll runs a close approximation of volume workflow
// based on operations from the volume manager/reconciler/operation executor
func TestCSI_VolumeAll(t *testing.T) {
	defaultFSGroupPolicy := storage.ReadWriteOnceWithFSTypeFSGroupPolicy

	tests := []struct {
		name                 string
		specName             string
		driver               string
		volName              string
		specFunc             func(specName, driver, volName string) *volume.Spec
		podFunc              func() *api.Pod
		isInline             bool
		findPluginShouldFail bool
		driverSpec           *storage.CSIDriverSpec
		watchTimeout         time.Duration
	}{
		{
			name:     "PersistentVolume",
			specName: "pv2",
			driver:   "simple-driver",
			volName:  "vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromPersistentVolume(makeTestPV(specName, 20, driver, volName), false)
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
		},
		{
			name:     "PersistentVolume with driver info",
			specName: "pv2",
			driver:   "simple-driver",
			volName:  "vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromPersistentVolume(makeTestPV(specName, 20, driver, volName), false)
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			driverSpec: &storage.CSIDriverSpec{
				// Required for the driver to be accepted for the persistent volume.
				VolumeLifecycleModes: []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:     "PersistentVolume with wrong mode in driver info",
			specName: "pv2",
			driver:   "simple-driver",
			volName:  "vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromPersistentVolume(makeTestPV(specName, 20, driver, volName), false)
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			driverSpec: &storage.CSIDriverSpec{
				// This will cause the volume to be rejected.
				VolumeLifecycleModes: []storage.VolumeLifecycleMode{storage.VolumeLifecycleEphemeral},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:    "ephemeral inline supported",
			driver:  "inline-driver-1",
			volName: "test.vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromVolume(makeTestVol(specName, driver))
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			isInline: true,
			driverSpec: &storage.CSIDriverSpec{
				// Required for the driver to be accepted for the inline volume.
				VolumeLifecycleModes: []storage.VolumeLifecycleMode{storage.VolumeLifecycleEphemeral},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:    "ephemeral inline also supported",
			driver:  "inline-driver-1",
			volName: "test.vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromVolume(makeTestVol(specName, driver))
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			isInline: true,
			driverSpec: &storage.CSIDriverSpec{
				// Required for the driver to be accepted for the inline volume.
				VolumeLifecycleModes: []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent, storage.VolumeLifecycleEphemeral},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:    "ephemeral inline without CSIDriver info",
			driver:  "inline-driver-2",
			volName: "test.vol3",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromVolume(makeTestVol(specName, driver))
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			isInline: true,
		},
		{
			name:    "ephemeral inline with driver that has no mode",
			driver:  "inline-driver-3",
			volName: "test.vol4",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromVolume(makeTestVol(specName, driver))
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			isInline: true,
			driverSpec: &storage.CSIDriverSpec{
				// This means the driver *cannot* handle the inline volume because
				// the default is "persistent".
				VolumeLifecycleModes: nil,
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:    "ephemeral inline with driver that has wrong mode",
			driver:  "inline-driver-3",
			volName: "test.vol4",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return volume.NewSpecFromVolume(makeTestVol(specName, driver))
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			isInline: true,
			driverSpec: &storage.CSIDriverSpec{
				// This means the driver *cannot* handle the inline volume.
				VolumeLifecycleModes: []storage.VolumeLifecycleMode{storage.VolumeLifecyclePersistent},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		},
		{
			name:     "missing spec",
			specName: "pv2",
			driver:   "simple-driver",
			volName:  "vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return nil
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			findPluginShouldFail: true,
		},
		{
			name:     "incomplete spec",
			specName: "pv2",
			driver:   "simple-driver",
			volName:  "vol2",
			specFunc: func(specName, driver, volName string) *volume.Spec {
				return &volume.Spec{ReadOnly: true}
			},
			podFunc: func() *api.Pod {
				podUID := types.UID(fmt.Sprintf("%08X", rand.Uint64()))
				return &api.Pod{ObjectMeta: meta.ObjectMeta{UID: podUID, Namespace: testns}}
			},
			findPluginShouldFail: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("csi-test")
			if err != nil {
				t.Fatalf("can't create temp dir: %v", err)
			}
			defer os.RemoveAll(tmpDir)

			var driverInfo *storage.CSIDriver
			objs := []runtime.Object{}
			if test.driverSpec != nil {
				driverInfo = &storage.CSIDriver{
					ObjectMeta: meta.ObjectMeta{
						Name: test.driver,
					},
					Spec: *test.driverSpec,
				}
				objs = append(objs, driverInfo)
			}
			objs = append(objs, &api.Node{
				ObjectMeta: meta.ObjectMeta{
					Name: "fakeNode",
				},
				Spec: api.NodeSpec{},
			})

			client := fakeclient.NewSimpleClientset(objs...)

			factory := informers.NewSharedInformerFactory(client, time.Hour /* disable resync */)
			csiDriverInformer := factory.Storage().V1().CSIDrivers()
			volumeAttachmentInformer := factory.Storage().V1().VolumeAttachments()
			if driverInfo != nil {
				csiDriverInformer.Informer().GetStore().Add(driverInfo)
			}

			factory.Start(wait.NeverStop)
			factory.WaitForCacheSync(wait.NeverStop)

			attachDetachVolumeHost := volumetest.NewFakeAttachDetachVolumeHostWithCSINodeName(t,
				tmpDir,
				client,
				ProbeVolumePlugins(),
				"fakeNode",
				csiDriverInformer.Lister(),
				volumeAttachmentInformer.Lister(),
			)
			attachDetachPlugMgr := attachDetachVolumeHost.GetPluginMgr()
			csiClient := setupClient(t, true)

			volSpec := test.specFunc(test.specName, test.driver, test.volName)
			pod := test.podFunc()
			attachName := getAttachmentName(test.volName, test.driver, string(attachDetachVolumeHost.GetNodeName()))
			t.Log("csiTest.VolumeAll starting...")

			// *************** Attach/Mount volume resources ****************//
			// attach volume
			t.Log("csiTest.VolumeAll Attaching volume...")
			attachPlug, err := attachDetachPlugMgr.FindAttachablePluginBySpec(volSpec)
			if err != nil {
				if !test.findPluginShouldFail {
					t.Fatalf("csiTest.VolumeAll PluginManager.FindAttachablePluginBySpec failed: %v", err)
				} else {
					t.Log("csiTest.VolumeAll failed: ", err)
					return
				}
			}

			if test.isInline && attachPlug != nil {
				t.Fatal("csiTest.VolumeAll AttachablePlugin found with ephemeral volume")
			}
			if !test.isInline && attachPlug == nil {
				t.Fatal("csiTest.VolumeAll AttachablePlugin not found with PV")
			}

			var devicePath string
			if attachPlug != nil {
				t.Log("csiTest.VolumeAll attacher.Attach starting")

				var volAttacher volume.Attacher

				volAttacher, err := attachPlug.NewAttacher()
				if err != nil {
					t.Fatal("csiTest.VolumeAll failed to create new attacher: ", err)
				}

				// creates VolumeAttachment and blocks until it is marked attached (done by external attacher)
				go func() {
					attachID, err := volAttacher.Attach(volSpec, attachDetachVolumeHost.GetNodeName())
					if err != nil {
						t.Errorf("csiTest.VolumeAll attacher.Attach failed: %s", err)
						return
					}
					t.Logf("csiTest.VolumeAll got attachID %s", attachID)
				}()

				// Simulates external-attacher and marks VolumeAttachment.Status.Attached = true
				markVolumeAttached(t, attachDetachVolumeHost.GetKubeClient(), nil, attachName, storage.VolumeAttachmentStatus{Attached: true})

				// Observe attach on this node.
				devicePath, err = volAttacher.WaitForAttach(volSpec, "", pod, 500*time.Millisecond)
				if err != nil {
					t.Fatal("csiTest.VolumeAll attacher.WaitForAttach failed:", err)
				}

				if devicePath != attachName {
					t.Fatalf("csiTest.VolumeAll attacher.WaitForAttach got unexpected value %s", devicePath)
				}

				t.Log("csiTest.VolumeAll attacher.WaitForAttach succeeded OK, attachment ID:", devicePath)

			} else {
				t.Log("csiTest.VolumeAll volume attacher not found, skipping attachment")
			}

			// The reason for separate volume hosts here is because the attach/detach behavior is exclusive to the
			// CSI plugin running in the AttachDetachController. Similarly, the mount/unmount behavior is exclusive
			// to the CSI plugin running in the Kubelet.
			kubeletVolumeHost := volumetest.NewFakeKubeletVolumeHostWithCSINodeName(t,
				tmpDir,
				client,
				ProbeVolumePlugins(),
				"fakeNode",
				csiDriverInformer.Lister(),
				volumeAttachmentInformer.Lister(),
			)
			kubeletPlugMgr := kubeletVolumeHost.GetPluginMgr()

			// Mount Device
			t.Log("csiTest.VolumeAll Mouting device...")
			devicePlug, err := kubeletPlugMgr.FindDeviceMountablePluginBySpec(volSpec)
			if err != nil {
				t.Fatalf("csiTest.VolumeAll PluginManager.FindDeviceMountablePluginBySpec failed: %v", err)
			}

			if test.isInline && devicePlug != nil {
				t.Fatal("csiTest.VolumeAll DeviceMountablePlugin found with ephemeral volume")
			}
			if !test.isInline && devicePlug == nil {
				t.Fatal("csiTest.VolumeAll DeviceMountablePlugin not found with PV")
			}

			var devMounter volume.DeviceMounter
			if devicePlug != nil {
				devMounter, err = devicePlug.NewDeviceMounter()
				if err != nil {
					t.Fatal("csiTest.VolumeAll failed to create new device mounter: ", err)
				}
			}

			if devMounter != nil {
				csiDevMounter := getCsiAttacherFromDeviceMounter(devMounter, test.watchTimeout)
				csiDevMounter.csiClient = csiClient
				devMountPath, err := csiDevMounter.GetDeviceMountPath(volSpec)
				if err != nil {
					t.Fatalf("csiTest.VolumeAll deviceMounter.GetdeviceMountPath failed %s", err)
				}
				if err := csiDevMounter.MountDevice(volSpec, devicePath, devMountPath, volume.DeviceMounterArgs{}); err != nil {
					t.Fatalf("csiTest.VolumeAll deviceMounter.MountDevice failed: %v", err)
				}
				t.Log("csiTest.VolumeAll device mounted at path:", devMountPath)
			} else {
				t.Log("csiTest.VolumeAll DeviceMountablePlugin not found, skipping deviceMounter.MountDevice")
			}

			// mount volume
			t.Log("csiTest.VolumeAll Mouting volume...")
			volPlug, err := kubeletPlugMgr.FindPluginBySpec(volSpec)
			if err != nil || volPlug == nil {
				t.Fatalf("csiTest.VolumeAll PluginMgr.FindPluginBySpec failed: %v", err)
			}

			if volPlug == nil {
				t.Fatalf("csiTest.VolumeAll volumePlugin is nil")
			}

			if !volPlug.CanSupport(volSpec) {
				t.Fatal("csiTest.VolumeAll volumePlugin.CanSupport returned false")
			}

			mounter, err := volPlug.NewMounter(volSpec, pod, volume.VolumeOptions{})
			if err != nil || mounter == nil {
				t.Fatalf("csiTest.VolumeAll volPlugin.NewMounter is nil or error: %s", err)
			}

			var fsGroup *int64
			if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroup != nil {
				fsGroup = pod.Spec.SecurityContext.FSGroup
			}

			csiMounter := mounter.(*csiMountMgr)
			csiMounter.csiClient = csiClient
			var mounterArgs volume.MounterArgs
			mounterArgs.FsGroup = fsGroup
			err = csiMounter.SetUp(mounterArgs)
			if test.isInline && (test.driverSpec == nil || !containsVolumeMode(test.driverSpec.VolumeLifecycleModes, storage.VolumeLifecycleEphemeral)) {
				// This *must* fail because a CSIDriver.Spec.VolumeLifecycleModes entry "ephemeral"
				// is required.
				if err == nil {
					t.Fatalf("csiTest.VolumeAll volPlugin.NewMounter should have failed for inline volume due to lack of support for inline volumes, got: %+v, %s", mounter, err)
				}
				return
			}
			if !test.isInline && test.driverSpec != nil && !containsVolumeMode(test.driverSpec.VolumeLifecycleModes, storage.VolumeLifecyclePersistent) {
				// This *must* fail because a CSIDriver.Spec.VolumeLifecycleModes entry "persistent"
				// is required when a driver object is available.
				if err == nil {
					t.Fatalf("csiTest.VolumeAll volPlugin.NewMounter should have failed for persistent volume due to lack of support for persistent volumes, got: %+v, %s", mounter, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("csiTest.VolumeAll mounter.Setup(fsGroup) failed: %s", err)
			}
			t.Log("csiTest.VolumeAll mounter.Setup(fsGroup) done OK")

			dataFile := filepath.Join(filepath.Dir(mounter.GetPath()), volDataFileName)
			if _, err := os.Stat(dataFile); err != nil {
				t.Fatalf("csiTest.VolumeAll metadata JSON file not found: %s", dataFile)
			}
			t.Log("csiTest.VolumeAll JSON datafile generated OK:", dataFile)

			// ******** Volume Reconstruction ************* //
			volPath := filepath.Dir(csiMounter.GetPath())
			t.Log("csiTest.VolumeAll entering plugin.ConstructVolumeSpec for path", volPath)
			rec, err := volPlug.ConstructVolumeSpec(test.volName, volPath)
			if err != nil {
				t.Fatalf("csiTest.VolumeAll plugin.ConstructVolumeSpec failed: %s", err)
			} else {
				if rec.Spec == nil {
					t.Fatalf("csiTest.VolumeAll plugin.ConstructVolumeSpec returned nil spec")
				} else {
					volSpec = rec.Spec

					if test.isInline {
						if volSpec.Volume == nil || volSpec.Volume.CSI == nil {
							t.Fatal("csiTest.VolumeAll reconstruction of ephemeral volumeSpec missing CSI Volume source")
						}
						if volSpec.Volume.CSI.Driver == "" {
							t.Fatal("csiTest.VolumeAll reconstruction ephemral volume missing driver name")
						}
					} else {
						if volSpec.PersistentVolume == nil || volSpec.PersistentVolume.Spec.CSI == nil {
							t.Fatal("csiTest.VolumeAll reconstruction of volumeSpec missing CSI PersistentVolume source")
						}
						csi := volSpec.PersistentVolume.Spec.CSI
						if csi.Driver == "" {
							t.Fatal("csiTest.VolumeAll reconstruction of PV missing driver name")
						}
						if csi.VolumeHandle == "" {
							t.Fatal("csiTest.VolumeAll reconstruction of PV missing volume handle")
						}
					}
				}
			}

			// ************* Teardown everything **************** //
			t.Log("csiTest.VolumeAll Tearing down...")
			// unmount volume
			t.Log("csiTest.VolumeAll Unmouting volume...")
			volPlug, err = kubeletPlugMgr.FindPluginBySpec(volSpec)
			if err != nil || volPlug == nil {
				t.Fatalf("csiTest.VolumeAll PluginMgr.FindPluginBySpec failed: %v", err)
			}
			if volPlug == nil {
				t.Fatalf("csiTest.VolumeAll volumePlugin is nil")
			}
			mounter, err = volPlug.NewMounter(volSpec, pod, volume.VolumeOptions{})
			if err != nil || mounter == nil {
				t.Fatalf("csiTest.VolumeAll volPlugin.NewMounter is nil or error: %s", err)
			}

			unmounter, err := volPlug.NewUnmounter(test.specName, pod.GetUID())
			if err != nil {
				t.Fatal("csiTest.VolumeAll volumePlugin.NewUnmounter failed:", err)
			}
			csiUnmounter := unmounter.(*csiMountMgr)
			csiUnmounter.csiClient = csiClient

			if err := csiUnmounter.TearDownAt(mounter.GetPath()); err != nil {
				t.Fatal("csiTest.VolumeAll unmounter.TearDownAt failed:", err)
			}
			t.Log("csiTest.VolumeAll unmounter.TearDownAt done OK for dir:", mounter.GetPath())

			// unmount device
			t.Log("csiTest.VolumeAll Unmouting device...")
			devicePlug, err = kubeletPlugMgr.FindDeviceMountablePluginBySpec(volSpec)
			if err != nil {
				t.Fatalf("csiTest.VolumeAll failed to create mountable device plugin: %s", err)
			}

			if test.isInline && devicePlug != nil {
				t.Fatal("csiTest.VolumeAll DeviceMountablePlugin found with ephemeral volume")
			}
			if !test.isInline && devicePlug == nil {
				t.Fatal("csiTest.VolumeAll DeviceMountablePlugin not found with PV")
			}

			var devUnmounter volume.DeviceUnmounter
			if devicePlug != nil {
				t.Log("csiTest.VolumeAll found DeviceMountablePlugin, entering device unmouting ...")
				devMounter, err = devicePlug.NewDeviceMounter()
				if err != nil {
					t.Fatal("csiTest.VolumeAll failed to create new device mounter: ", err)
				}
				devUnmounter, err = devicePlug.NewDeviceUnmounter()
				if err != nil {
					t.Fatal("csiTest.VolumeAll failed to create new device unmounter: ", err)
				}

				if devMounter != nil && devUnmounter != nil {
					csiDevMounter := getCsiAttacherFromDeviceMounter(devMounter, test.watchTimeout)
					csiDevUnmounter := getCsiAttacherFromDeviceUnmounter(devUnmounter, test.watchTimeout)
					csiDevUnmounter.csiClient = csiClient

					devMountPath, err := csiDevMounter.GetDeviceMountPath(volSpec)
					if err != nil {
						t.Fatalf("csiTest.VolumeAll deviceMounter.GetdeviceMountPath failed %s", err)
					}
					if err := csiDevUnmounter.UnmountDevice(devMountPath); err != nil {
						t.Fatalf("csiTest.VolumeAll deviceMounter.UnmountDevice failed: %s", err)
					}
					t.Log("csiTest.VolumeAll deviceUnmounter.UnmountDevice done OK for path", devMountPath)
				}
			} else {
				t.Log("csiTest.VolumeAll DeviceMountablePluginBySpec did not find a plugin, skipping unmounting.")
			}

			// detach volume
			t.Log("csiTest.VolumeAll Detaching volume...")
			attachPlug, err = attachDetachPlugMgr.FindAttachablePluginBySpec(volSpec)
			if err != nil {
				t.Fatalf("csiTest.VolumeAll PluginManager.FindAttachablePluginBySpec failed: %v", err)
			}

			if test.isInline && attachPlug != nil {
				t.Fatal("csiTest.VolumeAll AttachablePlugin found with ephemeral volume")
			}
			if !test.isInline && attachPlug == nil {
				t.Fatal("csiTest.VolumeAll AttachablePlugin not found with PV")
			}

			if attachPlug != nil {
				volDetacher, err := attachPlug.NewDetacher()
				if err != nil {
					t.Fatal("csiTest.VolumeAll failed to create new detacher: ", err)
				}

				t.Log("csiTest.VolumeAll preparing detacher.Detach...")
				volName, err := volPlug.GetVolumeName(volSpec)
				if err != nil {
					t.Fatal("csiTest.VolumeAll volumePlugin.GetVolumeName failed:", err)
				}
				csiDetacher := getCsiAttacherFromVolumeDetacher(volDetacher, test.watchTimeout)
				csiDetacher.csiClient = csiClient
				if err := csiDetacher.Detach(volName, attachDetachVolumeHost.GetNodeName()); err != nil {
					t.Fatal("csiTest.VolumeAll detacher.Detach failed:", err)
				}
				t.Log("csiTest.VolumeAll detacher.Detach succeeded for volume", volName)

			} else {
				t.Log("csiTest.VolumeAll attachable plugin not found for plugin.Detach call, skipping")
			}
		})
	}
}
