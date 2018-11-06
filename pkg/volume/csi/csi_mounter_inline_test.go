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
	"os"
	"reflect"
	"testing"
	"time"

	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestInlineProvisioning(t *testing.T) {
	fakeClient := fakeclient.NewSimpleClientset()
	plug, tmpDir := newTestPlugin(t, fakeClient, nil)
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name          string
		volName       string
		volSize       int
		namespace     string
		handleName    string
		pvcVol        *api.PersistentVolumeClaimVolumeSource
		csiCrlSecrets *api.LocalObjectReference

		shouldFail bool
	}{
		{
			name:      "normal provision with claim",
			volName:   "test-vol",
			volSize:   10,
			namespace: "test-ns",
			pvcVol: &api.PersistentVolumeClaimVolumeSource{
				ClaimName: "test-vol-pvc",
			},
		},
		{
			name:      "normal provision with no claim",
			volName:   "test-vol",
			volSize:   0,
			namespace: "test-ns",
		},
	}

	for _, tc := range testCases {
		volSource := makeTestVolumeSource(tc.volName, "test-driver", tc.handleName)
		volSource.PersistentVolumeClaim = tc.pvcVol
		volSpec := &volume.Spec{Volume: volSource}

		mounter, err := plug.NewMounter(
			volSpec,
			&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: tc.namespace}},
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

		// create related PVC
		if tc.pvcVol != nil {
			pvcName := tc.pvcVol.ClaimName
			pvc := makeTestPVC(pvcName, tc.volSize)
			_, err = csiMounter.k8s.CoreV1().PersistentVolumeClaims(tc.namespace).Create(pvc)
			if err != nil {
				t.Fatalf("failed to create PVC %s: %v", pvcName, err)
			}
		}

		createdVol, err := csiMounter.inlineProvision(tc.volName, tc.namespace, volSource)
		if err != nil {
			t.Fatalf("failed to provision inline volume: %v", err)
		}

		if createdVol.CapacityBytes != int64(util.GIB*tc.volSize) {
			t.Fatalf("unexpected provisioned volume size %d", createdVol.CapacityBytes)
		}

		// validation
		createdVols := csiMounter.csiClient.(*fakeCsiDriverClient).controllerClient.GetControllerCreatedVolumes()
		if createdVols == nil || len(createdVols) == 0 {
			t.Fatal("inline volumes not being provisioned OK")
		}
	}

}

func TestInlineAttach(t *testing.T) {
	testCases := []struct {
		name       string
		volHandle  string
		csiSource  *api.CSIVolumeSource
		attached   bool
		shouldFail bool
	}{
		{
			name:       "missing csi source",
			volHandle:  "test-vol-handle-0",
			csiSource:  nil,
			shouldFail: true,
		},
		{
			name:      "attached timeout",
			volHandle: "test-vol-handle-1",
			csiSource: &api.CSIVolumeSource{
				Driver: "test-driver",
			},
			attached:   false,
			shouldFail: true,
		},
		{
			name:      "attached OK",
			volHandle: "test-vol-handle-2",
			csiSource: &api.CSIVolumeSource{
				Driver: "test-driver",
			},
			attached:   true,
			shouldFail: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			t.Logf("running test %s", tc.name)
			volSource := makeTestVolumeSource("test-vol", "test-driver", tc.volHandle)
			volSpec := &volume.Spec{Volume: volSource}

			mounter, err := plug.NewMounter(
				volSpec,
				&api.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: "test-ns"}},
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

			nodeName := string(plug.host.GetNodeName())
			driverName := volSource.CSI.Driver
			attachID := getAttachmentName(tc.volHandle, driverName, nodeName)
			status := storage.VolumeAttachmentStatus{Attached: tc.attached}

			go func() {
				markVolumeAttached(t, csiMounter.k8s, fakeWatcher, attachID, status)
			}()

			inlineAttachmentID, err := csiMounter.inlineAttach(tc.volHandle, tc.csiSource, time.Millisecond*50)

			if tc.shouldFail && err == nil {
				t.Fatal("expecting failure, but err is nil")

			}
			if !tc.shouldFail && err != nil {
				t.Fatalf("unexpected failure: %v", err)
			}

			if err == nil {
				if attachID != inlineAttachmentID {
					t.Fatalf("unexpected attachment ID %s", inlineAttachmentID)
				}
				attachment, err := csiMounter.k8s.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
				if err != nil {
					if !apierrs.IsNotFound(err) {
						t.Fatalf("unexpected err: %v", err)
					}
					t.Fatalf("attachment not found")
				}
				if attachment == nil {
					t.Error("expecting attachment not to be nil, but it is")
				}
				if !attachment.Status.Attached {
					t.Errorf("unexpected attachment status of %t", attachment.Status.Attached)
				}
			}
		})
	}
}

// tests
func TestInlineSetUp(t *testing.T) {
	volHandle := "test-vol-handle"
	readOnly := false
	//nodeName := string(plug.host.GetNodeName())

	tests := []struct {
		name       string
		volSpec    *volume.Spec
		volAttribs map[string]string
		shouldFail bool
	}{
		{
			name: "inline - dynamic provision with missing volHandle",
			volSpec: &volume.Spec{
				Volume: &api.Volume{
					Name: testVol,
					VolumeSource: api.VolumeSource{
						CSI: &api.CSIVolumeSource{
							Driver:   testDriver,
							ReadOnly: &readOnly,
						},
					},
				},
			},
		},
		{
			name:       "inline - dynamic provision with volume attribs",
			volAttribs: map[string]string{"foo0": "bar0", "foo1": "bar1"},
			volSpec: &volume.Spec{
				Volume: &api.Volume{
					Name: testVol,
					VolumeSource: api.VolumeSource{
						CSI: &api.CSIVolumeSource{
							Driver:   testDriver,
							ReadOnly: &readOnly,
						},
					},
				},
			},
		},
		{
			name:       "inline - dynamic provision with secrets",
			volAttribs: map[string]string{"foo0": "bar0", "foo1": "bar1"},
			volSpec: &volume.Spec{
				Volume: &api.Volume{
					Name: testVol,
					VolumeSource: api.VolumeSource{
						CSI: &api.CSIVolumeSource{
							Driver:               testDriver,
							ReadOnly:             &readOnly,
							NodePublishSecretRef: &api.LocalObjectReference{Name: "test-secret"},
						},
					},
				},
			},
		},
		{
			name: "inline - pre-provisioned with volHandle",
			volSpec: &volume.Spec{
				Volume: &api.Volume{
					Name: testVol,
					VolumeSource: api.VolumeSource{
						CSI: &api.CSIVolumeSource{
							Driver:       testDriver,
							VolumeHandle: &volHandle,
							ReadOnly:     &readOnly,
						},
					},
				},
			},
		},
		{
			name: "pvc test - normal",
			volSpec: &volume.Spec{
				PersistentVolume: makeTestPV("test-pv", 5, testDriver, volHandle),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeClient := fakeclient.NewSimpleClientset()
			plug, tmpDir := newTestPlugin(t, fakeClient, nil)
			defer os.RemoveAll(tmpDir)

			t.Logf("running test: %s", tc.name)
			mounter, err := plug.NewMounter(
				tc.volSpec,
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
			k8sClient := csiMounter.k8s
			csiInlineUsed := false

			// Resolve info based on volume spec passed (CSIVolume or CSIPersistentVolume)
			if tc.volSpec.Volume != nil {
				csiInlineUsed = true
				tc.volSpec.Volume.CSI.VolumeAttributes = tc.volAttribs
				// setup a secret if necessary
				if tc.volSpec.Volume.CSI.NodePublishSecretRef != nil {
					secretName := tc.volSpec.Volume.CSI.NodePublishSecretRef.Name
					if _, err := k8sClient.CoreV1().Secrets(testns).Create(&api.Secret{
						ObjectMeta: meta.ObjectMeta{
							Namespace: testns,
							Name:      secretName,
						},
						StringData: map[string]string{"sec0": "val0"},
					}); err != nil {
						t.Error(err)
					}
				}

				go func() {
					var attachment *storage.VolumeAttachment
					ticker := time.NewTicker(10 * time.Millisecond)
					defer ticker.Stop()
					// wait for an attachment to show up
					for i := 0; i < 100; i++ {
						l, err := k8sClient.StorageV1beta1().VolumeAttachments().List(meta.ListOptions{})
						if err != nil {
							t.Error(err)
							break
						}
						if len(l.Items) > 0 {
							attachment = &l.Items[0]
							t.Logf("go an attachment from list [ID: %s]", attachment.Name)
							break
						} else {
							<-ticker.C
						}
					}
					if attachment != nil {
						t.Logf("updating attachment to attached=true [ID: %s]", attachment.Name)
						attachment.Status.Attached = true
						_, err := k8sClient.StorageV1beta1().VolumeAttachments().Update(attachment)
						if err != nil {
							t.Error(err)
						}
					}
				}()
			} else if tc.volSpec.PersistentVolume != nil {
				csiInlineUsed = false
				tc.volSpec.PersistentVolume.Spec.CSI.VolumeAttributes = tc.volAttribs
				// setup secret if needed
				if tc.volSpec.PersistentVolume.Spec.CSI.NodePublishSecretRef != nil {
					secretRef := tc.volSpec.PersistentVolume.Spec.CSI.NodePublishSecretRef
					if _, err := k8sClient.CoreV1().Secrets(secretRef.Namespace).Create(&api.Secret{
						ObjectMeta: meta.ObjectMeta{
							Namespace: secretRef.Namespace,
							Name:      secretRef.Name,
						},
						StringData: map[string]string{"sec0": "val0"},
					}); err != nil {
						t.Error(err)
					}
				}

				go func() {
					// wait for attachment
					attachID := getAttachmentName(csiMounter.volumeID, csiMounter.driverName, string(plug.host.GetNodeName()))
					attachment := makeTestAttachment(attachID, string(plug.host.GetNodeName()), tc.volSpec.PersistentVolume.Name)
					attachment.Status.Attached = true
					_, err = csiMounter.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
					if err != nil {
						t.Fatalf("failed to create VolumeAttachment: %v", err)
					}
				}()
			} else {
				t.Error("missing volume.Spec")
			}

			if err := csiMounter.SetUp(nil); err != nil {
				t.Errorf("mounter.Setup failed: %v", err)
			}
			if err != nil && !tc.shouldFail {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && tc.shouldFail {
				t.Fatalf("expecting error, but got err == nil")
			}
			if tc.volAttribs != nil && reflect.DeepEqual(tc.volAttribs, csiMounter.volumeInfo) {
				t.Errorf("Volume attributes not passed to VolumeAttachment. Expecting %#v, got %#v", tc.volAttribs, csiMounter.volumeInfo)
			}
			pubs := csiMounter.csiClient.(*fakeCsiDriverClient).nodeClient.GetNodePublishedVolumes()
			vol, ok := pubs[csiMounter.volumeID]
			if !ok {
				t.Error("csi server may not have received NodePublishVolume call")
			}
			if csiInlineUsed {
				if tc.volSpec.Volume.CSI.NodePublishSecretRef != nil && vol.NodePublishSecrets == nil {
					t.Error("inline NodePublishSecretRef provided, but secrets not sent to driver")
				}
			} else {
				if tc.volSpec.PersistentVolume.Spec.CSI.NodePublishSecretRef != nil && vol.NodePublishSecrets == nil {
					t.Error("CSI PV NodePublishSecretRef provided, but secrets not sent to driver")
				}
			}
		})
	}
}
