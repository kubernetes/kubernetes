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
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	storageinformer "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	storagelister "k8s.io/client-go/listers/storage/v1"
	core "k8s.io/client-go/testing"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	fakecsi "k8s.io/kubernetes/pkg/volume/csi/fake"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

var (
	bFalse = false
	bTrue  = true
)

func makeTestAttachment(attachID, nodeName, pvName string) *storage.VolumeAttachment {
	return &storage.VolumeAttachment{
		ObjectMeta: meta.ObjectMeta{
			Name: attachID,
		},
		Spec: storage.VolumeAttachmentSpec{
			NodeName: nodeName,
			Attacher: "mock",
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
}

func markVolumeAttached(t *testing.T, client clientset.Interface, watch *watch.RaceFreeFakeWatcher, attachID string, status storage.VolumeAttachmentStatus) {
	ticker := time.NewTicker(10 * time.Millisecond)
	var attach *storage.VolumeAttachment
	var err error
	defer ticker.Stop()
	// wait for attachment to be saved
	for i := 0; i < 100; i++ {
		attach, err = client.StorageV1().VolumeAttachments().Get(context.TODO(), attachID, meta.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				<-ticker.C
				continue
			}
			t.Error(err)
		}
		if attach != nil {
			t.Logf("attachment found on try %d, stopping wait...", i)
			break
		}
	}
	t.Logf("stopped waiting for attachment")

	if attach == nil {
		t.Logf("attachment not found for id:%v", attachID)
	} else {
		attach.Status = status
		t.Logf("updating attachment %s with attach status %v", attachID, status)
		_, err := client.StorageV1().VolumeAttachments().Update(context.TODO(), attach, metav1.UpdateOptions{})
		if err != nil {
			t.Error(err)
		}
		if watch != nil {
			watch.Modify(attach)
		}
	}
}

func TestAttacherAttach(t *testing.T) {
	testCases := []struct {
		name                string
		nodeName            string
		driverName          string
		volumeName          string
		attachID            string
		spec                *volume.Spec
		injectAttacherError bool
		shouldFail          bool
	}{
		{
			name:       "test ok 1",
			nodeName:   "testnode-01",
			driverName: "testdriver-01",
			volumeName: "testvol-01",
			attachID:   getAttachmentName("testvol-01", "testdriver-01", "testnode-01"),
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "testdriver-01", "testvol-01"), false),
		},
		{
			name:       "test ok 2",
			nodeName:   "node02",
			driverName: "driver02",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver02", "vol02"), false),
		},
		{
			name:       "mismatch vol",
			nodeName:   "node02",
			driverName: "driver02",
			volumeName: "vol01",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver02", "vol01"), false),
			shouldFail: true,
		},
		{
			name:       "mismatch driver",
			nodeName:   "node02",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver01", "vol02"), false),
			shouldFail: true,
		},
		{
			name:       "mismatch node",
			nodeName:   "node000",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			spec:       volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver02", "vol02"), false),
			shouldFail: true,
		},
		{
			name:                "attacher error",
			nodeName:            "node02",
			driverName:          "driver02",
			volumeName:          "vol02",
			attachID:            getAttachmentName("vol02", "driver02", "node02"),
			spec:                volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver02", "vol02"), false),
			injectAttacherError: true,
			shouldFail:          true,
		},
		{
			name:       "test with volume source",
			nodeName:   "node000",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			spec:       volume.NewSpecFromVolume(makeTestVol("pv01", "driver02")),
			shouldFail: true, // csi not enabled
		},
		{
			name:       "missing spec",
			nodeName:   "node000",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			shouldFail: true, // csi not enabled
		},
	}

	// attacher loop
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("test case: %s", tc.name)
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}

			csiAttacher := attacher.(*csiAttacher)

			// FIXME: We need to ensure this goroutine exits in the test.
			go func(spec *volume.Spec, nodename string, fail bool) {
				attachID, err := csiAttacher.Attach(spec, types.NodeName(nodename))
				if !fail && err != nil {
					t.Errorf("expecting no failure, but got err: %v", err)
				}
				if fail && err == nil {
					t.Errorf("expecting failure, but got no err")
				}
				if attachID != "" {
					t.Errorf("expecting empty attachID, got %v", attachID)
				}
			}(tc.spec, tc.nodeName, tc.shouldFail)

			var status storage.VolumeAttachmentStatus
			if tc.injectAttacherError {
				status.Attached = false
				status.AttachError = &storage.VolumeError{
					Message: "attacher error",
				}
				errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error")).Status()
				fakeWatcher.Error(&errStatus)
			} else {
				status.Attached = true
			}
			markVolumeAttached(t, csiAttacher.k8s, fakeWatcher, tc.attachID, status)
		})
	}
}

func TestAttacherAttachWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	testCases := []struct {
		name                string
		nodeName            string
		driverName          string
		volumeName          string
		attachID            string
		spec                *volume.Spec
		injectAttacherError bool
		shouldFail          bool
	}{
		{
			name:     "test ok 1 with PV",
			nodeName: "node01",
			attachID: getAttachmentName("vol01", "driver01", "node01"),
			spec:     volume.NewSpecFromPersistentVolume(makeTestPV("pv01", 10, "driver01", "vol01"), false),
		},
		{
			name:       "test failure, attach with volSrc",
			nodeName:   "node01",
			attachID:   getAttachmentName("vol01", "driver01", "node01"),
			spec:       volume.NewSpecFromVolume(makeTestVol("vol01", "driver01")),
			shouldFail: true,
		},
		{
			name:                "attacher error",
			nodeName:            "node02",
			attachID:            getAttachmentName("vol02", "driver02", "node02"),
			spec:                volume.NewSpecFromPersistentVolume(makeTestPV("pv02", 10, "driver02", "vol02"), false),
			injectAttacherError: true,
			shouldFail:          true,
		},
		{
			name:       "missing spec",
			nodeName:   "node02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			shouldFail: true,
		},
	}

	// attacher loop
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("test case: %s", tc.name)
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)

			// FIXME: We need to ensure this goroutine exits in the test.
			go func(spec *volume.Spec, nodename string, fail bool) {
				attachID, err := csiAttacher.Attach(spec, types.NodeName(nodename))
				if fail != (err != nil) {
					t.Errorf("expecting no failure, but got err: %v", err)
				}
				if attachID != "" {
					t.Errorf("expecting empty attachID, got %v", attachID)
				}
			}(tc.spec, tc.nodeName, tc.shouldFail)

			var status storage.VolumeAttachmentStatus
			if tc.injectAttacherError {
				status.Attached = false
				status.AttachError = &storage.VolumeError{
					Message: "attacher error",
				}
			} else {
				status.Attached = true
			}
			markVolumeAttached(t, csiAttacher.k8s, fakeWatcher, tc.attachID, status)
		})
	}
}

func TestAttacherWithCSIDriver(t *testing.T) {
	tests := []struct {
		name                   string
		driver                 string
		expectVolumeAttachment bool
	}{
		{
			name:                   "CSIDriver not attachable",
			driver:                 "not-attachable",
			expectVolumeAttachment: false,
		},
		{
			name:                   "CSIDriver is attachable",
			driver:                 "attachable",
			expectVolumeAttachment: true,
		},
		{
			name:                   "CSIDriver.AttachRequired not set  -> failure",
			driver:                 "nil",
			expectVolumeAttachment: true,
		},
		{
			name:                   "CSIDriver does not exist not set  -> failure",
			driver:                 "unknown",
			expectVolumeAttachment: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := fakeclient.NewSimpleClientset(
				getTestCSIDriver("not-attachable", nil, &bFalse, nil),
				getTestCSIDriver("attachable", nil, &bTrue, nil),
				getTestCSIDriver("nil", nil, nil, nil),
			)
			plug, _, tmpDir, _ := newTestWatchPlugin(t, fakeClient, true)
			defer os.RemoveAll(tmpDir)

			attachmentWatchCreated := make(chan core.Action)
			// Make sure this is the first reactor
			fakeClient.Fake.PrependWatchReactor("volumeattachments", func(action core.Action) (bool, watch.Interface, error) {
				select {
				case <-attachmentWatchCreated:
					// already closed
				default:
					// The attacher is already watching the attachment, notify the test goroutine to
					// update the status of attachment.
					// TODO: In theory this still has a race condition, because the actual watch is created by
					// the next reactor in the chain and we unblock the test goroutine before returning here.
					close(attachmentWatchCreated)
				}
				return false, nil, nil
			})

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, test.driver, "test-vol"), false)

			pluginCanAttach, err := plug.CanAttach(spec)
			if err != nil {
				t.Fatalf("attacher.CanAttach failed: %s", err)
			}
			if pluginCanAttach != test.expectVolumeAttachment {
				t.Errorf("attacher.CanAttach does not match expected attachment status %t", test.expectVolumeAttachment)
			}

			if !pluginCanAttach {
				t.Log("plugin is not attachable")
				return
			}
			var wg sync.WaitGroup
			wg.Add(1)
			go func(volSpec *volume.Spec) {
				attachID, err := csiAttacher.Attach(volSpec, "fakeNode")
				defer wg.Done()

				if err != nil {
					t.Errorf("Attach() failed: %s", err)
				}
				if attachID != "" {
					t.Errorf("Expected empty attachID, got %q", attachID)
				}
			}(spec)

			if test.expectVolumeAttachment {
				expectedAttachID := getAttachmentName("test-vol", test.driver, "fakeNode")
				status := storage.VolumeAttachmentStatus{
					Attached: true,
				}
				// We want to ensure the watcher, which is created in csiAttacher,
				// has been started before updating the status of attachment.
				<-attachmentWatchCreated
				markVolumeAttached(t, csiAttacher.k8s, nil, expectedAttachID, status)
			}
			wg.Wait()
		})
	}
}

func TestAttacherWaitForVolumeAttachmentWithCSIDriver(t *testing.T) {
	// In order to detect if the volume plugin would skip WaitForAttach for non-attachable drivers,
	// we do not instantiate any VolumeAttachment. So if the plugin does not skip attach,  WaitForVolumeAttachment
	// will return an error that volume attachment was not found.
	tests := []struct {
		name        string
		driver      string
		expectError bool
	}{
		{
			name:        "CSIDriver not attachable -> success",
			driver:      "not-attachable",
			expectError: false,
		},
		{
			name:        "CSIDriver is attachable -> failure",
			driver:      "attachable",
			expectError: true,
		},
		{
			name:        "CSIDriver.AttachRequired not set  -> failure",
			driver:      "nil",
			expectError: true,
		},
		{
			name:        "CSIDriver does not exist not set  -> failure",
			driver:      "unknown",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeClient := fakeclient.NewSimpleClientset(
				getTestCSIDriver("not-attachable", nil, &bFalse, nil),
				getTestCSIDriver("attachable", nil, &bTrue, nil),
				getTestCSIDriver("nil", nil, nil, nil),
				&v1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakeNode",
					},
					Spec: v1.NodeSpec{},
				},
			)
			plug, tmpDir := newTestPlugin(t, fakeClient)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, test.driver, "test-vol"), false)

			pluginCanAttach, err := plug.CanAttach(spec)
			if err != nil {
				t.Fatalf("plugin.CanAttach test failed: %s", err)
			}
			if !pluginCanAttach {
				t.Log("plugin is not attachable")
				return
			}

			_, err = csiAttacher.WaitForAttach(spec, "", nil, time.Second)
			if err != nil && !test.expectError {
				t.Errorf("Unexpected error: %s", err)
			}
			if err == nil && test.expectError {
				t.Errorf("Expected error, got none")
			}
		})
	}
}

func TestAttacherWaitForAttach(t *testing.T) {
	tests := []struct {
		name             string
		driver           string
		makeAttachment   func() *storage.VolumeAttachment
		spec             *volume.Spec
		expectedAttachID string
		expectError      bool
	}{
		{
			name:   "successful attach",
			driver: "attachable",
			makeAttachment: func() *storage.VolumeAttachment {

				testAttachID := getAttachmentName("test-vol", "attachable", "fakeNode")
				successfulAttachment := makeTestAttachment(testAttachID, "fakeNode", "test-pv")
				successfulAttachment.Status.Attached = true
				return successfulAttachment
			},
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "attachable", "test-vol"), false),
			expectedAttachID: getAttachmentName("test-vol", "attachable", "fakeNode"),
			expectError:      false,
		},
		{
			name: "failed attach with vol source",
			makeAttachment: func() *storage.VolumeAttachment {

				testAttachID := getAttachmentName("test-vol", "attachable", "fakeNode")
				successfulAttachment := makeTestAttachment(testAttachID, "fakeNode", "volSrc01")
				successfulAttachment.Status.Attached = true
				return successfulAttachment
			},
			spec:        volume.NewSpecFromVolume(makeTestVol("volSrc01", "attachable")),
			expectError: true,
		},
		{
			name:        "failed attach",
			driver:      "attachable",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plug, _, tmpDir, _ := newTestWatchPlugin(t, nil, true)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)

			if test.makeAttachment != nil {
				attachment := test.makeAttachment()
				_, err = csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create VolumeAttachment: %v", err)
				}
				gotAttachment, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachment.Name, meta.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get created VolumeAttachment: %v", err)
				}
				t.Logf("created test VolumeAttachment %+v", gotAttachment)
			}

			attachID, err := csiAttacher.WaitForAttach(test.spec, "", nil, time.Second)
			if err != nil && !test.expectError {
				t.Errorf("Unexpected error: %s", err)
			}
			if err == nil && test.expectError {
				t.Errorf("Expected error, got none")
			}
			if attachID != test.expectedAttachID {
				t.Errorf("Expected attachID %q, got %q", test.expectedAttachID, attachID)
			}
		})
	}
}

func TestAttacherWaitForAttachWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()

	tests := []struct {
		name             string
		driver           string
		makeAttachment   func() *storage.VolumeAttachment
		spec             *volume.Spec
		expectedAttachID string
		expectError      bool
	}{
		{
			name: "successful attach with PV",
			makeAttachment: func() *storage.VolumeAttachment {

				testAttachID := getAttachmentName("test-vol", "attachable", "fakeNode")
				successfulAttachment := makeTestAttachment(testAttachID, "fakeNode", "test-pv")
				successfulAttachment.Status.Attached = true
				return successfulAttachment
			},
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "attachable", "test-vol"), false),
			expectedAttachID: getAttachmentName("test-vol", "attachable", "fakeNode"),
			expectError:      false,
		},
		{
			name: "failed attach with volSrc",
			makeAttachment: func() *storage.VolumeAttachment {

				testAttachID := getAttachmentName("test-vol", "attachable", "fakeNode")
				successfulAttachment := makeTestAttachment(testAttachID, "fakeNode", "volSrc01")
				successfulAttachment.Status.Attached = true
				return successfulAttachment
			},
			spec:        volume.NewSpecFromVolume(makeTestVol("volSrc01", "attachable")),
			expectError: true,
		},
		{
			name:        "failed attach",
			driver:      "non-attachable",
			spec:        volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "non-attachable", "test-vol"), false),
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plug, _, tmpDir, _ := newTestWatchPlugin(t, nil, true)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)

			if test.makeAttachment != nil {
				attachment := test.makeAttachment()
				_, err = csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create VolumeAttachment: %v", err)
				}
				gotAttachment, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), attachment.Name, meta.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get created VolumeAttachment: %v", err)
				}
				t.Logf("created test VolumeAttachment %+v", gotAttachment)
			}

			attachID, err := csiAttacher.WaitForAttach(test.spec, "", nil, time.Second)
			if test.expectError != (err != nil) {
				t.Errorf("Unexpected error: %s", err)
				return
			}
			if attachID != test.expectedAttachID {
				t.Errorf("Expected attachID %q, got %q", test.expectedAttachID, attachID)
			}
		})
	}
}

func TestAttacherWaitForVolumeAttachment(t *testing.T) {
	nodeName := "fakeNode"
	testCases := []struct {
		name                 string
		initAttached         bool
		finalAttached        bool
		trigerWatchEventTime time.Duration
		initAttachErr        *storage.VolumeError
		finalAttachErr       *storage.VolumeError
		timeout              time.Duration
		shouldFail           bool
	}{
		{
			name:         "attach success at get",
			initAttached: true,
			timeout:      50 * time.Millisecond,
			shouldFail:   false,
		},
		{
			name:          "attachment error ant get",
			initAttachErr: &storage.VolumeError{Message: "missing volume"},
			timeout:       30 * time.Millisecond,
			shouldFail:    true,
		},
		{
			name:                 "attach success at watch",
			initAttached:         false,
			finalAttached:        true,
			trigerWatchEventTime: 5 * time.Millisecond,
			timeout:              50 * time.Millisecond,
			shouldFail:           false,
		},
		{
			name:                 "attachment error ant watch",
			initAttached:         false,
			finalAttached:        false,
			finalAttachErr:       &storage.VolumeError{Message: "missing volume"},
			trigerWatchEventTime: 5 * time.Millisecond,
			timeout:              30 * time.Millisecond,
			shouldFail:           true,
		},
		{
			name:                 "time ran out",
			initAttached:         false,
			finalAttached:        true,
			trigerWatchEventTime: 100 * time.Millisecond,
			timeout:              50 * time.Millisecond,
			shouldFail:           true,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			t.Logf("running test: %v", tc.name)
			pvName := fmt.Sprintf("test-pv-%d", i)
			volID := fmt.Sprintf("test-vol-%d", i)
			attachID := getAttachmentName(volID, testDriver, nodeName)
			attachment := makeTestAttachment(attachID, nodeName, pvName)
			attachment.Status.Attached = tc.initAttached
			attachment.Status.AttachError = tc.initAttachErr
			_, err = csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}

			trigerWatchEventTime := tc.trigerWatchEventTime
			finalAttached := tc.finalAttached
			finalAttachErr := tc.finalAttachErr
			// after timeout, fakeWatcher will be closed by csiAttacher.waitForVolumeAttachment
			if tc.trigerWatchEventTime > 0 && tc.trigerWatchEventTime < tc.timeout {
				go func() {
					time.Sleep(trigerWatchEventTime)
					attachment := makeTestAttachment(attachID, nodeName, pvName)
					attachment.Status.Attached = finalAttached
					attachment.Status.AttachError = finalAttachErr
					fakeWatcher.Modify(attachment)
				}()
			}

			retID, err := csiAttacher.waitForVolumeAttachment(volID, attachID, tc.timeout)
			if tc.shouldFail && err == nil {
				t.Error("expecting failure, but err is nil")
			}
			if tc.initAttachErr != nil {
				if tc.initAttachErr.Message != err.Error() {
					t.Errorf("expecting error [%v], got [%v]", tc.initAttachErr.Message, err.Error())
				}
			}
			if err == nil && retID != attachID {
				t.Errorf("attacher.WaitForAttach not returning attachment ID")
			}
		})
	}
}

func TestAttacherVolumesAreAttached(t *testing.T) {
	type attachedSpec struct {
		volName  string
		spec     *volume.Spec
		attached bool
	}
	testCases := []struct {
		name          string
		attachedSpecs []attachedSpec
	}{
		{
			"attach and detach",
			[]attachedSpec{
				{"vol0", volume.NewSpecFromPersistentVolume(makeTestPV("pv0", 10, testDriver, "vol0"), false), true},
				{"vol1", volume.NewSpecFromPersistentVolume(makeTestPV("pv1", 20, testDriver, "vol1"), false), true},
				{"vol2", volume.NewSpecFromPersistentVolume(makeTestPV("pv2", 10, testDriver, "vol2"), false), false},
				{"vol3", volume.NewSpecFromPersistentVolume(makeTestPV("pv3", 10, testDriver, "vol3"), false), false},
				{"vol4", volume.NewSpecFromPersistentVolume(makeTestPV("pv4", 20, testDriver, "vol4"), false), true},
			},
		},
		{
			"all detached",
			[]attachedSpec{
				{"vol0", volume.NewSpecFromPersistentVolume(makeTestPV("pv0", 10, testDriver, "vol0"), false), false},
				{"vol1", volume.NewSpecFromPersistentVolume(makeTestPV("pv1", 20, testDriver, "vol1"), false), false},
				{"vol2", volume.NewSpecFromPersistentVolume(makeTestPV("pv2", 10, testDriver, "vol2"), false), false},
			},
		},
		{
			"all attached",
			[]attachedSpec{
				{"vol0", volume.NewSpecFromPersistentVolume(makeTestPV("pv0", 10, testDriver, "vol0"), false), true},
				{"vol1", volume.NewSpecFromPersistentVolume(makeTestPV("pv1", 20, testDriver, "vol1"), false), true},
			},
		},
		{
			"include non-attable",
			[]attachedSpec{
				{"vol0", volume.NewSpecFromPersistentVolume(makeTestPV("pv0", 10, testDriver, "vol0"), false), true},
				{"vol1", volume.NewSpecFromVolume(makeTestVol("pv1", testDriver)), false},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			nodeName := "fakeNode"

			var specs []*volume.Spec
			// create and save volume attchments
			for _, attachedSpec := range tc.attachedSpecs {
				specs = append(specs, attachedSpec.spec)
				attachID := getAttachmentName(attachedSpec.volName, testDriver, nodeName)
				attachment := makeTestAttachment(attachID, nodeName, attachedSpec.spec.Name())
				attachment.Status.Attached = attachedSpec.attached
				_, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to attach: %v", err)
				}
			}

			// retrieve attached status
			stats, err := csiAttacher.VolumesAreAttached(specs, types.NodeName(nodeName))
			if err != nil {
				t.Fatal(err)
			}
			if len(tc.attachedSpecs) != len(stats) {
				t.Errorf("expecting %d attachment status, got %d", len(tc.attachedSpecs), len(stats))
			}

			// compare attachment status for each spec
			for _, attached := range tc.attachedSpecs {
				stat, ok := stats[attached.spec]
				if attached.attached && !ok {
					t.Error("failed to retrieve attached status for:", attached.spec)
				}
				if attached.attached != stat {
					t.Errorf("expecting volume attachment %t, got %t", attached.attached, stat)
				}
			}
		})
	}
}

func TestAttacherVolumesAreAttachedWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	type attachedSpec struct {
		volName  string
		spec     *volume.Spec
		attached bool
	}
	testCases := []struct {
		name          string
		attachedSpecs []attachedSpec
	}{
		{
			"attach and detach with volume sources",
			[]attachedSpec{
				{"vol0", volume.NewSpecFromPersistentVolume(makeTestPV("pv0", 10, testDriver, "vol0"), false), true},
				{"vol1", volume.NewSpecFromVolume(makeTestVol("pv1", testDriver)), false},
				{"vol2", volume.NewSpecFromPersistentVolume(makeTestPV("pv2", 10, testDriver, "vol2"), false), true},
				{"vol3", volume.NewSpecFromVolume(makeTestVol("pv3", testDriver)), false},
				{"vol4", volume.NewSpecFromPersistentVolume(makeTestPV("pv4", 20, testDriver, "vol4"), false), true},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			nodeName := "fakeNode"

			var specs []*volume.Spec
			// create and save volume attchments
			for _, attachedSpec := range tc.attachedSpecs {
				specs = append(specs, attachedSpec.spec)
				attachID := getAttachmentName(attachedSpec.volName, testDriver, nodeName)
				attachment := makeTestAttachment(attachID, nodeName, attachedSpec.spec.Name())
				attachment.Status.Attached = attachedSpec.attached
				_, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to attach: %v", err)
				}
			}

			// retrieve attached status
			stats, err := csiAttacher.VolumesAreAttached(specs, types.NodeName(nodeName))
			if err != nil {
				t.Fatal(err)
			}
			if len(tc.attachedSpecs) != len(stats) {
				t.Errorf("expecting %d attachment status, got %d", len(tc.attachedSpecs), len(stats))
			}

			// compare attachment status for each spec
			for _, attached := range tc.attachedSpecs {
				stat, ok := stats[attached.spec]
				if attached.attached && !ok {
					t.Error("failed to retrieve attached status for:", attached.spec)
				}
				if attached.attached != stat {
					t.Errorf("expecting volume attachment %t, got %t", attached.attached, stat)
				}
			}
		})
	}
}

func TestAttacherDetach(t *testing.T) {
	nodeName := "fakeNode"
	testCases := []struct {
		name         string
		volID        string
		attachID     string
		shouldFail   bool
		watcherError bool
		reactor      func(action core.Action) (handled bool, ret runtime.Object, err error)
	}{
		{name: "normal test", volID: "vol-001", attachID: getAttachmentName("vol-001", testDriver, nodeName)},
		{name: "normal test 2", volID: "vol-002", attachID: getAttachmentName("vol-002", testDriver, nodeName)},
		{name: "object not found", volID: "vol-non-existing", attachID: getAttachmentName("vol-003", testDriver, nodeName)},
		{
			name:       "API error",
			volID:      "vol-004",
			attachID:   getAttachmentName("vol-004", testDriver, nodeName),
			shouldFail: true, // All other API errors should be propagated to caller
			reactor: func(action core.Action) (handled bool, ret runtime.Object, err error) {
				// return Forbidden to all DELETE requests
				if action.Matches("delete", "volumeattachments") {
					return true, nil, apierrors.NewForbidden(action.GetResource().GroupResource(), action.GetNamespace(), fmt.Errorf("mock error"))
				}
				return false, nil, nil
			},
		},
		{
			name:         "API watch error happen",
			volID:        "vol-005",
			attachID:     getAttachmentName("vol-005", testDriver, nodeName),
			shouldFail:   true,
			watcherError: true,
			reactor: func(action core.Action) (handled bool, ret runtime.Object, err error) {
				if action.Matches("get", "volumeattachments") {
					return true, makeTestAttachment(getAttachmentName("vol-005", testDriver, nodeName), nodeName, "vol-005"), nil
				}
				return false, nil, nil
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("running test: %v", tc.name)
			plug, fakeWatcher, tmpDir, client := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)
			if tc.reactor != nil {
				client.PrependReactor("*", "*", tc.reactor)
			}

			attacher, err0 := plug.NewAttacher()
			if err0 != nil {
				t.Fatalf("failed to create new attacher: %v", err0)
			}
			csiAttacher := attacher.(*csiAttacher)

			pv := makeTestPV("test-pv", 10, testDriver, tc.volID)
			spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
			attachment := makeTestAttachment(tc.attachID, nodeName, "test-pv")
			_, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}
			volumeName, err := plug.GetVolumeName(spec)
			if err != nil {
				t.Errorf("test case %s failed: %v", tc.name, err)
			}
			watchError := tc.watcherError
			csiAttacher.waitSleepTime = 100 * time.Millisecond
			go func() {
				if watchError {
					errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error")).Status()
					fakeWatcher.Error(&errStatus)
					return
				}
				fakeWatcher.Delete(attachment)
			}()
			err = csiAttacher.Detach(volumeName, types.NodeName(nodeName))
			if tc.shouldFail && err == nil {
				t.Fatal("expecting failure, but err = nil")
			}
			if !tc.shouldFail && err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			attach, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Get(context.TODO(), tc.attachID, meta.GetOptions{})
			if err != nil {
				if !apierrors.IsNotFound(err) {
					t.Fatalf("unexpected err: %v", err)
				}
			} else {
				if attach == nil {
					t.Errorf("expecting attachment not to be nil, but it is")
				}
			}
		})
	}
}

func TestAttacherGetDeviceMountPath(t *testing.T) {
	// Setup
	// Create a new attacher
	plug, _, tmpDir, _ := newTestWatchPlugin(t, nil, true)
	defer os.RemoveAll(tmpDir)
	attacher, err0 := plug.NewAttacher()
	if err0 != nil {
		t.Fatalf("failed to create new attacher: %v", err0)
	}
	csiAttacher := attacher.(*csiAttacher)

	pluginDir := csiAttacher.plugin.host.GetPluginDir(plug.GetPluginName())

	testCases := []struct {
		testName          string
		pvName            string
		expectedMountPath string
		shouldFail        bool
	}{
		{
			testName:          "normal test",
			pvName:            "test-pv1",
			expectedMountPath: pluginDir + "/pv/test-pv1/globalmount",
		},
		{
			testName:          "no pv name",
			pvName:            "",
			expectedMountPath: pluginDir + "/pv/test-pv1/globalmount",
			shouldFail:        true,
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.testName)
		var spec *volume.Spec

		// Create spec
		pv := makeTestPV(tc.pvName, 10, testDriver, "testvol")
		spec = volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

		// Run
		mountPath, err := csiAttacher.GetDeviceMountPath(spec)

		// Verify
		if err != nil && !tc.shouldFail {
			t.Errorf("test should not fail, but error occurred: %v", err)
		} else if err == nil {
			if tc.shouldFail {
				t.Errorf("test should fail, but no error occurred")
			} else if mountPath != tc.expectedMountPath {
				t.Errorf("mountPath does not equal expectedMountPath. Got: %s. Expected: %s", mountPath, tc.expectedMountPath)
			}
		}
	}
}

func TestAttacherMountDevice(t *testing.T) {
	pvName := "test-pv"
	nonFinalError := volumetypes.NewUncertainProgressError("")
	transientError := volumetypes.NewTransientOperationFailure("")

	testCases := []struct {
		testName         string
		volName          string
		devicePath       string
		deviceMountPath  string
		stageUnstageSet  bool
		shouldFail       bool
		createAttachment bool
		exitError        error
		spec             *volume.Spec
	}{
		{
			testName:         "normal PV",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:         "normal PV with mount options",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPVWithMountOptions(pvName, 10, testDriver, "test-vol1", []string{"test-op"}), false),
		},
		{
			testName:         "normal PV but with missing attachment should result in no-change",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			createAttachment: false,
			shouldFail:       true,
			exitError:        transientError,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPVWithMountOptions(pvName, 10, testDriver, "test-vol1", []string{"test-op"}), false),
		},
		{
			testName:         "no vol name",
			volName:          "",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			shouldFail:       true,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, ""), false),
		},
		{
			testName:         "no device path",
			volName:          "test-vol1",
			devicePath:       "",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			shouldFail:       false,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:         "no device mount path",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "",
			stageUnstageSet:  true,
			shouldFail:       true,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:         "stage_unstage cap not set",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  false,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:         "failure with volume source",
			volName:          "test-vol1",
			devicePath:       "path1",
			deviceMountPath:  "path2",
			shouldFail:       true,
			createAttachment: true,
			spec:             volume.NewSpecFromVolume(makeTestVol(pvName, testDriver)),
		},
		{
			testName:         "pv with nodestage timeout should result in in-progress device",
			volName:          fakecsi.NodeStageTimeOut_VolumeID,
			devicePath:       "path1",
			deviceMountPath:  "path2",
			stageUnstageSet:  true,
			createAttachment: true,
			spec:             volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, fakecsi.NodeStageTimeOut_VolumeID), false),
			exitError:        nonFinalError,
			shouldFail:       true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			t.Logf("Running test case: %s", tc.testName)

			// Setup
			// Create a new attacher
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)
			attacher, err0 := plug.NewAttacher()
			if err0 != nil {
				t.Fatalf("failed to create new attacher: %v", err0)
			}
			csiAttacher := attacher.(*csiAttacher)
			csiAttacher.csiClient = setupClient(t, tc.stageUnstageSet)

			if tc.deviceMountPath != "" {
				tc.deviceMountPath = filepath.Join(tmpDir, tc.deviceMountPath)
			}

			nodeName := string(csiAttacher.plugin.host.GetNodeName())
			attachID := getAttachmentName(tc.volName, testDriver, nodeName)

			if tc.createAttachment {
				// Set up volume attachment
				attachment := makeTestAttachment(attachID, nodeName, pvName)
				_, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to attach: %v", err)
				}
				go func() {
					fakeWatcher.Delete(attachment)
				}()
			}

			// Run
			err := csiAttacher.MountDevice(tc.spec, tc.devicePath, tc.deviceMountPath)

			// Verify
			if err != nil {
				if !tc.shouldFail {
					t.Errorf("test should not fail, but error occurred: %v", err)
				}
				return
			}
			if err == nil && tc.shouldFail {
				t.Errorf("test should fail, but no error occurred")
			}

			if tc.exitError != nil && reflect.TypeOf(tc.exitError) != reflect.TypeOf(err) {
				t.Fatalf("expected exitError: %v got: %v", tc.exitError, err)
			}

			// Verify call goes through all the way
			numStaged := 1
			if !tc.stageUnstageSet {
				numStaged = 0
			}

			cdc := csiAttacher.csiClient.(*fakeCsiDriverClient)
			staged := cdc.nodeClient.GetNodeStagedVolumes()
			if len(staged) != numStaged {
				t.Errorf("got wrong number of staged volumes, expecting %v got: %v", numStaged, len(staged))
			}
			if tc.stageUnstageSet {
				vol, ok := staged[tc.volName]
				if !ok {
					t.Errorf("could not find staged volume: %s", tc.volName)
				}
				if vol.Path != tc.deviceMountPath {
					t.Errorf("expected mount path: %s. got: %s", tc.deviceMountPath, vol.Path)
				}
				if !reflect.DeepEqual(vol.MountFlags, tc.spec.PersistentVolume.Spec.MountOptions) {
					t.Errorf("expected mount options: %v, got: %v", tc.spec.PersistentVolume.Spec.MountOptions, vol.MountFlags)
				}
			}
		})
	}
}

func TestAttacherMountDeviceWithInline(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIInlineVolume, true)()
	pvName := "test-pv"
	testCases := []struct {
		testName        string
		volName         string
		devicePath      string
		deviceMountPath string
		stageUnstageSet bool
		shouldFail      bool
		spec            *volume.Spec
	}{
		{
			testName:        "normal PV",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: true,
			spec:            volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:        "failure with volSrc",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			shouldFail:      true,
			spec:            volume.NewSpecFromVolume(makeTestVol(pvName, testDriver)),
		},
		{
			testName:        "no vol name",
			volName:         "",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: true,
			shouldFail:      true,
			spec:            volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, ""), false),
		},
		{
			testName:        "no device path",
			volName:         "test-vol1",
			devicePath:      "",
			deviceMountPath: "path2",
			stageUnstageSet: true,
			shouldFail:      false,
			spec:            volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:        "no device mount path",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "",
			stageUnstageSet: true,
			shouldFail:      true,
			spec:            volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:        "stage_unstage cap not set",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: false,
			spec:            volume.NewSpecFromPersistentVolume(makeTestPV(pvName, 10, testDriver, "test-vol1"), false),
		},
		{
			testName:        "missing spec",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			shouldFail:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			t.Logf("Running test case: %s", tc.testName)

			// Setup
			// Create a new attacher
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil, false)
			defer os.RemoveAll(tmpDir)
			attacher, err0 := plug.NewAttacher()
			if err0 != nil {
				t.Fatalf("failed to create new attacher: %v", err0)
			}
			csiAttacher := attacher.(*csiAttacher)
			csiAttacher.csiClient = setupClient(t, tc.stageUnstageSet)

			if tc.deviceMountPath != "" {
				tc.deviceMountPath = filepath.Join(tmpDir, tc.deviceMountPath)
			}

			nodeName := string(csiAttacher.plugin.host.GetNodeName())
			attachID := getAttachmentName(tc.volName, testDriver, nodeName)

			// Set up volume attachment
			attachment := makeTestAttachment(attachID, nodeName, pvName)
			_, err := csiAttacher.k8s.StorageV1().VolumeAttachments().Create(context.TODO(), attachment, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}
			go func() {
				fakeWatcher.Delete(attachment)
			}()

			// Run
			err = csiAttacher.MountDevice(tc.spec, tc.devicePath, tc.deviceMountPath)

			// Verify
			if err != nil {
				if !tc.shouldFail {
					t.Errorf("test should not fail, but error occurred: %v", err)
				}
				return
			}
			if err == nil && tc.shouldFail {
				t.Errorf("test should fail, but no error occurred")
			}

			// Verify call goes through all the way
			numStaged := 1
			if !tc.stageUnstageSet {
				numStaged = 0
			}

			cdc := csiAttacher.csiClient.(*fakeCsiDriverClient)
			staged := cdc.nodeClient.GetNodeStagedVolumes()
			if len(staged) != numStaged {
				t.Errorf("got wrong number of staged volumes, expecting %v got: %v", numStaged, len(staged))
			}
			if tc.stageUnstageSet {
				vol, ok := staged[tc.volName]
				if !ok {
					t.Errorf("could not find staged volume: %s", tc.volName)
				}
				if vol.Path != tc.deviceMountPath {
					t.Errorf("expected mount path: %s. got: %s", tc.deviceMountPath, vol.Path)
				}
			}
		})
	}
}

func TestAttacherUnmountDevice(t *testing.T) {
	testCases := []struct {
		testName        string
		volID           string
		deviceMountPath string
		jsonFile        string
		createPV        bool
		stageUnstageSet bool
		shouldFail      bool
	}{
		{
			testName:        "normal, json file exists",
			volID:           "project/zone/test-vol1",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        `{"driverName": "csi", "volumeHandle":"project/zone/test-vol1"}`,
			createPV:        false,
			stageUnstageSet: true,
		},
		{
			testName:        "normal, json file doesn't exist -> use PV",
			volID:           "project/zone/test-vol1",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        "",
			createPV:        true,
			stageUnstageSet: true,
		},
		{
			testName:        "invalid json ->  use PV",
			volID:           "project/zone/test-vol1",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        `{"driverName"}}`,
			createPV:        true,
			stageUnstageSet: true,
		},
		{
			testName:        "no json, no PV.volID",
			volID:           "",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        "",
			createPV:        true,
			shouldFail:      true,
		},
		{
			testName:        "no json, no PV",
			volID:           "project/zone/test-vol1",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        "",
			createPV:        false,
			stageUnstageSet: true,
			shouldFail:      true,
		},
		{
			testName:        "stage_unstage not set no vars should not fail",
			deviceMountPath: "plugins/csi/pv/test-pv-name/globalmount",
			jsonFile:        `{"driverName":"test-driver","volumeHandle":"test-vol1"}`,
			stageUnstageSet: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			t.Logf("Running test case: %s", tc.testName)
			// Setup
			// Create a new attacher
			plug, _, tmpDir, _ := newTestWatchPlugin(t, nil, true)
			defer os.RemoveAll(tmpDir)
			attacher, err0 := plug.NewAttacher()
			if err0 != nil {
				t.Fatalf("failed to create new attacher: %v", err0)
			}
			csiAttacher := attacher.(*csiAttacher)
			csiAttacher.csiClient = setupClient(t, tc.stageUnstageSet)

			if tc.deviceMountPath != "" {
				tc.deviceMountPath = filepath.Join(tmpDir, tc.deviceMountPath)
			}

			// Add the volume to NodeStagedVolumes
			cdc := csiAttacher.csiClient.(*fakeCsiDriverClient)
			cdc.nodeClient.AddNodeStagedVolume(tc.volID, tc.deviceMountPath, nil)

			// Make JSON for this object
			if tc.deviceMountPath != "" {
				if err := os.MkdirAll(tc.deviceMountPath, 0755); err != nil {
					t.Fatalf("error creating directory %s: %s", tc.deviceMountPath, err)
				}
			}
			dir := filepath.Dir(tc.deviceMountPath)
			if tc.jsonFile != "" {
				dataPath := filepath.Join(dir, volDataFileName)
				if err := ioutil.WriteFile(dataPath, []byte(tc.jsonFile), 0644); err != nil {
					t.Fatalf("error creating %s: %s", dataPath, err)
				}
			}
			if tc.createPV {
				// Make the PV for this object
				pvName := filepath.Base(dir)
				pv := makeTestPV(pvName, 5, "csi", tc.volID)
				_, err := csiAttacher.k8s.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
				if err != nil && !tc.shouldFail {
					t.Fatalf("Failed to create PV: %v", err)
				}
			}

			// Run
			err := csiAttacher.UnmountDevice(tc.deviceMountPath)
			// Verify
			if err != nil {
				if !tc.shouldFail {
					t.Errorf("test should not fail, but error occurred: %v", err)
				}
				return
			}
			if err == nil && tc.shouldFail {
				t.Errorf("test should fail, but no error occurred")
			}

			// Verify call goes through all the way
			expectedSet := 0
			if !tc.stageUnstageSet {
				expectedSet = 1
			}
			staged := cdc.nodeClient.GetNodeStagedVolumes()
			if len(staged) != expectedSet {
				t.Errorf("got wrong number of staged volumes, expecting %v got: %v", expectedSet, len(staged))
			}

			_, ok := staged[tc.volID]
			if ok && tc.stageUnstageSet {
				t.Errorf("found unexpected staged volume: %s", tc.volID)
			} else if !ok && !tc.stageUnstageSet {
				t.Errorf("could not find expected staged volume: %s", tc.volID)
			}

			if tc.jsonFile != "" && !tc.shouldFail {
				dataPath := filepath.Join(dir, volDataFileName)
				if _, err := os.Stat(dataPath); !os.IsNotExist(err) {
					if err != nil {
						t.Errorf("error checking file %s: %s", dataPath, err)
					} else {
						t.Errorf("json file %s should not exists, but it does", dataPath)
					}
				} else {
					t.Logf("json file %s was correctly removed", dataPath)
				}
			}
		})
	}
}

// create a plugin mgr to load plugins and setup a fake client
func newTestWatchPlugin(t *testing.T, fakeClient *fakeclient.Clientset, setupInformer bool) (*csiPlugin, *watch.RaceFreeFakeWatcher, string, *fakeclient.Clientset) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	if fakeClient == nil {
		fakeClient = fakeclient.NewSimpleClientset()
	}
	fakeClient.Tracker().Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "fakeNode",
		},
		Spec: v1.NodeSpec{},
	})
	fakeWatcher := watch.NewRaceFreeFake()
	if !setupInformer {
		// TODO: In the fakeClient, if default watchReactor is overwritten, the volumeAttachmentInformer
		// and the csiAttacher.Attach both endup reading from same channel causing hang in Attach().
		// So, until this is fixed, we don't overwrite default reactor while setting up volumeAttachment informer.
		fakeClient.Fake.PrependWatchReactor("volumeattachments", core.DefaultWatchReactor(fakeWatcher, nil))
	}

	// Start informer for CSIDrivers.
	factory := informers.NewSharedInformerFactory(fakeClient, CsiResyncPeriod)
	csiDriverInformer := factory.Storage().V1().CSIDrivers()
	csiDriverLister := csiDriverInformer.Lister()
	var volumeAttachmentInformer storageinformer.VolumeAttachmentInformer
	var volumeAttachmentLister storagelister.VolumeAttachmentLister
	if setupInformer {
		volumeAttachmentInformer = factory.Storage().V1().VolumeAttachments()
		volumeAttachmentLister = volumeAttachmentInformer.Lister()
	}

	factory.Start(wait.NeverStop)
	ctx, cancel := context.WithTimeout(context.Background(), TestInformerSyncTimeout)
	defer cancel()
	for ty, ok := range factory.WaitForCacheSync(ctx.Done()) {
		if !ok {
			t.Fatalf("failed to sync: %#v", ty)
		}
	}

	host := volumetest.NewFakeVolumeHostWithCSINodeName(t,
		tmpDir,
		fakeClient,
		ProbeVolumePlugins(),
		"fakeNode",
		csiDriverLister,
		volumeAttachmentLister,
	)
	plugMgr := host.GetPluginMgr()

	plug, err := plugMgr.FindPluginByName(CSIPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", CSIPluginName)
	}

	csiPlug, ok := plug.(*csiPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type csiPlugin")
	}

	return csiPlug, fakeWatcher, tmpDir, fakeClient
}
