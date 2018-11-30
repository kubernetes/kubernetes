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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	clientset "k8s.io/client-go/kubernetes"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	utiltesting "k8s.io/client-go/util/testing"
	fakecsi "k8s.io/csi-api/pkg/client/clientset/versioned/fake"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
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
		attach, err = client.StorageV1beta1().VolumeAttachments().Get(attachID, meta.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				<-ticker.C
				continue
			}
			t.Error(err)
		}
		if attach != nil {
			klog.Infof("stopping wait")
			break
		}
	}
	klog.Infof("stopped wait")

	if attach == nil {
		t.Logf("attachment not found for id:%v", attachID)
	} else {
		attach.Status = status
		_, err := client.StorageV1beta1().VolumeAttachments().Update(attach)
		if err != nil {
			t.Error(err)
		}
		watch.Modify(attach)
	}
}

func TestAttacherAttach(t *testing.T) {

	testCases := []struct {
		name                string
		nodeName            string
		driverName          string
		volumeName          string
		attachID            string
		injectAttacherError bool
		shouldFail          bool
	}{
		{
			name:       "test ok 1",
			nodeName:   "testnode-01",
			driverName: "testdriver-01",
			volumeName: "testvol-01",
			attachID:   getAttachmentName("testvol-01", "testdriver-01", "testnode-01"),
		},
		{
			name:       "test ok 2",
			nodeName:   "node02",
			driverName: "driver02",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
		},
		{
			name:       "mismatch vol",
			nodeName:   "node02",
			driverName: "driver02",
			volumeName: "vol01",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			shouldFail: true,
		},
		{
			name:       "mismatch driver",
			nodeName:   "node02",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			shouldFail: true,
		},
		{
			name:       "mismatch node",
			nodeName:   "node000",
			driverName: "driver000",
			volumeName: "vol02",
			attachID:   getAttachmentName("vol02", "driver02", "node02"),
			shouldFail: true,
		},
		{
			name:                "attacher error",
			nodeName:            "node02",
			driverName:          "driver02",
			volumeName:          "vol02",
			attachID:            getAttachmentName("vol02", "driver02", "node02"),
			injectAttacherError: true,
			shouldFail:          true,
		},
	}

	// attacher loop
	for i, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil)
		defer os.RemoveAll(tmpDir)

		attacher, err := plug.NewAttacher()
		if err != nil {
			t.Fatalf("failed to create new attacher: %v", err)
		}

		csiAttacher := attacher.(*csiAttacher)

		spec := volume.NewSpecFromPersistentVolume(makeTestPV(fmt.Sprintf("test-pv%d", i), 10, tc.driverName, tc.volumeName), false)

		go func(id, nodename string, fail bool) {
			attachID, err := csiAttacher.Attach(spec, types.NodeName(nodename))
			if !fail && err != nil {
				t.Errorf("expecting no failure, but got err: %v", err)
			}
			if fail && err == nil {
				t.Errorf("expecting failure, but got no err")
			}
			if attachID != id && !fail {
				t.Errorf("expecting attachID %v, got %v", id, attachID)
			}
		}(tc.attachID, tc.nodeName, tc.shouldFail)

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
	}
}

func TestAttacherWithCSIDriver(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIDriverRegistry, true)()

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
			fakeCSIClient := fakecsi.NewSimpleClientset(
				getCSIDriver("not-attachable", nil, &bFalse),
				getCSIDriver("attachable", nil, &bTrue),
				getCSIDriver("nil", nil, nil),
			)
			plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, fakeCSIClient)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, test.driver, "test-vol"), false)

			expectedAttachID := getAttachmentName("test-vol", test.driver, "node")
			status := storage.VolumeAttachmentStatus{
				Attached: true,
			}
			if test.expectVolumeAttachment {
				go markVolumeAttached(t, csiAttacher.k8s, fakeWatcher, expectedAttachID, status)
			}
			attachID, err := csiAttacher.Attach(spec, types.NodeName("node"))
			if err != nil {
				t.Errorf("Attach() failed: %s", err)
			}
			if test.expectVolumeAttachment && attachID == "" {
				t.Errorf("Epected attachID, got nothing")
			}
			if !test.expectVolumeAttachment && attachID != "" {
				t.Errorf("Epected empty attachID, got %q", attachID)
			}
		})
	}
}

func TestAttacherWaitForVolumeAttachmentWithCSIDriver(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIDriverRegistry, true)()

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
			fakeCSIClient := fakecsi.NewSimpleClientset(
				getCSIDriver("not-attachable", nil, &bFalse),
				getCSIDriver("attachable", nil, &bTrue),
				getCSIDriver("nil", nil, nil),
			)
			plug, tmpDir := newTestPlugin(t, nil, fakeCSIClient)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, test.driver, "test-vol"), false)
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
		expectedAttachID string
		expectError      bool
	}{
		{
			name:   "successful attach",
			driver: "attachable",
			makeAttachment: func() *storage.VolumeAttachment {

				testAttachID := getAttachmentName("test-vol", "attachable", "node")
				successfulAttachment := makeTestAttachment(testAttachID, "node", "test-pv")
				successfulAttachment.Status.Attached = true
				return successfulAttachment
			},
			expectedAttachID: getAttachmentName("test-vol", "attachable", "node"),
			expectError:      false,
		},
		{
			name:        "failed attach",
			driver:      "attachable",
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plug, _, tmpDir, _ := newTestWatchPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			attacher, err := plug.NewAttacher()
			if err != nil {
				t.Fatalf("failed to create new attacher: %v", err)
			}
			csiAttacher := attacher.(*csiAttacher)
			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, test.driver, "test-vol"), false)

			if test.makeAttachment != nil {
				attachment := test.makeAttachment()
				_, err = csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
				if err != nil {
					t.Fatalf("failed to create VolumeAttachment: %v", err)
				}
				gotAttachment, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Get(attachment.Name, meta.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get created VolumeAttachment: %v", err)
				}
				t.Logf("created test VolumeAttachment %+v", gotAttachment)
			}

			attachID, err := csiAttacher.WaitForAttach(spec, "", nil, time.Second)
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

func TestAttacherWaitForVolumeAttachment(t *testing.T) {
	nodeName := "test-node"
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
		plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil)
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
		_, err = csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
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
	}
}

func TestAttacherVolumesAreAttached(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil, nil)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}
	csiAttacher := attacher.(*csiAttacher)
	nodeName := "test-node"

	testCases := []struct {
		name          string
		attachedStats map[string]bool
	}{
		{"attach + detach", map[string]bool{"vol-01": true, "vol-02": true, "vol-03": false, "vol-04": false, "vol-05": true}},
		{"all detached", map[string]bool{"vol-11": false, "vol-12": false, "vol-13": false, "vol-14": false, "vol-15": false}},
		{"all attached", map[string]bool{"vol-21": true, "vol-22": true, "vol-23": true, "vol-24": true, "vol-25": true}},
	}

	for _, tc := range testCases {
		var specs []*volume.Spec
		// create and save volume attchments
		for volName, stat := range tc.attachedStats {
			pv := makeTestPV("test-pv", 10, testDriver, volName)
			spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
			specs = append(specs, spec)
			attachID := getAttachmentName(volName, testDriver, nodeName)
			attachment := makeTestAttachment(attachID, nodeName, pv.GetName())
			attachment.Status.Attached = stat
			_, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}
		}

		// retrieve attached status
		stats, err := csiAttacher.VolumesAreAttached(specs, types.NodeName(nodeName))
		if err != nil {
			t.Fatal(err)
		}
		if len(tc.attachedStats) != len(stats) {
			t.Errorf("expecting %d attachment status, got %d", len(tc.attachedStats), len(stats))
		}

		// compare attachment status for each spec
		for spec, stat := range stats {
			source, err := getCSISourceFromSpec(spec)
			if err != nil {
				t.Error(err)
			}
			if stat != tc.attachedStats[source.VolumeHandle] {
				t.Errorf("expecting volume attachment %t, got %t", tc.attachedStats[source.VolumeHandle], stat)
			}
		}
	}
}

func TestAttacherDetach(t *testing.T) {

	nodeName := "test-node"
	testCases := []struct {
		name       string
		volID      string
		attachID   string
		shouldFail bool
		reactor    func(action core.Action) (handled bool, ret runtime.Object, err error)
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
					return true, nil, apierrs.NewForbidden(action.GetResource().GroupResource(), action.GetNamespace(), fmt.Errorf("mock error"))
				}
				return false, nil, nil
			},
		},
	}

	for _, tc := range testCases {
		t.Logf("running test: %v", tc.name)
		plug, fakeWatcher, tmpDir, client := newTestWatchPlugin(t, nil)
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
		_, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
		if err != nil {
			t.Fatalf("failed to attach: %v", err)
		}
		volumeName, err := plug.GetVolumeName(spec)
		if err != nil {
			t.Errorf("test case %s failed: %v", tc.name, err)
		}
		go func() {
			fakeWatcher.Delete(attachment)
		}()
		err = csiAttacher.Detach(volumeName, types.NodeName(nodeName))
		if tc.shouldFail && err == nil {
			t.Fatal("expecting failure, but err = nil")
		}
		if !tc.shouldFail && err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		attach, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Get(tc.attachID, meta.GetOptions{})
		if err != nil {
			if !apierrs.IsNotFound(err) {
				t.Fatalf("unexpected err: %v", err)
			}
		} else {
			if attach == nil {
				t.Errorf("expecting attachment not to be nil, but it is")
			}
		}
	}
}

func TestAttacherGetDeviceMountPath(t *testing.T) {
	// Setup
	// Create a new attacher
	plug, _, tmpDir, _ := newTestWatchPlugin(t, nil)
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
	testCases := []struct {
		testName        string
		volName         string
		devicePath      string
		deviceMountPath string
		stageUnstageSet bool
		shouldFail      bool
	}{
		{
			testName:        "normal",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: true,
		},
		{
			testName:        "no vol name",
			volName:         "",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: true,
			shouldFail:      true,
		},
		{
			testName:        "no device path",
			volName:         "test-vol1",
			devicePath:      "",
			deviceMountPath: "path2",
			stageUnstageSet: true,
			shouldFail:      false,
		},
		{
			testName:        "no device mount path",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "",
			stageUnstageSet: true,
			shouldFail:      true,
		},
		{
			testName:        "stage_unstage cap not set",
			volName:         "test-vol1",
			devicePath:      "path1",
			deviceMountPath: "path2",
			stageUnstageSet: false,
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.testName)
		var spec *volume.Spec
		pvName := "test-pv"

		// Setup
		// Create a new attacher
		plug, fakeWatcher, tmpDir, _ := newTestWatchPlugin(t, nil)
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

		// Create spec
		pv := makeTestPV(pvName, 10, testDriver, tc.volName)
		spec = volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

		attachID := getAttachmentName(tc.volName, testDriver, nodeName)

		// Set up volume attachment
		attachment := makeTestAttachment(attachID, nodeName, pvName)
		_, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
		if err != nil {
			t.Fatalf("failed to attach: %v", err)
		}
		go func() {
			fakeWatcher.Delete(attachment)
		}()

		// Run
		err = csiAttacher.MountDevice(spec, tc.devicePath, tc.deviceMountPath)

		// Verify
		if err != nil {
			if !tc.shouldFail {
				t.Errorf("test should not fail, but error occurred: %v", err)
			}
			continue
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
		t.Logf("Running test case: %s", tc.testName)
		// Setup
		// Create a new attacher
		plug, _, tmpDir, _ := newTestWatchPlugin(t, nil)
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
			_, err := csiAttacher.k8s.CoreV1().PersistentVolumes().Create(pv)
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
			continue
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
	}
}

// create a plugin mgr to load plugins and setup a fake client
func newTestWatchPlugin(t *testing.T, csiClient *fakecsi.Clientset) (*csiPlugin, *watch.RaceFreeFakeWatcher, string, *fakeclient.Clientset) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}

	fakeClient := fakeclient.NewSimpleClientset()
	fakeWatcher := watch.NewRaceFreeFake()
	fakeClient.Fake.PrependWatchReactor("*", core.DefaultWatchReactor(fakeWatcher, nil))
	fakeClient.Fake.WatchReactionChain = fakeClient.Fake.WatchReactionChain[:1]
	if csiClient == nil {
		csiClient = fakecsi.NewSimpleClientset()
	}
	host := volumetest.NewFakeVolumeHostWithCSINodeName(
		tmpDir,
		fakeClient,
		csiClient,
		nil,
		"node",
	)
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, host)

	plug, err := plugMgr.FindPluginByName(csiPluginName)
	if err != nil {
		t.Fatalf("can't find plugin %v", csiPluginName)
	}

	csiPlug, ok := plug.(*csiPlugin)
	if !ok {
		t.Fatalf("cannot assert plugin to be type csiPlugin")
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		// Wait until the informer in CSI volume plugin has all CSIDrivers.
		wait.PollImmediate(testInformerSyncPeriod, testInformerSyncTimeout, func() (bool, error) {
			return csiPlug.csiDriverInformer.Informer().HasSynced(), nil
		})
	}

	return csiPlug, fakeWatcher, tmpDir, fakeClient
}
