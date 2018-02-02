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
	"testing"
	"time"

	storage "k8s.io/api/storage/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
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

func TestAttacherAttach(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}

	csiAttacher := attacher.(*csiAttacher)

	testCases := []struct {
		name       string
		nodeName   string
		driverName string
		volumeName string
		attachID   string
		shouldFail bool
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
	}

	// attacher loop
	for i, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		spec := volume.NewSpecFromPersistentVolume(makeTestPV(fmt.Sprintf("test-pv%d", i), 10, tc.driverName, tc.volumeName), false)

		go func(id, nodename string, fail bool) {
			attachID, err := csiAttacher.Attach(spec, types.NodeName(nodename))
			if !fail && err != nil {
				t.Errorf("expecting no failure, but got err: %v", err)
			}
			if attachID != id && !fail {
				t.Errorf("expecting attachID %v, got %v", id, attachID)
			}
		}(tc.attachID, tc.nodeName, tc.shouldFail)

		// update attachment to avoid long waitForAttachment
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		// wait for attachment to be saved
		var attach *storage.VolumeAttachment
		for i := 0; i < 100; i++ {
			attach, err = csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Get(tc.attachID, meta.GetOptions{})
			if err != nil {
				if apierrs.IsNotFound(err) {
					<-ticker.C
					continue
				}
				t.Error(err)
			}
			if attach != nil {
				break
			}
		}

		if attach == nil {
			t.Error("attachment not found")
		}
		attach.Status.Attached = true
		_, err = csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Update(attach)
		if err != nil {
			t.Error(err)
		}
	}
}

func TestAttacherWaitForVolumeAttachment(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}
	csiAttacher := attacher.(*csiAttacher)
	nodeName := "test-node"

	testCases := []struct {
		name       string
		attached   bool
		attachErr  *storage.VolumeError
		sleepTime  time.Duration
		timeout    time.Duration
		shouldFail bool
	}{
		{name: "attach ok", attached: true, sleepTime: 10 * time.Millisecond, timeout: 50 * time.Millisecond},
		{name: "attachment error", attachErr: &storage.VolumeError{Message: "missing volume"}, sleepTime: 10 * time.Millisecond, timeout: 30 * time.Millisecond},
		{name: "time ran out", attached: false, sleepTime: 5 * time.Millisecond},
	}

	for i, tc := range testCases {
		t.Logf("running test: %s", tc.name)
		pvName := fmt.Sprintf("test-pv-%d", i)
		volID := fmt.Sprintf("test-vol-%d", i)
		attachID := getAttachmentName(volID, testDriver, nodeName)
		attachment := makeTestAttachment(attachID, nodeName, pvName)
		attachment.Status.Attached = tc.attached
		attachment.Status.AttachError = tc.attachErr
		csiAttacher.waitSleepTime = tc.sleepTime

		go func() {
			_, err := csiAttacher.k8s.StorageV1beta1().VolumeAttachments().Create(attachment)
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}
		}()

		retID, err := csiAttacher.waitForVolumeAttachment(volID, attachID, tc.timeout)
		if tc.shouldFail && err == nil {
			t.Error("expecting failure, but err is nil")
		}
		if tc.attachErr != nil {
			if tc.attachErr.Message != err.Error() {
				t.Errorf("expecting error [%v], got [%v]", tc.attachErr.Message, err.Error())
			}
		}
		if err == nil && retID != attachID {
			t.Errorf("attacher.WaitForAttach not returning attachment ID")
		}
	}
}

func TestAttacherVolumesAreAttached(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
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
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}
	csiAttacher := attacher.(*csiAttacher)
	nodeName := "test-node"
	testCases := []struct {
		name       string
		volID      string
		attachID   string
		shouldFail bool
	}{
		{name: "normal test", volID: "vol-001", attachID: getAttachmentName("vol-001", testDriver, nodeName)},
		{name: "normal test 2", volID: "vol-002", attachID: getAttachmentName("vol-002", testDriver, nodeName)},
		{name: "object not found", volID: "vol-001", attachID: getAttachmentName("vol-002", testDriver, nodeName), shouldFail: true},
	}

	for _, tc := range testCases {
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
