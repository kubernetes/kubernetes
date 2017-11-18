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

package csi

import (
	"crypto/sha256"
	"fmt"
	"os"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1alpha1"
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
		pv         *v1.PersistentVolume
		nodeName   string
		attachHash [32]byte
		shouldFail bool
	}{
		{
			name:       "test ok 1",
			pv:         makeTestPV("test-pv-001", 10, testDriver, testVol),
			nodeName:   "test-node",
			attachHash: sha256.Sum256([]byte(fmt.Sprintf("%s%s", "test-pv-001", "test-node"))),
		},
		{
			name:       "test ok 2",
			pv:         makeTestPV("test-pv-002", 10, testDriver, testVol),
			nodeName:   "test-node",
			attachHash: sha256.Sum256([]byte(fmt.Sprintf("%s%s", "test-pv-002", "test-node"))),
		},
		{
			name:       "missing spec",
			pv:         nil,
			nodeName:   "test-node",
			attachHash: sha256.Sum256([]byte(fmt.Sprintf("%s%s", "test-pv-002", "test-node"))),
			shouldFail: true,
		},
	}

	for _, tc := range testCases {
		var spec *volume.Spec = nil
		if tc.pv != nil {
			spec = volume.NewSpecFromPersistentVolume(tc.pv, tc.pv.Spec.PersistentVolumeSource.CSI.ReadOnly)
		}

		attachID, err := csiAttacher.Attach(spec, types.NodeName(tc.nodeName))
		if tc.shouldFail && err == nil {
			t.Error("expected failure, but got nil err")
		}
		if attachID != "" {
			expectedID := fmt.Sprintf("pv-%x", tc.attachHash)
			if attachID != expectedID {
				t.Errorf("expecting attachID %v, got %v", expectedID, attachID)
			}
		}
	}
}

func TestAttacherWaitForAttach(t *testing.T) {
	plug, tmpDir := newTestPlugin(t)
	defer os.RemoveAll(tmpDir)

	attacher, err := plug.NewAttacher()
	if err != nil {
		t.Fatalf("failed to create new attacher: %v", err)
	}
	csiAttacher := attacher.(*csiAttacher)
	nodeName := "test-node"
	pod := &v1.Pod{ObjectMeta: meta.ObjectMeta{UID: testPodUID, Namespace: testns}}

	testCases := []struct {
		name       string
		attached   bool
		attachErr  *storage.VolumeError
		sleepTime  time.Duration
		timeout    time.Duration
		shouldFail bool
	}{
		{name: "attach ok", attached: true, sleepTime: 10 * time.Millisecond, timeout: 100 * time.Millisecond},
		{name: "attachment error", attachErr: &storage.VolumeError{Message: "missing volume"}, sleepTime: 10 * time.Millisecond, timeout: 100 * time.Millisecond},
		{name: "time ran out", attached: false, sleepTime: 1 * time.Millisecond, timeout: 10 * time.Millisecond},
	}

	for i, tc := range testCases {
		t.Logf("running test: %v", tc.name)
		pvName := fmt.Sprintf("test-pv-%d", i)
		attachID := fmt.Sprintf("pv-%s", hashAttachmentName(pvName, nodeName))
		pv := makeTestPV(pvName, 10, testDriver, testVol)
		spec := volume.NewSpecFromPersistentVolume(pv, pv.Spec.PersistentVolumeSource.CSI.ReadOnly)

		attachment := makeTestAttachment(attachID, nodeName, pvName)
		attachment.Status.Attached = tc.attached
		attachment.Status.AttachError = tc.attachErr
		csiAttacher.waitSleepTime = tc.sleepTime

		go func() {
			_, err := csiAttacher.k8s.StorageV1alpha1().VolumeAttachments().Create(attachment)
			if err != nil {
				t.Fatalf("failed to attach: %v", err)
			}
		}()

		retID, err := csiAttacher.WaitForAttach(spec, attachID, pod, tc.timeout)
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
