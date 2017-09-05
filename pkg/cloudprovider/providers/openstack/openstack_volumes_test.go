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

package openstack

import (
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	volumeAvailableStatus = "available"
	volumeInUseStatus     = "in-use"
	testClusterName       = "testCluster"

	volumeStatusTimeoutSeconds = 30
	// volumeStatus* is configuration of exponential backoff for
	// waiting for specified volume status. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 13 steps at maximum
	// it will time out after 32s, which roughly corresponds to 30s
	volumeStatusInitDealy = 1 * time.Second
	volumeStatusFactor    = 1.2
	volumeStatusSteps     = 13
)

func WaitForVolumeStatus(t *testing.T, os *OpenStack, volumeName string, status string) {
	backoff := wait.Backoff{
		Duration: volumeStatusInitDealy,
		Factor:   volumeStatusFactor,
		Steps:    volumeStatusSteps,
	}
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		getVol, err := os.getVolume(volumeName)
		if err != nil {
			return false, err
		}
		if getVol.Status == status {
			t.Logf("Volume (%s) status changed to %s after %v seconds\n",
				volumeName,
				status,
				volumeStatusTimeoutSeconds)
			return true, nil
		} else {
			return false, nil
		}
	})
	if err == wait.ErrWaitTimeout {
		t.Logf("Volume (%s) status did not change to %s after %v seconds\n",
			volumeName,
			status,
			volumeStatusTimeoutSeconds)
		return
	}
	if err != nil {
		t.Fatalf("Cannot get existing Cinder volume (%s): %v", volumeName, err)
	}
}

func TestVolumes(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	os, err := newOpenStack(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate OpenStack: %s", err)
	}

	tags := map[string]string{
		"test": "value",
	}
	vol, _, err := os.CreateVolume("kubernetes-test-volume-"+rand.String(10), 1, "", "", &tags)
	if err != nil {
		t.Fatalf("Cannot create a new Cinder volume: %v", err)
	}
	t.Logf("Volume (%s) created\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	id, err := os.InstanceID()
	if err != nil {
		t.Fatalf("Cannot find instance id: %v", err)
	}

	diskId, err := os.AttachDisk(id, vol)
	if err != nil {
		t.Fatalf("Cannot AttachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) attached, disk ID: %s\n", vol, diskId)

	WaitForVolumeStatus(t, os, vol, volumeInUseStatus)

	devicePath := os.GetDevicePath(diskId)
	if !strings.HasPrefix(devicePath, "/dev/disk/by-id/") {
		t.Fatalf("GetDevicePath returned and unexpected path for Cinder volume %s, returned %s", vol, devicePath)
	}
	t.Logf("Volume (%s) found at path: %s\n", vol, devicePath)

	err = os.DetachDisk(id, vol)
	if err != nil {
		t.Fatalf("Cannot DetachDisk Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) detached\n", vol)

	WaitForVolumeStatus(t, os, vol, volumeAvailableStatus)

	err = os.DeleteVolume(vol)
	if err != nil {
		t.Fatalf("Cannot delete Cinder volume %s: %v", vol, err)
	}
	t.Logf("Volume (%s) deleted\n", vol)

}
