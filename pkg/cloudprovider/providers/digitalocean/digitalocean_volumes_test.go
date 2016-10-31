/*
Copyright 2016 The Kubernetes Authors.

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

package digitalocean

import (
	"os"
	"strconv"
	"testing"
)

func TestCreateDeleteVolume(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	do, err := newDigitalOcean(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate Digitalocean: %s", err)
	}
	// Create Volume
	volumeID, err := do.CreateVolume("nyc1", "testvolume", "This is a test volume", 8)
	if err != nil {
		t.Fatalf("createVolume returned false: %s", err)
	} else {
		t.Logf("Created volume with ID: %s\n", volumeID)
	}
	// Attach Volume
	// Optionally pass instance ID to test volume attachments
	strInstanceID := os.Getenv("DO_INSTANCEID")
	if strInstanceID != "" {
		int64InstanceID, _ := strconv.ParseInt(strInstanceID, 10, 0)
		instanceID := int(int64InstanceID)
		_, err = do.AttachVolume(instanceID, volumeID)
		if err != nil {
			t.Fatalf("AttachVolume returned false: %s", err)
		} else {
			t.Logf("Attached volume with ID: %s to instance with ID %d\n", volumeID, instanceID)
		}
		volumePath, err := do.GetAttachmentVolumePath(instanceID, volumeID)
		if err != nil {
			t.Fatalf("Volume Path returned false: %s", err)
		} else {
			t.Logf("Volume Path: %s\n", volumePath)
		}
		err = do.DetachVolume(instanceID, volumeID)
		if err != nil {
			t.Fatalf("Detach returned false: %s", err)
		} else {
			t.Logf("Detach volume with ID: %s to instance with ID %d\n", volumeID, instanceID)
		}
	}
	// Delete Volume
	err = do.DeleteVolume(volumeID)
	if err != nil {
		t.Fatalf("deleteVolume returned false: %s", err)
	} else {
		t.Logf("Deleted volume with ID: %s\n", volumeID)
	}
}
