// +build !providerless

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

package azuredd

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
)

func createVolSpec(name string, readOnly bool) *volume.Spec {
	return &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				AzureDisk: &v1.AzureDiskVolumeSource{
					DiskName: name,
					ReadOnly: &readOnly,
				},
			},
		},
	}
}
func TestWaitForAttach(t *testing.T) {
	tests := []struct {
		devicePath  string
		expected    string
		expectError bool
	}{
		{
			devicePath:  "/dev/disk/azure/scsi1/lun0",
			expected:    "/dev/disk/azure/scsi1/lun0",
			expectError: false,
		},
		{
			devicePath:  "/dev/sdc",
			expected:    "/dev/sdc",
			expectError: false,
		},
		{
			devicePath:  "/dev/disk0",
			expected:    "/dev/disk0",
			expectError: false,
		},
	}

	attacher := azureDiskAttacher{}
	spec := createVolSpec("fakedisk", false)

	for _, test := range tests {
		result, err := attacher.WaitForAttach(spec, test.devicePath, nil, 3000*time.Millisecond)
		assert.Equal(t, result, test.expected)
		assert.Equal(t, err != nil, test.expectError, fmt.Sprintf("error msg: %v", err))
	}
}
