//go:build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package cadvisor

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/google/cadvisor/container/crio"
	cadvisorfs "github.com/google/cadvisor/fs"
	"github.com/opencontainers/cgroups"
	"k8s.io/klog/v2"
)

func TestIsPsiEnabled(t *testing.T) {
	testcases := []struct {
		description   string
		createPSIFile bool
		expected      bool
	}{{
		description:   "PSI enabled when cgroup pressure file exists",
		createPSIFile: true,
		expected:      true,
	}, {
		description:   "PSI disabled when cgroup pressure file does not exist",
		createPSIFile: false,
		expected:      false,
	}}

	cgroups.TestMode = true
	defer func() { cgroups.TestMode = false }()

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			tmpDir := t.TempDir()

			if tc.createPSIFile {
				err := os.WriteFile(filepath.Join(tmpDir, "cpu.pressure"), []byte("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"), 0644)
				require.NoError(t, err)
			}

			result := isPsiEnabled(klog.Background(), tmpDir, "cpu.pressure")
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestImageFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioImages,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no imagefs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ImageFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}

func TestContainerFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioWriteableImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioContainers,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no containerfs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ContainerFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}
