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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/google/cadvisor/container/crio"
	cadvisorfs "github.com/google/cadvisor/fs"
)

func TestIsPsiEnabled(t *testing.T) {
	testcases := []struct {
		description    string
		createPSIDir   bool
		cmdlineContent string
		expected       bool
	}{{
		description:    "PSI enabled when pressure directory exists and no psi=0 in cmdline",
		createPSIDir:   true,
		cmdlineContent: "BOOT_IMAGE=/vmlinuz root=/dev/sda1 ro quiet",
		expected:       true,
	}, {
		description:    "PSI disabled when pressure directory does not exist",
		createPSIDir:   false,
		cmdlineContent: "BOOT_IMAGE=/vmlinuz root=/dev/sda1 ro quiet",
		expected:       false,
	}, {
		description:    "PSI disabled when pressure directory exists but psi=0 in cmdline",
		createPSIDir:   true,
		cmdlineContent: "BOOT_IMAGE=/vmlinuz root=/dev/sda1 ro quiet psi=0",
		expected:       false,
	}, {
		description:    "PSI enabled when psi=0 followed by psi=1 (last value wins)",
		createPSIDir:   true,
		cmdlineContent: "BOOT_IMAGE=/vmlinuz psi=0 psi=1",
		expected:       true,
	}, {
		description:    "PSI disabled when psi=1 followed by psi=0 (last value wins)",
		createPSIDir:   true,
		cmdlineContent: "BOOT_IMAGE=/vmlinuz psi=1 psi=0",
		expected:       false,
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			tmpDir := t.TempDir()

			psiPath := filepath.Join(tmpDir, "pressure")
			if tc.createPSIDir {
				err := os.Mkdir(psiPath, 0755)
				require.NoError(t, err)
			}

			cmdlinePath := filepath.Join(tmpDir, "cmdline")
			err := os.WriteFile(cmdlinePath, []byte(tc.cmdlineContent), 0644)
			require.NoError(t, err)

			result := isPsiEnabled(context.Background(), psiPath, cmdlinePath)
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
