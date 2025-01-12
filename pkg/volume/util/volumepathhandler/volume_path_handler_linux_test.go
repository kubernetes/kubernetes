/*
Copyright 2023 The Kubernetes Authors.

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

package volumepathhandler

import (
	"fmt"
	"path/filepath"
	"testing"

	utilexec "k8s.io/utils/exec"
)

func pathWithSuffix(suffix string) string {
	return fmt.Sprintf("%s%s", "/var/lib/kubelet/plugins/kubernetes.io/csi/volumeDevices/pvc-1d205234-06cd-4fe4-a7ea-0e8f3e2faf5f/dev/e196ebd3-2ab1-4185-bed4-b997ba38d1dc", suffix)
}

func createTestDevice(t *testing.T) string {
	executor := utilexec.New()
	backingFile := filepath.Join(t.TempDir(), "backingFile")

	out, err := executor.Command("fallocate", "-l", "1M", backingFile).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to create backing file: %v, %v", err, string(out))
	}
	devicePath, err := makeLoopDevice(backingFile)
	if err != nil {
		t.Fatalf("failed to create loop device: %v", err)
	}
	t.Cleanup(func() {
		err := removeLoopDevice(devicePath)
		if err != nil {
			t.Errorf("failed to remove loop device: %v", err)
		}
	})
	return devicePath
}

func TestCleanBackingFilePath(t *testing.T) {
	const defaultPath = "/var/lib/kubelet/plugins/kubernetes.io/csi/volumeDevices/pvc-1d205234-06cd-4fe4-a7ea-0e8f3e2faf5f/dev/e196ebd3-2ab1-4185-bed4-b997ba38d1dc"
	testCases := []struct {
		name          string
		input         string
		expectedOuput string
	}{
		{
			name:          "regular path",
			input:         defaultPath,
			expectedOuput: defaultPath,
		},
		{
			name:          "path is suffixed with whitespaces",
			input:         fmt.Sprintf("%s\r\t\n ", defaultPath),
			expectedOuput: defaultPath,
		},
		{
			name:          "path is suffixed with \"(deleted)\"",
			input:         pathWithSuffix("(deleted)"),
			expectedOuput: defaultPath,
		},
		{
			name:          "path is suffixed with \"(deleted)\" and whitespaces",
			input:         pathWithSuffix(" (deleted)\t"),
			expectedOuput: defaultPath,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output := cleanBackingFilePath(tc.input)
			if output != tc.expectedOuput {
				t.Fatalf("expected %q, got %q", tc.expectedOuput, output)
			}
		})
	}
}

func TestIsDeviceBindMountExist(t *testing.T) {
	devicePath := createTestDevice(t)

	handler := VolumePathHandler{}

	mapPath := t.TempDir()
	linkName := "link"
	err := mapBindMountDevice(devicePath, mapPath, linkName)
	if err != nil {
		t.Fatalf("failed to map bind mount: %v", err)
	}
	t.Cleanup(func() {
		err := unmapBindMountDevice(handler, mapPath, linkName)
		if err != nil {
			t.Fatalf("failed to unmap bind mount: %v", err)
		}
	})

	linkPath := filepath.Join(mapPath, linkName)

	testCases := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "existing device bind mount",
			path:     linkPath,
			expected: true,
		},
		{
			name:     "non-existing path",
			path:     filepath.Join(mapPath, "non_existing_link"),
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			exists, err := handler.IsDeviceBindMountExist(tc.path)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if exists != tc.expected {
				t.Fatalf("expected %v, got %v", tc.expected, exists)
			}
		})
	}
}
