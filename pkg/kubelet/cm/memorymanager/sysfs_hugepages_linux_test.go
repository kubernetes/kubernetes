//go:build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package memorymanager

import (
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestSysfsHugepagesReader(t *testing.T) {
	// Create a temporary sysfs-like directory structure
	tmpDir := t.TempDir()

	// Create node0 hugepages directory structure
	node0HugepagesDir := filepath.Join(tmpDir, "node0", "hugepages", "hugepages-2048kB")
	if err := os.MkdirAll(node0HugepagesDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Write free_hugepages file with 100 pages
	freeHugepagesPath := filepath.Join(node0HugepagesDir, "free_hugepages")
	if err := os.WriteFile(freeHugepagesPath, []byte("100\n"), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Create reader with custom path
	reader := newSysfsHugepagesReaderWithPath(tmpDir)

	// Test reading free hugepages
	// 2048kB = 2097152 bytes
	freePages, err := reader.GetFreeHugepages(0, 2097152)
	if err != nil {
		t.Errorf("GetFreeHugepages failed: %v", err)
	}
	if freePages != 100 {
		t.Errorf("Expected 100 free pages, got %d", freePages)
	}
}

func TestSysfsHugepagesReaderNonExistent(t *testing.T) {
	tmpDir := t.TempDir()
	reader := newSysfsHugepagesReaderWithPath(tmpDir)

	// Test reading from non-existent NUMA node
	_, err := reader.GetFreeHugepages(99, 2097152)
	if err == nil {
		t.Error("Expected error when reading from non-existent NUMA node")
	}
}

func TestSysfsHugepagesReaderMultiplePageSizes(t *testing.T) {
	tmpDir := t.TempDir()

	testCases := []struct {
		name           string
		numaNode       int
		pageSizeKB     uint64
		pageSizeBytes  uint64
		freePages      string
		expectedPages  uint64
	}{
		{
			name:           "2Mi hugepages",
			numaNode:       0,
			pageSizeKB:     2048,
			pageSizeBytes:  2 * 1024 * 1024, // 2Mi
			freePages:      "50",
			expectedPages:  50,
		},
		{
			name:           "1Gi hugepages",
			numaNode:       1,
			pageSizeKB:     1048576,
			pageSizeBytes:  1024 * 1024 * 1024, // 1Gi
			freePages:      "10",
			expectedPages:  10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create directory structure
			hugepagesDir := filepath.Join(tmpDir,
				"node"+string(rune('0'+tc.numaNode)),
				"hugepages",
				"hugepages-"+itoa(tc.pageSizeKB)+"kB")
			if err := os.MkdirAll(hugepagesDir, 0755); err != nil {
				t.Fatalf("Failed to create test directory: %v", err)
			}

			// Write free_hugepages
			freeHugepagesPath := filepath.Join(hugepagesDir, "free_hugepages")
			if err := os.WriteFile(freeHugepagesPath, []byte(tc.freePages+"\n"), 0644); err != nil {
				t.Fatalf("Failed to write test file: %v", err)
			}

			reader := newSysfsHugepagesReaderWithPath(tmpDir)
			freePages, err := reader.GetFreeHugepages(tc.numaNode, tc.pageSizeBytes)
			if err != nil {
				t.Errorf("GetFreeHugepages failed: %v", err)
			}
			if freePages != tc.expectedPages {
				t.Errorf("Expected %d free pages, got %d", tc.expectedPages, freePages)
			}
		})
	}
}

func TestResourceNameToPageSize(t *testing.T) {
	testCases := []struct {
		resourceName   v1.ResourceName
		expectedSize   uint64
		expectError    bool
	}{
		{
			resourceName:   v1.ResourceName("hugepages-2Mi"),
			expectedSize:   2 * 1024 * 1024,
			expectError:    false,
		},
		{
			resourceName:   v1.ResourceName("hugepages-1Gi"),
			expectedSize:   1024 * 1024 * 1024,
			expectError:    false,
		},
		{
			resourceName:   v1.ResourceMemory,
			expectedSize:   0,
			expectError:    true,
		},
		{
			resourceName:   v1.ResourceName("cpu"),
			expectedSize:   0,
			expectError:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(string(tc.resourceName), func(t *testing.T) {
			size, err := resourceNameToPageSize(tc.resourceName)
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if size != tc.expectedSize {
					t.Errorf("Expected size %d, got %d", tc.expectedSize, size)
				}
			}
		})
	}
}

// Helper function to convert uint64 to string without importing strconv
func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}
