//go:build windows
// +build windows

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

package mount

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeWindowsPath(t *testing.T) {
	path := `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4/volumes/kubernetes.io~azure-disk`
	normalizedPath := NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/var/lib/kubelet/pods/146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\var\lib\kubelet\pods\146f8428-83e7-11e7-8dd4-000d3a31dac4\volumes\kubernetes.io~azure-disk` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}

	path = `/`
	normalizedPath = NormalizeWindowsPath(path)
	if normalizedPath != `c:\` {
		t.Errorf("normizeWindowsPath test failed, normalizedPath : %q", normalizedPath)
	}
}

func TestValidateDiskNumber(t *testing.T) {
	tests := []struct {
		diskNum     string
		expectError bool
	}{
		{
			diskNum:     "0",
			expectError: false,
		},
		{
			diskNum:     "invalid",
			expectError: true,
		},
		{
			diskNum:     "99",
			expectError: false,
		},
		{
			diskNum:     "100",
			expectError: false,
		},
		{
			diskNum:     "200",
			expectError: false,
		},
	}

	for _, test := range tests {
		err := ValidateDiskNumber(test.diskNum)
		if (err != nil) != test.expectError {
			t.Errorf("TestValidateDiskNumber test failed, disk number: %s, error: %v", test.diskNum, err)
		}
	}
}

func TestIsPathValid(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "valid-file")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	tmpDir, err := os.MkdirTemp("", "valid-dir")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	nonexistent := filepath.Join(os.TempDir(), "does-not-exist-"+filepath.Base(tmpFile.Name()))

	invalid := string([]byte{0x00}) // illegal null character

	tests := []struct {
		name             string
		path             string
		expectValid      bool
		expectErrMessage string
	}{
		{
			name:             "ValidFile",
			path:             tmpFile.Name(),
			expectValid:      true,
			expectErrMessage: "",
		},
		{
			name:             "ValidDirectory",
			path:             tmpDir,
			expectValid:      true,
			expectErrMessage: "",
		},
		{
			name:             "NonExistentPath",
			path:             nonexistent,
			expectValid:      false,
			expectErrMessage: "",
		},
		{
			name:             "InvalidPath",
			path:             invalid,
			expectValid:      false,
			expectErrMessage: "invalid path: invalid argument",
		},
		{
			name:             "Drive C",
			path:             "c:",
			expectValid:      true,
			expectErrMessage: "",
		},
		{
			name:             "InvalidRemotePath",
			path:             "invalid-remote-path",
			expectValid:      false,
			expectErrMessage: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, err := IsPathValid(tt.path)
			if valid != tt.expectValid {
				t.Errorf("Expected valid = %v, got %v", tt.expectValid, valid)
			}
			if err == nil && tt.expectErrMessage != "" {
				t.Errorf("Expected error message = %s, got no error", tt.expectErrMessage)
			}
			if err != nil {
				if tt.expectErrMessage != "" && err.Error() != tt.expectErrMessage {
					t.Errorf("Expected error message = %s, got error = %s", tt.expectErrMessage, err.Error())
				} else if tt.expectErrMessage == "" {
					t.Errorf("Expected no error, got error = %s", err.Error())
				}
			}
		})
	}
}

func runPowershellCmd(t *testing.T, command string) (string, error) {
	cmd := exec.Command("powershell", "/c", fmt.Sprintf("& { $global:ProgressPreference = 'SilentlyContinue'; %s }", command))
	t.Logf("Executing command: %q", cmd.String())
	result, err := cmd.CombinedOutput()
	return string(result), err
}

func createMountedFolder(t *testing.T, vhdxPath, mountedPath string, initialSize int) {
	cmd := fmt.Sprintf("New-VHD -Path %s -SizeBytes %d", vhdxPath, initialSize)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error: %v. Command: %q. Out: %s.", err, cmd, out)
	}
	cmd = fmt.Sprintf("Mount-VHD -Path %s", vhdxPath)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error: %v. Command: %q. Out: %s", err, cmd, out)
	}
	cmd = fmt.Sprintf("Mount-VHD -Path %s", vhdxPath)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error: %v. Command: %q. Out: %s", err, cmd, out)
	}
	cmd = fmt.Sprintf("(Get-VHD -Path %s).DiskNumber", vhdxPath)
	diskNumUnparsed, err := runPowershellCmd(t, cmd)
	if err != nil {
		t.Fatalf("Error: %v. Command: %s", err, cmd)
	}
	diskNumUnparsed = strings.TrimSpace(diskNumUnparsed)
	cmd = fmt.Sprintf("Initialize-Disk -Number %s -PartitionStyle GPT", diskNumUnparsed)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error initializing disk: %v. Command: %q. Out: %s", err, cmd, out)
	}
	// Create a new partition using all available space
	cmd = fmt.Sprintf("New-Partition -DiskNumber %s -UseMaximumSize", diskNumUnparsed)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error creating partition: %v. Command: %q. Out: %s", err, cmd, out)
	}
	// Format the partition with NTFS
	cmd = fmt.Sprintf("(Get-Disk -Number %s | Get-Partition | Get-Volume) | Format-Volume -FileSystem NTFS -Confirm:$false", diskNumUnparsed)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error formatting volume: %v. Command: %q. Out: %s", err, cmd, out)
	}
	cmd = fmt.Sprintf(`(Get-Disk -Number %s | Get-Partition ) | Add-PartitionAccessPath -AccessPath %s`, diskNumUnparsed, mountedPath)
	if _, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error: %v. Command: %s", err, cmd)
	}
}

func unmountFolder(t *testing.T, vhdxPath, mountedPath string) {
	cmd := fmt.Sprintf("(Get-VHD -Path %s).DiskNumber", vhdxPath)
	diskNumUnparsed, err := runPowershellCmd(t, cmd)
	if err != nil {
		t.Fatalf("Error: %v. Command: %s", err, cmd)
	}
	diskNumUnparsed = strings.TrimSpace(diskNumUnparsed)
	cmd = fmt.Sprintf(`Get-Disk -Number %s | Get-Partition | Remove-PartitionAccessPath -AccessPath %s`, diskNumUnparsed, mountedPath)
	if _, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error: %v. Command: %s", err, cmd)
	}
	cmd = fmt.Sprintf("Dismount-VHD -Path %s", vhdxPath)
	if out, err := runPowershellCmd(t, cmd); err != nil {
		t.Fatalf("Error unmounting VHD: %v. Command: %q. Out: %s", err, cmd, out)
	}
}

func TestIsMountedFolder(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "test-dir")
	require.NoError(t, err, "Failed to create temporary directory.")

	tests := []struct {
		name           string
		path           string
		setup          func()
		cleanup        func()
		expectedResult bool
		expectedError  error
	}{
		{
			name:           "Non-existent path",
			path:           filepath.Join(tempDir, "nonexistent"),
			expectedResult: false,
			expectedError:  errors.New("The system cannot find the file specified."),
		},
		{
			name: "Regular directory",
			path: filepath.Join(tempDir, "regular_dir"),
			setup: func() {
				err := os.MkdirAll(filepath.Join(tempDir, "regular_dir"), 0644)
				require.NoError(t, err, "Failed to create regular_dir directory.")
			},
			expectedResult: false,
			expectedError:  nil,
		},
		{
			name: "Mounted folder",
			path: filepath.Join(tempDir, "mounted_folder"),
			setup: func() {
				err := os.MkdirAll(filepath.Join(tempDir, "mounted_folder"), 0644)
				require.NoError(t, err, "Failed to create regular_dir directory.")

				createMountedFolder(t, filepath.Join(tempDir, "test.vhdx"), filepath.Join(tempDir, "mounted_folder"), 1024*1024*1024)
			},
			cleanup: func() {
				unmountFolder(t, filepath.Join(tempDir, "test.vhdx"), filepath.Join(tempDir, "mounted_folder"))
			},
			expectedResult: true,
			expectedError:  nil,
		},
		{
			name: "Regular file",
			path: filepath.Join(tempDir, "regular_file"),
			setup: func() {
				err := os.WriteFile(filepath.Join(tempDir, "regular_file"), []byte("just_a_test"), 0644)
				require.NoError(t, err, "Failed to create regular_file.")
			},
			expectedResult: false,
			expectedError:  nil,
		},
		{
			name: "Regular symlink",
			path: filepath.Join(tempDir, "regular_symlink"),
			setup: func() {
				err := os.WriteFile(filepath.Join(tempDir, "regular_file"), []byte("just_a_test"), 0644)
				require.NoError(t, err, "Failed to create regular_file.")

				err = os.Symlink(filepath.Join(tempDir, "regular_file"), filepath.Join(tempDir, "regular_symlink"))
				require.NoError(t, err, "Failed to create regular_file.")
			},
			cleanup: func() {
				err := os.RemoveAll(filepath.Join(tempDir, "regular_file"))
				require.NoError(t, err, "Failed to delete regular_file.")

				err = os.RemoveAll(filepath.Join(tempDir, "regular_symlink"))
				require.NoError(t, err, "Failed to delete regular_symlink.")
			},
			expectedResult: true,
			expectedError:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setup != nil {
				tt.setup()
			}

			// Run test
			result, err := IsMountedFolder(tt.path)

			if tt.cleanup != nil {
				tt.cleanup()
			}

			// Assert results
			if tt.expectedError != nil {
				assert.Error(t, err)
				assert.Equal(t, tt.expectedError.Error(), err.Error())
			} else {
				assert.NoError(t, err)
			}
			assert.Equal(t, tt.expectedResult, result)
		})
	}

	err = os.RemoveAll(tempDir)
	require.NoError(t, err, "Failed to remove directory.")
}
