//go:build windows
// +build windows

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

package filesystem

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"golang.org/x/sys/windows"
)

func TestWindowsChmod(t *testing.T) {
	// Note: OWNER will be replaced with the actual owner SID in the test cases
	testCases := []struct {
		fileMode           os.FileMode
		expectedDescriptor string
	}{
		{
			fileMode:           0777,
			expectedDescriptor: "O:OWNERG:BAD:PAI(A;OICI;FA;;;OWNER)(A;OICI;FA;;;BA)(A;OICI;FA;;;BU)",
		},
		{
			fileMode:           0750,
			expectedDescriptor: "O:OWNERG:BAD:PAI(A;OICI;FA;;;OWNER)(A;OICI;0x1200a9;;;BA)", // 0x1200a9 = GENERIC_READ | GENERIC_EXECUTE
		},
		{
			fileMode:           0664,
			expectedDescriptor: "O:OWNERG:BAD:PAI(A;OICI;0x12019f;;;OWNER)(A;OICI;0x12019f;;;BA)(A;OICI;FR;;;BU)", // 0x12019f = GENERIC_READ | GENERIC_WRITE
		},
	}

	for _, testCase := range testCases {
		tempDir, err := os.MkdirTemp("", "test-dir")
		require.NoError(t, err, "Failed to create temporary directory.")
		defer os.RemoveAll(tempDir)

		// Set the file GROUP to BUILTIN\Administrators (BA) for test determinism and
		err = setGroupInfo(tempDir, "S-1-5-32-544")
		require.NoError(t, err, "Failed to set group for directory.")

		err = Chmod(tempDir, testCase.fileMode)
		require.NoError(t, err, "Failed to set permissions for directory.")

		owner, descriptor, err := getPermissionsInfo(tempDir)
		require.NoError(t, err, "Failed to get permissions for directory.")

		expectedDescriptor := strings.ReplaceAll(testCase.expectedDescriptor, "OWNER", owner)

		assert.Equal(t, expectedDescriptor, descriptor, "Unexpected DACL for directory. when setting permissions to %o", testCase.fileMode)
	}
}

// Gets the owner and entire security descriptor of a file or directory in the SDDL format
// https://learn.microsoft.com/en-us/windows/win32/secauthz/security-descriptor-definition-language
func getPermissionsInfo(path string) (string, string, error) {
	sd, err := windows.GetNamedSecurityInfo(
		path,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.OWNER_SECURITY_INFORMATION|windows.GROUP_SECURITY_INFORMATION)
	if err != nil {
		return "", "", fmt.Errorf("Error getting security descriptor for file %s: %v", path, err)
	}

	owner, _, err := sd.Owner()
	if err != nil {
		return "", "", fmt.Errorf("Error getting owner SID for file %s: %v", path, err)
	}

	sdString := sd.String()

	return owner.String(), sdString, nil
}

// Sets the GROUP of a file or a directory to the specified group
func setGroupInfo(path, group string) error {
	groupSID, err := windows.StringToSid(group)
	if err != nil {
		return fmt.Errorf("Error converting group name %s to SID: %v", group, err)

	}

	err = windows.SetNamedSecurityInfo(
		path,
		windows.SE_FILE_OBJECT,
		windows.GROUP_SECURITY_INFORMATION,
		nil, // owner SID
		groupSID,
		nil, // DACL
		nil, //SACL
	)

	if err != nil {
		return fmt.Errorf("Error setting group SID for file %s: %v", path, err)
	}

	return nil
}

// TestDeleteFilePermissions tests that when a folder's permissions are set to 0660, child items
// cannot be deleted in the folder but when a folder's permissions are set to 0770, child items can be deleted.
func TestDeleteFilePermissions(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "test-dir")
	require.NoError(t, err, "Failed to create temporary directory.")

	err = Chmod(tempDir, 0660)
	require.NoError(t, err, "Failed to set permissions for directory to 0660.")

	filePath := filepath.Join(tempDir, "test-file")
	err = os.WriteFile(filePath, []byte("test"), 0440)
	require.NoError(t, err, "Failed to create file in directory.")

	err = os.Remove(filePath)
	require.Error(t, err, "Expected expected error when trying to remove file in directory.")

	err = Chmod(tempDir, 0770)
	require.NoError(t, err, "Failed to set permissions for directory to 0770.")

	err = os.Remove(filePath)
	require.NoError(t, err, "Failed to remove file in directory.")

	err = os.Remove(tempDir)
	require.NoError(t, err, "Failed to remove directory.")
}
