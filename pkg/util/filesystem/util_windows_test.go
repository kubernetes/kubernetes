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
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	winio "github.com/Microsoft/go-winio"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"golang.org/x/sys/windows"
)

func TestIsUnixDomainSocketPipe(t *testing.T) {
	generatePipeName := func(suffixLen int) string {
		letter := []rune("abcdef0123456789")
		b := make([]rune, suffixLen)
		for i := range b {
			b[i] = letter[rand.Intn(len(letter))]
		}
		return "\\\\.\\pipe\\test-pipe" + string(b)
	}
	testFile := generatePipeName(4)
	pipeln, err := winio.ListenPipe(testFile, &winio.PipeConfig{SecurityDescriptor: "D:P(A;;GA;;;BA)(A;;GA;;;SY)"})
	defer pipeln.Close()

	require.NoErrorf(t, err, "Failed to listen on named pipe for test purposes: %v", err)
	result, err := IsUnixDomainSocket(testFile)
	assert.NoError(t, err, "Unexpected error from IsUnixDomainSocket.")
	assert.False(t, result, "Unexpected result: true from IsUnixDomainSocket.")
}

// This is required as on Windows it's possible for the socket file backing a Unix domain socket to
// exist but not be ready for socket communications yet as per
// https://github.com/kubernetes/kubernetes/issues/104584
func TestPendingUnixDomainSocket(t *testing.T) {
	// Create a temporary file that will simulate the Unix domain socket file in a
	// not-yet-ready state. We need this because the Kubelet keeps an eye on file
	// changes and acts on them, leading to potential race issues as described in
	// the referenced issue above
	f, err := os.CreateTemp("", "test-domain-socket")
	require.NoErrorf(t, err, "Failed to create file for test purposes: %v", err)
	testFile := f.Name()
	f.Close()

	// Start the check at this point
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		result, err := IsUnixDomainSocket(testFile)
		assert.Nil(t, err, "Unexpected error from IsUnixDomainSocket: %v", err)
		assert.True(t, result, "Unexpected result: false from IsUnixDomainSocket.")
		wg.Done()
	}()

	// Wait a sufficient amount of time to make sure the retry logic kicks in
	time.Sleep(socketDialRetryPeriod)

	// Replace the temporary file with an actual Unix domain socket file
	os.Remove(testFile)
	ta, err := net.ResolveUnixAddr("unix", testFile)
	require.NoError(t, err, "Failed to ResolveUnixAddr.")
	unixln, err := net.ListenUnix("unix", ta)
	require.NoError(t, err, "Failed to ListenUnix.")

	// Wait for the goroutine to finish, then close the socket
	wg.Wait()
	unixln.Close()
}

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

		// Set the file GROUP to BUILTIN\Administrators (BA) for test determinism
		err = setGroupInfo(tempDir, "S-1-5-32-544")
		require.NoError(t, err, "Failed to set group for directory.")

		err = Chmod(tempDir, testCase.fileMode)
		require.NoError(t, err, "Failed to set permissions for directory.")

		owner, descriptor, err := getPermissionsInfo(tempDir)
		require.NoError(t, err, "Failed to get permissions for directory.")

		if owner == "S-1-5-32-544" {
			owner = "BA"
		}

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
	testDir, err := os.MkdirTemp("", "test-dir")
	require.NoError(t, err, "Failed to create temporary directory.")

	// Remove inherited permissions for %TEMP% since this tests runs as a system account
	// in CI which has access to all files and folders by default
	cmd := exec.Command("icacls.exe", testDir, "/inheritance:r")
	output, err := cmd.CombinedOutput()
	require.NoError(t, err, "Failed to disable inheritance for directory: %s", output)

	_, afterInheritanceRemoval, _ := getPermissionsInfo(testDir)
	err = Chmod(testDir, 0660)
	_, afterChmod, _ := getPermissionsInfo(testDir)
	require.Fail(t, fmt.Sprintf("Permissions after removing inheritance %s, after chmod %s", afterInheritanceRemoval, afterChmod))

	require.NoError(t, err, "Failed to set permissions for directory to 0660.")

	filePath := filepath.Join(testDir, "test-file")
	err = os.WriteFile(filePath, []byte("test"), 0440)
	require.NoError(t, err, "Failed to create file in directory.")

	// temp logging to figure out why file is getting delete in CI when it should not
	_, folderDescriptior, _ := getPermissionsInfo(testDir)
	_, fileDescriptior, _ := getPermissionsInfo(filePath)
	err = os.Remove(filePath)
	require.Error(t, err, "Expected expected error when trying to remove file in directory. Folder perms %s, file perms %s", folderDescriptior, fileDescriptior)

	err = Chmod(testDir, 0770)
	require.NoError(t, err, "Failed to set permissions for directory to 0770.")

	err = os.Remove(filePath)
	require.NoError(t, err, "Failed to remove file in directory.")

	err = os.Remove(testDir)
	require.NoError(t, err, "Failed to remove directory.")
}

func TestAbsWithSlash(t *testing.T) {
	// On Windows, filepath.IsAbs will not return True for paths prefixed with a slash
	assert.True(t, IsAbs("/test"))
	assert.True(t, IsAbs("\\test"))

	assert.False(t, IsAbs("./local"))
	assert.False(t, IsAbs("local"))
}
