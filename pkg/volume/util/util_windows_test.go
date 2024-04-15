//go:build windows
// +build windows

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

package util

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWindowsFileMode(t *testing.T) {
	// Create a temp file, and later on remove all permissions from it.
	f, err := os.CreateTemp("", "permissions_test")
	require.NoError(t, err)

	// Write a sample string into the file.
	_, err = f.WriteString("hello!")
	require.NoError(t, err)

	f.Close()
	defer os.Remove(f.Name())

	// Remove all permissions from the file.
	err = Chmod(f.Name(), 0000)
	require.NoError(t, err)

	// Assert that the File Mode changed.
	mode, err := GetFileMode(f.Name())
	require.NoError(t, err)
	assert.Equal(t, 0000, int(mode))

	// Assert that we cannot read the file, as we don't have read permissions.
	// There shouldn't be any ACEs on the file, so os.Open should end up with a "Access is denied." error.
	// However, that doesn't happen. Interestingly, it does happen in other languages, or even Powershell.
	// We test using Powershell instead.
	cmd := exec.Command("powershell.exe", "-NonInteractive", "cat", f.Name())
	var errOut bytes.Buffer
	cmd.Stderr = &errOut
	err = cmd.Run()
	expectedErrMsg := fmt.Sprintf("Access to the path '%s' is denied.", f.Name())
	if err == nil || !strings.Contains(errOut.String(), expectedErrMsg) {
		t.Fatalf("Unexpected error message while opening the file for reading. Got: %v, expected: %v", err, expectedErrMsg)
	}

	// We can still open the file in golang and read from it.
	f, err = os.Open(f.Name())
	require.NoError(t, err)

	bytes := make([]byte, 64)
	n, err := f.Read(bytes)
	require.NoError(t, err)
	assert.Equal(t, "hello!", string(bytes[:n]))

	f.Close()

	// Assert that we cannot write in the file, as we do not have write permissions.
	_, err = os.Create(f.Name())
	expectedErrMsg = "Access is denied."
	if err == nil || !strings.Contains(err.Error(), expectedErrMsg) {
		t.Fatalf("Unexpected error message while opening the file for writing. Got: %v, expected: %v", err, expectedErrMsg)
	}

	// Change the File Mode again.
	err = Chmod(f.Name(), 0644)
	require.NoError(t, err)

	mode, err = GetFileMode(f.Name())
	require.NoError(t, err)
	assert.Equal(t, 0644, int(mode))

	// Open the file in write mode, it should not fail.
	f, err = os.Create(f.Name())
	require.NoError(t, err)

	expectedContent := "another hello!"
	_, err = f.WriteString(expectedContent)
	require.NoError(t, err)

	f.Close()

	// Open the file in read mode, we should be able to read the expected content.
	f, err = os.Open(f.Name())
	require.NoError(t, err)

	n, err = f.Read(bytes)
	require.NoError(t, err)
	assert.Equal(t, expectedContent, string(bytes[:n]))

	f.Close()

	err = os.Remove(f.Name())
	require.NoError(t, err)
}
