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

package kuberuntime

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

func TestLogSymLink(t *testing.T) {
	as := assert.New(t)
	containerLogsDir := "/foo/bar"
	podFullName := randStringBytes(128)
	containerName := randStringBytes(70)
	containerID := randStringBytes(80)
	// The file name cannot exceed 255 characters. Since .log suffix is required, the prefix cannot exceed 251 characters.
	expectedPath := filepath.Join(containerLogsDir, fmt.Sprintf("%s_%s-%s", podFullName, containerName, containerID)[:251]+".log")
	as.Equal(expectedPath, logSymlink(containerLogsDir, podFullName, containerName, containerID))
}

func TestLegacyLogSymLink(t *testing.T) {
	as := assert.New(t)
	containerID := randStringBytes(80)
	containerName := randStringBytes(70)
	podName := randStringBytes(128)
	podNamespace := randStringBytes(10)
	// The file name cannot exceed 255 characters. Since .log suffix is required, the prefix cannot exceed 251 characters.
	expectedPath := filepath.Join(legacyContainerLogsDir, fmt.Sprintf("%s_%s_%s-%s", podName, podNamespace, containerName, containerID)[:251]+".log")
	as.Equal(expectedPath, legacyLogSymlink(containerID, containerName, podName, podNamespace))
}

func TestGetContainerIDFromLegacyLogSymLink(t *testing.T) {
	containerID := randStringBytes(80)
	containerName := randStringBytes(70)
	podName := randStringBytes(128)
	podNamespace := randStringBytes(10)

	for _, test := range []struct {
		name        string
		logSymLink  string
		expected    string
		shouldError bool
	}{
		{
			name:        "unable to find separator",
			logSymLink:  "dummy.log",
			expected:    "",
			shouldError: true,
		},
		{
			name:        "invalid suffix",
			logSymLink:  filepath.Join(legacyContainerLogsDir, fmt.Sprintf("%s_%s_%s-%s", podName, podNamespace, containerName, containerID)[:251]+".invalidsuffix"),
			expected:    "",
			shouldError: true,
		},
		{
			name:        "container ID too short",
			logSymLink:  filepath.Join(legacyContainerLogsDir, fmt.Sprintf("%s_%s_%s-%s", podName, podNamespace, containerName, containerID[:5])+".log"),
			expected:    "",
			shouldError: true,
		},
		{
			name:        "valid path",
			logSymLink:  filepath.Join(legacyContainerLogsDir, fmt.Sprintf("%s_%s_%s-%s", podName, podNamespace, containerName, containerID)[:251]+".log"),
			expected:    containerID[:40],
			shouldError: false,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			containerID, err := getContainerIDFromLegacyLogSymlink(test.logSymLink)
			if test.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			assert.Equal(t, test.expected, containerID)
		})
	}
}
