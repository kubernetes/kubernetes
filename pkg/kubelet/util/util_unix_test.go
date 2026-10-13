//go:build freebsd || linux || darwin

/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLocalEndpoint(t *testing.T) {
	tests := []struct {
		path             string
		file             string
		expectError      bool
		expectedFullPath string
	}{
		{
			path:             "path",
			file:             "file",
			expectError:      false,
			expectedFullPath: "unix:/path/file.sock",
		},
	}
	for _, test := range tests {
		fullPath, err := LocalEndpoint(test.path, test.file)
		if test.expectError {
			assert.Error(t, err, "expected error")
			continue
		}
		assert.NoError(t, err, "expected no error")
		assert.Equal(t, test.expectedFullPath, fullPath)
	}
}
