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

package kernel

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/version"
)

func TestGetVersion(t *testing.T) {
	testCases := []struct {
		name         string
		readFileFunc readFileFunc
		expected     *version.Version
		err          error
	}{
		{
			name: "valid os-release file",
			readFileFunc: func(_ string) ([]byte, error) {
				return []byte("5.15.0-84-generic"), nil
			},
			expected: version.MajorMinor(5, 15),
		},
		{
			name: "valid os-release file",
			readFileFunc: func(_ string) ([]byte, error) {
				return []byte("5.4.0-128-generic"), nil
			},
			expected: version.MajorMinor(5, 4),
		},
		{
			name: "valid microsoft WSL2 kernel version",
			readFileFunc: func(_ string) ([]byte, error) {
				return []byte("5.15.146.1-microsoft-standard-WSL2"), nil
			},
			expected: version.MajorMinor(5, 15).WithPatch(146),
		},
		{
			name: "failed to read os-release file",
			readFileFunc: func(_ string) ([]byte, error) {
				return nil, errors.New("open /proc/sys/kernel/osrelease: failed to read file")
			},
			err:      errors.New("failed to read os-release file: open /proc/sys/kernel/osrelease: failed to read file"),
			expected: nil,
		},
		{
			name: "version not parsable",
			readFileFunc: func(_ string) ([]byte, error) {
				return []byte("5-15-0"), nil
			},
			err:      errors.New("failed to parse kernel version: illegal version string \"5-15-0\""),
			expected: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kernelVersion, err := getVersion(tc.readFileFunc)

			if tc.err != nil {
				assert.Equal(t, tc.err.Error(), err.Error())
				assert.Nil(t, kernelVersion)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.Major(), kernelVersion.Major())
				assert.Equal(t, tc.expected.Minor(), kernelVersion.Minor())
				assert.Equal(t, tc.expected.Patch(), kernelVersion.Patch())
			}
		})
	}
}
