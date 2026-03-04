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
	"testing"
)

func pathWithSuffix(suffix string) string {
	return fmt.Sprintf("%s%s", "/var/lib/kubelet/plugins/kubernetes.io/csi/volumeDevices/pvc-1d205234-06cd-4fe4-a7ea-0e8f3e2faf5f/dev/e196ebd3-2ab1-4185-bed4-b997ba38d1dc", suffix)
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
