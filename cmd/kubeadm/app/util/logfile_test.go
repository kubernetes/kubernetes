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
	"os"
	"path/filepath"
	"testing"
)

const kubeadmLogTestFile = "kubeadm-log-file-test"

var kubeadmLogTestFolder = os.TempDir()

func TestWriteKubeadmLogFile(t *testing.T) {
	tests := []struct {
		name          string
		writeData     string
		readData      string
		expectedError bool
	}{
		{
			name:          "read/write valid data",
			writeData:     "kubeadm init",
			readData:      "kubeadm init",
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			filePath := filepath.Join(kubeadmLogTestFolder, kubeadmLogTestFile)
			err := writeKubeadmLogFile(kubeadmLogTestFolder, kubeadmLogTestFile, tc.writeData)
			if err != nil {
				t.Fatalf("failed to write %q to temporary file: %s", tc.writeData, filePath)
			}
			data, err := readKubeadmLogFile(filePath)
			if err != nil {
				t.Fatalf("failed to read from temporary file: %s", filePath)
			}
			if (data != tc.readData) == tc.expectedError {
				t.Fatalf("expected data: %s, got: %s", tc.readData, data)
			}
			if err := removeKubeadmLogFile(filePath); err != nil {
				t.Fatalf("failed to remove temporary file: %s", filePath)
			}
		})
	}
}
