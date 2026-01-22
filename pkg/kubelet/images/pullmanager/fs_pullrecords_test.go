/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNewFSPullRecordsAccessor(t *testing.T) {
	tCtx := ktesting.Init(t)

	tests := []struct {
		name            string
		initRoot        bool
		initIntents     []string
		initPulled      []string
		modIntent       func(string) string
		modPulledRecord func(string) string
		wantErr         bool // TODO: do I ever want an error?
	}{
		{
			name:     "kubelet root dir does not exist",
			initRoot: false,
		},
		{
			name:     "kubelet root dir exists",
			initRoot: true,
		},
		{
			name:     "only alpha intents",
			initRoot: true,
			initIntents: []string{
				"sha256-6da3dced7eccc7ad0189517c31ec2235b50f730449d738d91525721f7e027fd4",
				"sha256-b6e1482bb6cb030924530b918e609fff30bbb07ab6845451d89deb823c3f72a9",
			},
			modIntent: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
		},
		{
			name:     "alpha and beta intents mixed",
			initRoot: true,
			initIntents: []string{
				"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56",
				"sha256-6da3dced7eccc7ad0189517c31ec2235b50f730449d738d91525721f7e027fd4",
				"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
				"sha256-b6e1482bb6cb030924530b918e609fff30bbb07ab6845451d89deb823c3f72a9",
			},
			modIntent: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
		},
		{
			name:     "only beta intents",
			initRoot: true,
			initIntents: []string{
				"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56",
				"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
				"sha256-f24acc752be18b93b0504c86312bbaf482c9efb0c45e925bbccb0a591cebd7af",
			},
		},
		{
			name:     "only alpha pulled records",
			initRoot: true,
			initPulled: []string{
				"sha256-7045bdd498b0e4b3963d50f906793ffbd68dc98686ec46c5b499a0a640a560b2",
				"sha256-dd1dd8dc6a329b7e1c2e43951302b23a047cb20fd3e10090e4ce7e8396035405",
			},
			modPulledRecord: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
		},
		{
			name:     "alpha and beta pulled records mixed",
			initRoot: true,
			initPulled: []string{
				"sha256-7045bdd498b0e4b3963d50f906793ffbd68dc98686ec46c5b499a0a640a560b2",
				"sha256-38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
				"sha256-dd1dd8dc6a329b7e1c2e43951302b23a047cb20fd3e10090e4ce7e8396035405",
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			modPulledRecord: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
		},
		{
			name:     "only beta pulled records",
			initRoot: true,
			initPulled: []string{
				"sha256-38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
		},
		{
			name:     "everything altogether",
			initRoot: true,
			initIntents: []string{
				"sha256-aef2af226629a35d5f3ef0fdbb29fdbebf038d0acd8850590e8c48e1e283aa56",
				"sha256-6da3dced7eccc7ad0189517c31ec2235b50f730449d738d91525721f7e027fd4",
				"sha256-ee81caca15454863449fb55a1d942904d56d5ed9f9b20a7cb3453944ea2c7e11",
				"sha256-b6e1482bb6cb030924530b918e609fff30bbb07ab6845451d89deb823c3f72a9",
			},
			modIntent: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
			initPulled: []string{
				"sha256-7045bdd498b0e4b3963d50f906793ffbd68dc98686ec46c5b499a0a640a560b2",
				"sha256-38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
				"sha256-dd1dd8dc6a329b7e1c2e43951302b23a047cb20fd3e10090e4ce7e8396035405",
				"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
				"sha256-f8778b6393eaf39315e767a58cbeacf2c4b270d94b4d6926ee993d9e49444991",
			},
			modPulledRecord: func(intent string) string {
				return strings.Replace(intent, "kubelet.config.k8s.io/v1alpha1", "kubelet.config.k8s.io/v1beta1", 1)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kubeletDir := t.TempDir()
			if !tt.initRoot {
				kubeletDir = filepath.Join(kubeletDir, "does-not-exist", "level2dir")
			}

			intentsDir := filepath.Join(kubeletDir, "image_manager", "pulling")
			pulledRecordsDir := filepath.Join(kubeletDir, "image_manager", "pulled")

			expectedIntents := teeTestData(t, intentsDir, "pulling", tt.initIntents)
			if tt.modIntent != nil {
				for i := range expectedIntents {
					expectedIntents[i] = tt.modIntent(expectedIntents[i])
				}
			}
			pulledRecordBytes := teeTestData(t, pulledRecordsDir, "pulled", tt.initPulled)
			if tt.modPulledRecord != nil {
				for i := range pulledRecordBytes {
					pulledRecordBytes[i] = tt.modPulledRecord(pulledRecordBytes[i])
				}
			}

			_, err := NewFSPullRecordsAccessor(tCtx.Logger(), kubeletDir)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewFSPullRecordsAccessor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			for _, dir := range []string{
				"image_manager",
				filepath.Join("image_manager", "pulling"),
				filepath.Join("image_manager", "pulled"),
			} {
				if info, err := os.Stat(filepath.Join(kubeletDir, dir)); err != nil {
					t.Errorf("error encountered accessing %q: %v", dir, err)
				} else if !info.IsDir() {
					t.Errorf("%q is not a directory", dir)
				}
			}

			for intent, expectedContent := range expectedIntents {
				filePath := filepath.Join(intentsDir, intent)
				testFileContentMatch(t, filePath, expectedContent)
			}

			for pulled, expectedContent := range pulledRecordBytes {
				filePath := filepath.Join(pulledRecordsDir, pulled)
				testFileContentMatch(t, filePath, expectedContent)
			}
		})
	}
}

func testFileContentMatch(t *testing.T, filePath string, expectedContent string) {
	t.Helper()
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read file %q: %v", filePath, err)
	}

	if diff := cmp.Diff(expectedContent, string(fileContent)); diff != "" {
		t.Errorf("file contents of %q do not match the expectation: %s", filePath, diff)
	}
}
