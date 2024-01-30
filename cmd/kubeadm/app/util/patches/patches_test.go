/*
Copyright 2020 The Kubernetes Authors.

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

package patches

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

var testKnownTargets = []string{
	"etcd",
	"kube-apiserver",
	"kube-controller-manager",
	"kube-scheduler",
	"kubeletconfiguration",
}

const testDirPattern = "patch-files"

func TestParseFilename(t *testing.T) {
	tests := []struct {
		name               string
		fileName           string
		expectedTargetName string
		expectedPatchType  types.PatchType
		expectedWarning    bool
		expectedError      bool
	}{
		{
			name:               "valid: known target and patch type",
			fileName:           "etcd+merge.json",
			expectedTargetName: "etcd",
			expectedPatchType:  types.MergePatchType,
		},
		{
			name:               "valid: known target and default patch type",
			fileName:           "etcd0.yaml",
			expectedTargetName: "etcd",
			expectedPatchType:  types.StrategicMergePatchType,
		},
		{
			name:               "valid: known target and custom patch type",
			fileName:           "etcd0+merge.yaml",
			expectedTargetName: "etcd",
			expectedPatchType:  types.MergePatchType,
		},
		{
			name:            "invalid: unknown target",
			fileName:        "foo.yaml",
			expectedWarning: true,
		},
		{
			name:            "invalid: unknown extension",
			fileName:        "etcd.foo",
			expectedWarning: true,
		},
		{
			name:            "invalid: missing extension",
			fileName:        "etcd",
			expectedWarning: true,
		},
		{
			name:          "invalid: unknown patch type",
			fileName:      "etcd+foo.json",
			expectedError: true,
		},
		{
			name:          "invalid: missing patch type",
			fileName:      "etcd+.json",
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			targetName, patchType, warn, err := parseFilename(tc.fileName, testKnownTargets)
			if (err != nil) != tc.expectedError {
				t.Errorf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if (warn != nil) != tc.expectedWarning {
				t.Errorf("expected warning: %v, got: %v, warning: %v", tc.expectedWarning, warn != nil, warn)
			}
			if targetName != tc.expectedTargetName {
				t.Errorf("expected target name: %v, got: %v", tc.expectedTargetName, targetName)
			}
			if patchType != tc.expectedPatchType {
				t.Errorf("expected patch type: %v, got: %v", tc.expectedPatchType, patchType)
			}
		})
	}
}

func TestCreatePatchSet(t *testing.T) {
	tests := []struct {
		name             string
		targetName       string
		patchType        types.PatchType
		expectedPatchSet *patchSet
		data             string
	}{
		{

			name:       "valid: YAML patches are separated and converted to JSON",
			targetName: "etcd",
			patchType:  types.StrategicMergePatchType,
			data:       "foo: bar\n---\nfoo: baz\n",
			expectedPatchSet: &patchSet{
				targetName: "etcd",
				patchType:  types.StrategicMergePatchType,
				patches:    []string{`{"foo":"bar"}`, `{"foo":"baz"}`},
			},
		},
		{
			name:       "valid: JSON patches are separated",
			targetName: "etcd",
			patchType:  types.StrategicMergePatchType,
			data:       `{"foo":"bar"}` + "\n---\n" + `{"foo":"baz"}`,
			expectedPatchSet: &patchSet{
				targetName: "etcd",
				patchType:  types.StrategicMergePatchType,
				patches:    []string{`{"foo":"bar"}`, `{"foo":"baz"}`},
			},
		},
		{
			name:       "valid: empty patches are ignored",
			targetName: "etcd",
			patchType:  types.StrategicMergePatchType,
			data:       `{"foo":"bar"}` + "\n---\n     ---\n" + `{"foo":"baz"}`,
			expectedPatchSet: &patchSet{
				targetName: "etcd",
				patchType:  types.StrategicMergePatchType,
				patches:    []string{`{"foo":"bar"}`, `{"foo":"baz"}`},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ps, _ := createPatchSet(tc.targetName, tc.patchType, tc.data)
			if !reflect.DeepEqual(ps, tc.expectedPatchSet) {
				t.Fatalf("expected patch set:\n%+v\ngot:\n%+v\n", tc.expectedPatchSet, ps)
			}
		})
	}
}

func TestGetPatchSetsForPathMustBeDirectory(t *testing.T) {
	tempFile, err := os.CreateTemp("", "test-file")
	if err != nil {
		t.Errorf("error creating temporary file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, tempFile)

	_, _, _, err = getPatchSetsFromPath(tempFile.Name(), testKnownTargets, io.Discard)
	var pathErr *os.PathError
	if !errors.As(err, &pathErr) {
		t.Fatalf("expected os.PathError for non-directory path %q, but got %v", tempFile.Name(), err)
	}
}

func TestGetPatchSetsForPath(t *testing.T) {
	const patchData = `{"foo":"bar"}`

	tests := []struct {
		name                 string
		filesToWrite         []string
		expectedPatchSets    []*patchSet
		expectedPatchFiles   []string
		expectedIgnoredFiles []string
		expectedError        bool
		patchData            string
	}{
		{
			name:         "valid: patch files are sorted and non-patch files are ignored",
			filesToWrite: []string{"kube-scheduler+merge.json", "kube-apiserver+json.yaml", "etcd.yaml", "foo", "bar.json"},
			patchData:    patchData,
			expectedPatchSets: []*patchSet{
				{
					targetName: "etcd",
					patchType:  types.StrategicMergePatchType,
					patches:    []string{patchData},
				},
				{
					targetName: "kube-apiserver",
					patchType:  types.JSONPatchType,
					patches:    []string{patchData},
				},
				{
					targetName: "kube-scheduler",
					patchType:  types.MergePatchType,
					patches:    []string{patchData},
				},
			},
			expectedPatchFiles:   []string{"etcd.yaml", "kube-apiserver+json.yaml", "kube-scheduler+merge.json"},
			expectedIgnoredFiles: []string{"bar.json", "foo"},
		},
		{
			name:                 "valid: empty files are ignored",
			patchData:            "",
			filesToWrite:         []string{"kube-scheduler.json"},
			expectedPatchFiles:   []string{},
			expectedIgnoredFiles: []string{"kube-scheduler.json"},
			expectedPatchSets:    []*patchSet{},
		},
		{
			name:          "invalid: bad patch type in filename returns and error",
			filesToWrite:  []string{"kube-scheduler+foo.json"},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tempDir, err := os.MkdirTemp("", testDirPattern)
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tempDir)

			for _, file := range tc.filesToWrite {
				filePath := filepath.Join(tempDir, file)
				err := os.WriteFile(filePath, []byte(tc.patchData), 0644)
				if err != nil {
					t.Fatalf("could not write temporary file %q", filePath)
				}
			}

			patchSets, patchFiles, ignoredFiles, err := getPatchSetsFromPath(tempDir, testKnownTargets, io.Discard)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}

			if !reflect.DeepEqual(tc.expectedPatchFiles, patchFiles) {
				t.Fatalf("expected patch files:\n%+v\ngot:\n%+v", tc.expectedPatchFiles, patchFiles)
			}
			if !reflect.DeepEqual(tc.expectedIgnoredFiles, ignoredFiles) {
				t.Fatalf("expected ignored files:\n%+v\ngot:\n%+v", tc.expectedIgnoredFiles, ignoredFiles)
			}
			if !reflect.DeepEqual(tc.expectedPatchSets, patchSets) {
				t.Fatalf("expected patch sets:\n%+v\ngot:\n%+v", tc.expectedPatchSets, patchSets)
			}
		})
	}
}

func TestGetPatchManagerForPath(t *testing.T) {
	type file struct {
		name string
		data string
	}

	tests := []struct {
		name          string
		files         []*file
		patchTarget   *PatchTarget
		expectedData  []byte
		expectedError bool
	}{
		{
			name: "valid: patch a kube-apiserver target using merge patch; json patch is applied first",
			patchTarget: &PatchTarget{
				Name:                      "kube-apiserver",
				StrategicMergePatchObject: v1.Pod{},
				Data:                      []byte("foo: bar\nbaz: qux\n"),
			},
			expectedData: []byte(`{"baz":"qux","foo":"patched"}`),
			files: []*file{
				{
					name: "kube-apiserver+merge.yaml",
					data: "foo: patched",
				},
				{
					name: "kube-apiserver+json.json",
					data: `[{"op": "replace", "path": "/foo", "value": "zzz"}]`,
				},
			},
		},
		{
			name: "valid: kube-apiserver target is patched with json patch",
			patchTarget: &PatchTarget{
				Name:                      "kube-apiserver",
				StrategicMergePatchObject: v1.Pod{},
				Data:                      []byte("foo: bar\n"),
			},
			expectedData: []byte(`{"foo":"zzz"}`),
			files: []*file{
				{
					name: "kube-apiserver+json.json",
					data: `[{"op": "replace", "path": "/foo", "value": "zzz"}]`,
				},
			},
		},
		{
			name: "valid: kubeletconfiguration target is patched with json patch",
			patchTarget: &PatchTarget{
				Name:                      "kubeletconfiguration",
				StrategicMergePatchObject: nil,
				Data:                      []byte("foo: bar\n"),
			},
			expectedData: []byte(`{"foo":"zzz"}`),
			files: []*file{
				{
					name: "kubeletconfiguration+json.json",
					data: `[{"op": "replace", "path": "/foo", "value": "zzz"}]`,
				},
			},
		},
		{
			name: "valid: kube-apiserver target is patched with strategic merge patch",
			patchTarget: &PatchTarget{
				Name:                      "kube-apiserver",
				StrategicMergePatchObject: v1.Pod{},
				Data:                      []byte("foo: bar\n"),
			},
			expectedData: []byte(`{"foo":"zzz"}`),
			files: []*file{
				{
					name: "kube-apiserver+strategic.json",
					data: `{"foo":"zzz"}`,
				},
			},
		},
		{
			name: "valid: etcd target is not changed because there are no patches for it",
			patchTarget: &PatchTarget{
				Name:                      "etcd",
				StrategicMergePatchObject: v1.Pod{},
				Data:                      []byte("foo: bar\n"),
			},
			expectedData: []byte("foo: bar\n"),
			files: []*file{
				{
					name: "kube-apiserver+merge.yaml",
					data: "foo: patched",
				},
			},
		},
		{
			name: "invalid: cannot patch etcd target due to malformed json patch",
			patchTarget: &PatchTarget{
				Name:                      "etcd",
				StrategicMergePatchObject: v1.Pod{},
				Data:                      []byte("foo: bar\n"),
			},
			files: []*file{
				{
					name: "etcd+json.json",
					data: `{"foo":"zzz"}`,
				},
			},
			expectedError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tempDir, err := os.MkdirTemp("", testDirPattern)
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tempDir)

			for _, file := range tc.files {
				filePath := filepath.Join(tempDir, file.name)
				err := os.WriteFile(filePath, []byte(file.data), 0644)
				if err != nil {
					t.Fatalf("could not write temporary file %q", filePath)
				}
			}

			pm, err := GetPatchManagerForPath(tempDir, testKnownTargets, nil)
			if err != nil {
				t.Fatal(err)
			}

			err = pm.ApplyPatchesToTarget(tc.patchTarget)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", tc.expectedError, err != nil, err)
			}
			if err != nil {
				return
			}

			if !bytes.Equal(tc.patchTarget.Data, tc.expectedData) {
				t.Fatalf("expected result:\n%s\ngot:\n%s", tc.expectedData, tc.patchTarget.Data)
			}
		})
	}
}

func TestGetPatchManagerForPathCache(t *testing.T) {
	tempDir, err := os.MkdirTemp("", testDirPattern)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	pmOld, err := GetPatchManagerForPath(tempDir, testKnownTargets, nil)
	if err != nil {
		t.Fatal(err)
	}
	pmNew, err := GetPatchManagerForPath(tempDir, testKnownTargets, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pmOld != pmNew {
		t.Logf("path %q was not cached, expected pointer: %p, got: %p", tempDir, pmOld, pmNew)
	}
}
