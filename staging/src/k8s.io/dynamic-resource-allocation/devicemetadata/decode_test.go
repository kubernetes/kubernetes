/*
Copyright The Kubernetes Authors.

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

package devicemetadata

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/api/metadata/v1alpha1"
	"k8s.io/utils/ptr"
)

var validV1Alpha1JSON = mustMarshal(&v1alpha1.DeviceMetadata{
	TypeMeta: metav1.TypeMeta{
		APIVersion: v1alpha1.SchemeGroupVersion.String(),
		Kind:       "DeviceMetadata",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name:       "my-claim",
		Namespace:  "default",
		UID:        "uid-1234",
		Generation: 1,
	},
	Requests: []v1alpha1.DeviceMetadataRequest{{
		Name: "gpu",
		Devices: []v1alpha1.Device{{
			Driver: "gpu.example.com",
			Pool:   "worker-0",
			Name:   "gpu-0",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
			},
		}},
	}},
})

func mustMarshal(obj interface{}) []byte {
	data, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	return append(data, '\n')
}

func TestDecodeMetadataFromStream(t *testing.T) {
	unknownVersionJSON := `{"apiVersion":"metadata.k8s.io/v99","kind":"DeviceMetadata","metadata":{"name":"test"}}` + "\n"
	unknownV2JSON := `{"apiVersion":"metadata.k8s.io/v100","kind":"DeviceMetadata","metadata":{"name":"test"}}` + "\n"
	missingKindJSON := `{"apiVersion":"metadata.resource.k8s.io/v1alpha1","metadata":{"name":"test"}}` + "\n"
	missingApiversionJSON := `{"kind":"DeviceMetadata","metadata":{"name":"test"}}` + "\n"

	expectedV1Alpha1 := &v1alpha1.DeviceMetadata{
		TypeMeta: metav1.TypeMeta{
			APIVersion: v1alpha1.SchemeGroupVersion.String(),
			Kind:       "DeviceMetadata",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-claim",
			Namespace:  "default",
			UID:        "uid-1234",
			Generation: 1,
		},
		Requests: []v1alpha1.DeviceMetadataRequest{{
			Name: "gpu",
			Devices: []v1alpha1.Device{{
				Driver: "gpu.example.com",
				Pool:   "worker-0",
				Name:   "gpu-0",
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
				},
			}},
		}},
	}

	expectedInternal := &metadata.DeviceMetadata{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-claim",
			Namespace:  "default",
			UID:        "uid-1234",
			Generation: 1,
		},
		Requests: []metadata.DeviceMetadataRequest{{
			Name: "gpu",
			Devices: []metadata.Device{{
				Driver: "gpu.example.com",
				Pool:   "worker-0",
				Name:   "gpu-0",
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
				},
			}},
		}},
	}

	testcases := map[string]struct {
		streamInput []byte
		dest        runtime.Object
		expected    runtime.Object
		expectError string
	}{
		"decode-to-v1alpha1": {
			streamInput: validV1Alpha1JSON,
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},
		"decode-to-internal": {
			streamInput: validV1Alpha1JSON,
			dest:        &metadata.DeviceMetadata{},
			expected:    expectedInternal,
		},
		"empty-stream": {
			streamInput: nil,
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no metadata objects found in stream",
		},
		"invalid-json": {
			streamInput: []byte("{not-json"),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "read metadata object from stream",
		},
		"truncated-json": {
			streamInput: []byte(`{"apiVersion":"metadata.resource.k8s.io/v1alpha1","kind":"DeviceMeta`),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "read metadata object from stream",
		},
		"missing-kind": {
			streamInput: []byte(missingKindJSON),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no compatible metadata version found in stream",
		},
		"missing-apiversion": {
			streamInput: []byte(missingApiversionJSON),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no compatible metadata version found in stream",
		},
		"only-unknown-versions": {
			streamInput: []byte(unknownVersionJSON),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no compatible metadata version found in stream",
		},
		"multiple-unknown-versions": {
			streamInput: append([]byte(unknownVersionJSON), []byte(unknownV2JSON)...),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no compatible metadata version found in stream",
		},
		"unknown-version-then-broken": {
			streamInput: append([]byte(unknownVersionJSON), []byte(missingKindJSON)...),
			dest:        &v1alpha1.DeviceMetadata{},
			expectError: "no compatible metadata version found in stream",
		},
		"known-version-then-broken": {
			streamInput: append(validV1Alpha1JSON, []byte("{broken")...),
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},

		// Forward compatibility: object-level errors are skipped so
		// that an older consumer can reach a version it understands.
		"skips-unknown-version": {
			streamInput: append([]byte(unknownVersionJSON), validV1Alpha1JSON...),
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},
		"skips-missing-kind": {
			streamInput: append([]byte(missingKindJSON), validV1Alpha1JSON...),
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},
		"skips-missing-apiversion": {
			streamInput: append([]byte(missingApiversionJSON), validV1Alpha1JSON...),
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},
		"skips-multiple-errors": {
			streamInput: append(append([]byte(unknownVersionJSON), []byte(missingKindJSON)...), validV1Alpha1JSON...),
			dest:        &v1alpha1.DeviceMetadata{},
			expected:    expectedV1Alpha1,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			err := DecodeMetadataFromStream(json.NewDecoder(bytes.NewReader(tc.streamInput)), tc.dest)

			if tc.expectError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectError)
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got: %v", tc.expectError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeMetadataFromStream: %v", err)
			}

			if diff := cmp.Diff(tc.expected, tc.dest); diff != "" {
				t.Errorf("metadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReadMetadata(t *testing.T) {
	dir := t.TempDir()
	validFile := filepath.Join(dir, "valid", "metadata.json")
	if err := os.MkdirAll(filepath.Dir(validFile), 0755); err != nil {
		t.Fatalf("create test dir: %v", err)
	}
	if err := os.WriteFile(validFile, validV1Alpha1JSON, 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	emptyObjectFile := filepath.Join(dir, "empty-object", "metadata.json")
	if err := os.MkdirAll(filepath.Dir(emptyObjectFile), 0755); err != nil {
		t.Fatalf("create test dir: %v", err)
	}
	if err := os.WriteFile(emptyObjectFile, []byte("{}"), 0644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	expectedInternal := &metadata.DeviceMetadata{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "my-claim",
			Namespace:  "default",
			UID:        "uid-1234",
			Generation: 1,
		},
		Requests: []metadata.DeviceMetadataRequest{{
			Name: "gpu",
			Devices: []metadata.Device{{
				Driver: "gpu.example.com",
				Pool:   "worker-0",
				Name:   "gpu-0",
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
				},
			}},
		}},
	}

	testcases := map[string]struct {
		path        string
		expected    *metadata.DeviceMetadata
		expectError string
	}{
		"valid-file": {
			path:     validFile,
			expected: expectedInternal,
		},
		"missing-file": {
			path:        "/nonexistent/path/metadata.json",
			expectError: "open metadata file",
		},
		"error-includes-path": {
			path:        emptyObjectFile,
			expectError: emptyObjectFile,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			dm, err := readMetadata(tc.path)

			if tc.expectError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectError)
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got: %v", tc.expectError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("readMetadata: %v", err)
			}

			if diff := cmp.Diff(tc.expected, dm); diff != "" {
				t.Errorf("metadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReadRequestDir(t *testing.T) {
	dir := t.TempDir()

	gpuInternal := metadata.DeviceMetadataRequest{
		Name: "gpu",
		Devices: []metadata.Device{{
			Driver: "gpu.example.com",
			Pool:   "worker-0",
			Name:   "gpu-0",
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
			},
		}},
	}
	nicInternal := metadata.DeviceMetadataRequest{
		Name: "nic",
		Devices: []metadata.Device{{
			Driver: "nic.example.com",
			Pool:   "worker-0",
			Name:   "nic-0",
		}},
	}

	gpuJSON := mustMarshal(&v1alpha1.DeviceMetadata{
		TypeMeta:   metav1.TypeMeta{APIVersion: v1alpha1.SchemeGroupVersion.String(), Kind: "DeviceMetadata"},
		ObjectMeta: metav1.ObjectMeta{Name: "my-claim", Namespace: "default", UID: "uid-1234", Generation: 1},
		Requests: []v1alpha1.DeviceMetadataRequest{{
			Name: "gpu",
			Devices: []v1alpha1.Device{{
				Driver: "gpu.example.com",
				Pool:   "worker-0",
				Name:   "gpu-0",
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"model": {StringValue: ptr.To("LATEST-GPU-MODEL")}, //nolint:modernize
				},
			}},
		}},
	})
	nicJSON := mustMarshal(&v1alpha1.DeviceMetadata{
		TypeMeta:   metav1.TypeMeta{APIVersion: v1alpha1.SchemeGroupVersion.String(), Kind: "DeviceMetadata"},
		ObjectMeta: metav1.ObjectMeta{Name: "my-claim", Namespace: "default", UID: "uid-1234", Generation: 1},
		Requests: []v1alpha1.DeviceMetadataRequest{{
			Name:    "nic",
			Devices: []v1alpha1.Device{{Driver: "nic.example.com", Pool: "worker-0", Name: "nic-0"}},
		}},
	})
	podClaimJSON := mustMarshal(&v1alpha1.DeviceMetadata{
		TypeMeta:     metav1.TypeMeta{APIVersion: v1alpha1.SchemeGroupVersion.String(), Kind: "DeviceMetadata"},
		ObjectMeta:   metav1.ObjectMeta{Name: "my-claim", Namespace: "default", UID: "uid-1234", Generation: 1},
		PodClaimName: ptr.To("my-gpu"), //nolint:modernize
		Requests: []v1alpha1.DeviceMetadataRequest{{
			Name:    "gpu",
			Devices: []v1alpha1.Device{{Driver: "gpu.example.com", Pool: "worker-0", Name: "gpu-0"}},
		}},
	})

	writeFile := func(subdir, name string, data []byte) string {
		p := filepath.Join(dir, subdir)
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", p, err)
		}
		if err := os.WriteFile(filepath.Join(p, name), data, 0644); err != nil {
			t.Fatalf("write %s/%s: %v", p, name, err)
		}
		return p
	}

	singleDir := writeFile("single", "gpu.example.com"+metadata.MetadataFileSuffix, gpuJSON)
	multiDir := writeFile("multi", "gpu.example.com"+metadata.MetadataFileSuffix, gpuJSON)
	writeFile("multi", "nic.example.com"+metadata.MetadataFileSuffix, nicJSON)
	emptyDir := writeFile("empty", "unrelated.txt", []byte("not metadata"))
	mixedDir := writeFile("mixed", "gpu.example.com"+metadata.MetadataFileSuffix, gpuJSON)
	writeFile("mixed", "unrelated.json", []byte(`{"foo":"bar"}`))
	podClaimDir := writeFile("podclaim", "gpu.example.com"+metadata.MetadataFileSuffix, podClaimJSON)

	testcases := map[string]struct {
		dir         string
		expected    *metadata.DeviceMetadata
		expectError string
	}{
		"single-driver": {
			dir: singleDir,
			expected: &metadata.DeviceMetadata{
				Requests: []metadata.DeviceMetadataRequest{gpuInternal},
			},
		},
		"multiple-drivers": {
			dir: multiDir,
			expected: &metadata.DeviceMetadata{
				Requests: []metadata.DeviceMetadataRequest{gpuInternal, nicInternal},
			},
		},
		"empty-directory": {
			dir:         emptyDir,
			expectError: "no metadata files found",
		},
		"nonexistent-directory": {
			dir:         "/nonexistent/path",
			expectError: "no metadata files found",
		},
		"ignores-non-matching-files": {
			dir: mixedDir,
			expected: &metadata.DeviceMetadata{
				Requests: []metadata.DeviceMetadataRequest{gpuInternal},
			},
		},
		"preserves-pod-claim-name": {
			dir: podClaimDir,
			expected: &metadata.DeviceMetadata{
				PodClaimName: ptr.To("my-gpu"), //nolint:modernize
				Requests: []metadata.DeviceMetadataRequest{{
					Name:    "gpu",
					Devices: []metadata.Device{{Driver: "gpu.example.com", Pool: "worker-0", Name: "gpu-0"}},
				}},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			dm, err := readRequestDir(tc.dir)

			if tc.expectError != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectError)
				}
				if !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got: %v", tc.expectError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("readRequestDir: %v", err)
			}

			if diff := cmp.Diff(tc.expected, dm); diff != "" {
				t.Errorf("metadata mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
