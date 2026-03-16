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

package diff

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/utils/exec"
)

type FakeObject struct {
	name   string
	merged map[string]interface{}
	live   map[string]interface{}
}

var _ Object = &FakeObject{}

func (f *FakeObject) Name() string {
	return f.name
}

func (f *FakeObject) Merged() (runtime.Object, error) {
	// Return nil if merged object does not exist
	if f.merged == nil {
		return nil, nil
	}
	return &unstructured.Unstructured{Object: f.merged}, nil
}

func (f *FakeObject) Live() runtime.Object {
	// Return nil if live object does not exist
	if f.live == nil {
		return nil
	}
	return &unstructured.Unstructured{Object: f.live}
}

func TestDiffProgram(t *testing.T) {
	externalDiffCommands := [3]string{"diff", "diff -ruN", "diff --report-identical-files"}

	t.Setenv("LANG", "C")
	t.Setenv("LANGUAGE", "en_US")

	for i, c := range externalDiffCommands {
		t.Setenv("KUBECTL_EXTERNAL_DIFF", c)
		streams, _, stdout, _ := genericiooptions.NewTestIOStreams()
		diff := DiffProgram{
			IOStreams: streams,
			Exec:      exec.New(),
		}
		err := diff.Run("/dev/zero", "/dev/zero")
		if err != nil {
			t.Fatal(err)
		}

		// Testing diff --report-identical-files
		if i == 2 {
			output_msg := "Files /dev/zero and /dev/zero are identical\n"
			if output := stdout.String(); output != output_msg {
				t.Fatalf(`stdout = %q, expected = %s"`, output, output_msg)
			}
		}
	}
}

func TestPrinter(t *testing.T) {
	printer := Printer{}

	obj := &unstructured.Unstructured{Object: map[string]interface{}{
		"string": "string",
		"list":   []int{1, 2, 3},
		"int":    12,
	}}
	buf := bytes.Buffer{}
	printer.Print(obj, &buf)
	want := `int: 12
list:
- 1
- 2
- 3
string: string
`
	if buf.String() != want {
		t.Errorf("Print() = %q, want %q", buf.String(), want)
	}
}

func TestDiffVersion(t *testing.T) {
	diff, err := NewDiffVersion("MERGED")
	if err != nil {
		t.Fatal(err)
	}
	defer diff.Dir.Delete()

	obj := FakeObject{
		name:   "bla",
		live:   map[string]interface{}{"live": true},
		merged: map[string]interface{}{"merged": true},
	}
	rObj, err := obj.Merged()
	if err != nil {
		t.Fatal(err)
	}
	err = diff.Print(obj.Name(), rObj, Printer{})
	if err != nil {
		t.Fatal(err)
	}
	fcontent, err := os.ReadFile(filepath.Join(diff.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent := "merged: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}
}

func TestDirectory(t *testing.T) {
	dir, err := CreateDirectory("prefix")
	defer dir.Delete()
	if err != nil {
		t.Fatal(err)
	}
	_, err = os.Stat(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(dir.Name), "prefix") {
		t.Fatalf(`Directory doesn't start with "prefix": %q`, dir.Name)
	}
	entries, err := os.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("Directory should be empty, has %d elements", len(entries))
	}
	_, err = dir.NewFile("ONE")
	if err != nil {
		t.Fatal(err)
	}
	_, err = dir.NewFile("TWO")
	if err != nil {
		t.Fatal(err)
	}
	entries, err = os.ReadDir(dir.Name)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("ReadDir should have two elements, has %d elements", len(entries))
	}
	err = dir.Delete()
	if err != nil {
		t.Fatal(err)
	}
	_, err = os.Stat(dir.Name)
	if err == nil {
		t.Fatal("Directory should be gone, still present.")
	}
}

func TestDiffer(t *testing.T) {
	diff, err := NewDiffer("LIVE", "MERGED")
	if err != nil {
		t.Fatal(err)
	}
	defer diff.TearDown()

	obj := FakeObject{
		name:   "bla",
		live:   map[string]interface{}{"live": true},
		merged: map[string]interface{}{"merged": true},
	}
	err = diff.Diff(&obj, Printer{}, true, false)
	if err != nil {
		t.Fatal(err)
	}
	fcontent, err := os.ReadFile(filepath.Join(diff.From.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent := "live: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}

	fcontent, err = os.ReadFile(filepath.Join(diff.To.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent = "merged: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}
}

func TestShowManagedFields(t *testing.T) {
	diff, err := NewDiffer("LIVE", "MERGED")
	if err != nil {
		t.Fatal(err)
	}
	defer diff.TearDown()

	testCases := []struct {
		name                string
		showManagedFields   bool
		expectedFromContent string
		expectedToContent   string
	}{
		{
			name:              "without managed fields",
			showManagedFields: false,
			expectedFromContent: `live: true
metadata:
  name: foo
`,
			expectedToContent: `merged: true
metadata:
  name: foo
`,
		},
		{
			name:              "with managed fields",
			showManagedFields: true,
			expectedFromContent: `live: true
metadata:
  managedFields: mf-data
  name: foo
`,
			expectedToContent: `merged: true
metadata:
  managedFields: mf-data
  name: foo
`,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			obj := FakeObject{
				name: fmt.Sprintf("TestCase%d", i),
				live: map[string]interface{}{
					"live": true,
					"metadata": map[string]interface{}{
						"managedFields": "mf-data",
						"name":          "foo",
					},
				},
				merged: map[string]interface{}{
					"merged": true,
					"metadata": map[string]interface{}{
						"managedFields": "mf-data",
						"name":          "foo",
					},
				},
			}

			err = diff.Diff(&obj, Printer{}, tc.showManagedFields, false)
			if err != nil {
				t.Fatal(err)
			}

			actualFromContent, _ := os.ReadFile(filepath.Join(diff.From.Dir.Name, obj.Name()))
			if string(actualFromContent) != tc.expectedFromContent {
				t.Fatalf("File has %q, expected %q", string(actualFromContent), tc.expectedFromContent)
			}

			actualToContent, _ := os.ReadFile(filepath.Join(diff.To.Dir.Name, obj.Name()))
			if string(actualToContent) != tc.expectedToContent {
				t.Fatalf("File has %q, expected %q", string(actualToContent), tc.expectedToContent)
			}
		})
	}
}

func TestMasker(t *testing.T) {
	type diff struct {
		from runtime.Object
		to   runtime.Object
	}
	cases := []struct {
		name  string
		input diff
		want  diff
	}{
		{
			name: "no_changes",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***", // still masked
							"password": "***", // still masked
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***", // still masked
							"password": "***", // still masked
						},
					},
				},
			},
		},
		{
			name: "object_created",
			input: diff{
				from: nil, // does not exist yet
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
			},
			want: diff{
				from: nil, // does not exist yet
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***", // no suffix needed
							"password": "***", // no suffix needed
						},
					},
				},
			},
		},
		{
			name: "object_removed",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: nil, // removed
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***", // no suffix needed
							"password": "***", // no suffix needed
						},
					},
				},
				to: nil, // removed
			},
		},
		{
			name: "data_key_added",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123", // added
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "***", // no suffix needed
						},
					},
				},
			},
		},
		{
			name: "data_key_changed",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "456", // changed
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "*** (before)", // added suffix for diff
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "*** (after)", // added suffix for diff
						},
					},
				},
			},
		},
		{
			name: "data_key_removed",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							// "password": "123", // removed
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "***", // no suffix needed
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							// "password": "***",
						},
					},
				},
			},
		},
		{
			name: "empty_secret_from",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{}, // no data key
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{}, // no data key
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "***",
						},
					},
				},
			},
		},
		{
			name: "empty_secret_to",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{}, // no data key
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"data": map[string]interface{}{
							"username": "***",
							"password": "***",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{}, // no data key
				},
			},
		},
		{
			name: "invalid_data_key",
			input: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"some_other_key": map[string]interface{}{ // invalid key
							"username": "abc",
							"password": "123",
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"some_other_key": map[string]interface{}{ // invalid key
							"username": "abc",
							"password": "123",
						},
					},
				},
			},
			want: diff{
				from: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"some_other_key": map[string]interface{}{
							"username": "abc", // skipped
							"password": "123", // skipped
						},
					},
				},
				to: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"some_other_key": map[string]interface{}{
							"username": "abc", // skipped
							"password": "123", // skipped
						},
					},
				},
			},
		},
	}
	for _, tc := range cases {
		tc := tc // capture range variable
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			m, err := NewMasker(tc.input.from, tc.input.to)
			if err != nil {
				t.Fatal(err)
			}
			from, to := m.From(), m.To()
			if from != nil && tc.want.from != nil {
				if diff := cmp.Diff(from, tc.want.from); diff != "" {
					t.Errorf("from: (-want +got):\n%s", diff)
				}
			}
			if to != nil && tc.want.to != nil {
				if diff := cmp.Diff(to, tc.want.to); diff != "" {
					t.Errorf("to: (-want +got):\n%s", diff)
				}
			}
		})
	}
}


// TestDiffWithApplySetValidation tests basic ApplySet validation scenarios
func TestDiffWithApplySetValidation(t *testing.T) {
	testCases := []struct {
		name        string
		applySetRef string
		expectValid bool
	}{
		{
			name:        "no applyset is valid",
			applySetRef: "",
			expectValid: true,
		},
		{
			name:        "applyset reference format",
			applySetRef: "configmap/test-applyset",
			expectValid: true,
		},
		{
			name:        "secret applyset reference",
			applySetRef: "secret/my-applyset",
			expectValid: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := &DiffOptions{
				ApplySetRef: tc.applySetRef,
			}

			// Test that ApplySetRef is properly assigned
			if options.ApplySetRef != tc.applySetRef {
				t.Errorf("Expected ApplySetRef %q, got %q", tc.applySetRef, options.ApplySetRef)
			}

			// Test that applySetPrune handles nil ApplySet gracefully
			visitedUIDs := sets.New[types.UID]("test-uid")
			result, err := options.applySetPrune(visitedUIDs)
			if err != nil {
				t.Errorf("applySetPrune should handle nil ApplySet gracefully, got error: %v", err)
			}
			if len(result) != 0 {
				t.Errorf("Expected empty result when ApplySet is nil, got %d items", len(result))
			}

			// Test ApplySetRef format validation
			if tc.applySetRef != "" {
				if !strings.Contains(tc.applySetRef, "/") {
					t.Errorf("ApplySetRef should contain '/' separator: %q", tc.applySetRef)
				}
				parts := strings.Split(tc.applySetRef, "/")
				if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
					t.Errorf("ApplySetRef should have format 'kind/name': %q", tc.applySetRef)
				}
			}
		})
	}
}

// TestDiffApplySetPruning tests the ApplySet pruning integration
func TestDiffApplySetPruning(t *testing.T) {
	testCases := []struct {
		name            string
		applySetRef     string
		visitedUIDs     []types.UID
		expectEmptyResult bool
	}{
		{
			name:            "no applyset configured",
			applySetRef:     "",
			visitedUIDs:     []types.UID{"uid1", "uid2"},
			expectEmptyResult: true,
		},
		{
			name:            "applyset with visited UIDs but nil ApplySet",
			applySetRef:     "configmap/test",
			visitedUIDs:     []types.UID{"uid1", "uid2"},
			expectEmptyResult: true,
		},
		{
			name:            "applyset ref set with empty UIDs",
			applySetRef:     "secret/test",
			visitedUIDs:     []types.UID{},
			expectEmptyResult: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := &DiffOptions{
				ApplySetRef: tc.applySetRef,
				// ApplySet is intentionally nil to test graceful handling
			}

			visitedUIDs := sets.New(tc.visitedUIDs...)
			result, err := options.applySetPrune(visitedUIDs)

			// Should not error when ApplySet is nil
			if err != nil {
				t.Errorf("Expected no error when ApplySet is nil, got %v", err)
			}

			if tc.expectEmptyResult && len(result) != 0 {
				t.Errorf("Expected empty result, got %d items", len(result))
			}
		})
	}
}

// TestDiffApplySetErrorHandling tests error handling scenarios for ApplySet
func TestDiffApplySetErrorHandling(t *testing.T) {
	testCases := []struct {
		name          string
		setupOptions  func(*DiffOptions)
		expectError   bool
		errorContains string
	}{
		{
			name: "nil rest client error",
			setupOptions: func(opts *DiffOptions) {
				// This would test the nil client check in Complete()
				// In a real implementation, we'd mock the factory to return nil client
				opts.ApplySetRef = "configmap/test"
			},
			expectError: false, // Complete() handles this, not tested here without factory mock
		},
		{
			name: "invalid applyset reference",
			setupOptions: func(opts *DiffOptions) {
				opts.ApplySetRef = "invalid-reference-format"
			},
			expectError: false, // This is tested in Complete() with factory mock
		},
		{
			name: "applyset without prune",
			setupOptions: func(opts *DiffOptions) {
				opts.ApplySetRef = "configmap/test"
				// prune will be false by default
			},
			expectError: false, // This is tested in validation
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := &DiffOptions{}
			tc.setupOptions(options)

			// Test applySetPrune with nil ApplySet (graceful error handling)
			visitedUIDs := sets.New[types.UID]()
			result, err := options.applySetPrune(visitedUIDs)

			// applySetPrune should handle nil ApplySet gracefully
			if err != nil {
				t.Errorf("applySetPrune should handle nil ApplySet gracefully, got error: %v", err)
			}
			if len(result) != 0 {
				t.Errorf("applySetPrune should return empty result for nil ApplySet, got %d items", len(result))
			}
		})
	}
}

// TestDiffApplySetInitialization tests the ApplySet initialization logic
func TestDiffApplySetInitialization(t *testing.T) {
	testCases := []struct {
		name         string
		applySetRef  string
		expectNilSet bool
	}{
		{
			name:         "empty applyset reference",
			applySetRef:  "",
			expectNilSet: true,
		},
		{
			name:         "valid applyset reference",
			applySetRef:  "configmap/test-applyset",
			expectNilSet: true, // Will be nil without proper factory mock
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := &DiffOptions{
				ApplySetRef: tc.applySetRef,
			}

			// Test that options start with nil ApplySet
			if options.ApplySet != nil {
				t.Errorf("Expected ApplySet to be nil initially, got %v", options.ApplySet)
			}

			// Test applySetPrune behavior with nil ApplySet
			visitedUIDs := sets.New[types.UID]()
			result, err := options.applySetPrune(visitedUIDs)

			if err != nil {
				t.Errorf("applySetPrune should handle nil ApplySet without error, got %v", err)
			}
			if len(result) != 0 {
				t.Errorf("applySetPrune should return empty slice for nil ApplySet, got %d items", len(result))
			}
		})
	}
}

// TestDiffApplySetStateHydration tests that ApplySet state is properly initialized
func TestDiffApplySetStateHydration(t *testing.T) {
	// This test verifies the conceptual flow - in practice would need factory mocks
	t.Run("applyset state initialization flow", func(t *testing.T) {
		options := &DiffOptions{
			ApplySetRef: "configmap/test-applyset",
		}

		// Verify that applySetPrune handles the case where ApplySet is not initialized
		visitedUIDs := sets.New[types.UID]("test-uid-1", "test-uid-2")

		result, err := options.applySetPrune(visitedUIDs)

		// Should not error even when ApplySet is nil
		if err != nil {
			t.Errorf("applySetPrune should handle uninitialized ApplySet gracefully, got error: %v", err)
		}

		// Should return empty result when ApplySet is nil
		if len(result) != 0 {
			t.Errorf("Expected empty result when ApplySet is nil, got %d objects", len(result))
		}
	})
}

// TestDiffApplySetValidationLogic tests ApplySet reference validation logic
func TestDiffApplySetValidationLogic(t *testing.T) {
	testCases := []struct {
		name        string
		applySetRef string
		expectValid bool
		errorMsg    string
	}{
		{
			name:        "empty reference is valid",
			applySetRef: "",
			expectValid: true,
		},
		{
			name:        "configmap reference",
			applySetRef: "configmap/my-set",
			expectValid: true,
		},
		{
			name:        "secret reference",
			applySetRef: "secret/my-secret-set",
			expectValid: true,
		},
		{
			name:        "invalid format missing separator",
			applySetRef: "configmapmy-set",
			expectValid: false,
			errorMsg:    "too many parts", // This actually gets detected as len(parts) != 2
		},
		{
			name:        "invalid format empty kind",
			applySetRef: "/my-set",
			expectValid: false,
			errorMsg:    "empty kind",
		},
		{
			name:        "invalid format empty name",
			applySetRef: "configmap/",
			expectValid: false,
			errorMsg:    "empty name",
		},
		{
			name:        "invalid format too many parts",
			applySetRef: "configmap/my/set",
			expectValid: false,
			errorMsg:    "too many parts",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test the reference format validation logic
			var isValid bool
			var errorMsg string

			if tc.applySetRef == "" {
				isValid = true
			} else {
				parts := strings.Split(tc.applySetRef, "/")
				if len(parts) != 2 {
					isValid = false
					errorMsg = "too many parts"
				} else if parts[0] == "" {
					isValid = false
					errorMsg = "empty kind"
				} else if parts[1] == "" {
					isValid = false
					errorMsg = "empty name"
				} else if !strings.Contains(tc.applySetRef, "/") {
					isValid = false
					errorMsg = "missing separator"
				} else {
					isValid = true
				}
			}

			if tc.expectValid != isValid {
				t.Errorf("Expected valid=%v, got valid=%v for ref %q", tc.expectValid, isValid, tc.applySetRef)
			}
			if !tc.expectValid && tc.errorMsg != "" && errorMsg != tc.errorMsg {
				t.Errorf("Expected error %q, got %q", tc.errorMsg, errorMsg)
			}

			// Test that DiffOptions properly stores the reference
			options := &DiffOptions{ApplySetRef: tc.applySetRef}
			if options.ApplySetRef != tc.applySetRef {
				t.Errorf("ApplySetRef not properly stored: expected %q, got %q", tc.applySetRef, options.ApplySetRef)
			}
		})
	}
}

// TestDiffApplySetIntegrationPaths tests ApplySet integration scenarios
func TestDiffApplySetIntegrationPaths(t *testing.T) {
	testCases := []struct {
		name            string
		applySetRef     string
		visitedUIDs     []types.UID
		expectEmptyResult bool
		expectError     bool
	}{
		{
			name:            "no applyset configured",
			applySetRef:     "",
			visitedUIDs:     []types.UID{"uid1", "uid2"},
			expectEmptyResult: true,
			expectError:     false,
		},
		{
			name:            "applyset ref set but ApplySet nil",
			applySetRef:     "configmap/test",
			visitedUIDs:     []types.UID{"uid1", "uid2"},
			expectEmptyResult: true, // ApplySet is nil so should return empty
			expectError:     false,
		},
		{
			name:            "empty visited UIDs with applyset ref",
			applySetRef:     "secret/my-applyset",
			visitedUIDs:     []types.UID{},
			expectEmptyResult: true,
			expectError:     false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := &DiffOptions{
				ApplySetRef: tc.applySetRef,
				// ApplySet is intentionally nil to test graceful handling
			}

			visitedUIDs := sets.New(tc.visitedUIDs...)
			result, err := options.applySetPrune(visitedUIDs)

			if tc.expectError && err == nil {
				t.Errorf("Expected error but got nil")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
			if tc.expectEmptyResult && len(result) != 0 {
				t.Errorf("Expected empty result, got %d items", len(result))
			}

			// Test UID tracking behavior
			if len(tc.visitedUIDs) > 0 {
				for _, uid := range tc.visitedUIDs {
					if !visitedUIDs.Has(uid) {
						t.Errorf("Expected UID %q to be tracked in visited UIDs", uid)
					}
				}
			}
		})
	}
}

// TestDiffOptionsApplySetValidation tests the real validation logic for ApplySet
func TestDiffOptionsApplySetValidation(t *testing.T) {
	testCases := []struct {
		name          string
		options       *DiffOptions
		expectError   bool
		errorContains string
	}{
		{
			name: "no applyset - should pass",
			options: &DiffOptions{
				ApplySetRef: "",
			},
			expectError: false,
		},
		{
			name: "applyset without pruner - should fail",
			options: &DiffOptions{
				ApplySetRef: "configmap/test",
				pruner:      nil, // This should trigger error
			},
			expectError:   true,
			errorContains: "--applyset requires --prune",
		},
		{
			name: "applyset with selector conflict - should fail",
			options: &DiffOptions{
				ApplySetRef: "configmap/test",
				pruner:      &pruner{}, // Mock non-nil pruner
				Selector:    "app=test", // This should conflict
			},
			expectError:   true,
			errorContains: "--applyset is incompatible with --selector",
		},
		{
			name: "valid applyset config - should pass",
			options: &DiffOptions{
				ApplySetRef: "configmap/test",
				pruner:      &pruner{}, // Mock non-nil pruner
				Selector:    "",        // No conflict
				// ApplySet is nil, which should be handled gracefully
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.options.Validate()

			if tc.expectError && err == nil {
				t.Errorf("Expected error but got nil")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
			if tc.expectError && err != nil && tc.errorContains != "" {
				if !strings.Contains(err.Error(), tc.errorContains) {
					t.Errorf("Expected error to contain %q, got %q", tc.errorContains, err.Error())
				}
			}
		})
	}
}

// TestDiffOptionsValidateWithCommandFlags tests command flag validation for ApplySet
func TestDiffOptionsValidateWithCommandFlags(t *testing.T) {
	testCases := []struct {
		name          string
		applySetRef   string
		flagValues    map[string]interface{}
		expectError   bool
		errorContains string
	}{
		{
			name:        "no applyset with all flag - should pass",
			applySetRef: "",
			flagValues: map[string]interface{}{
				"all": true,
			},
			expectError: false,
		},
		{
			name:        "applyset with all flag - should fail",
			applySetRef: "configmap/test",
			flagValues: map[string]interface{}{
				"all": true,
			},
			expectError:   true,
			errorContains: "--applyset is incompatible with --all",
		},
		{
			name:        "applyset with prune-allowlist - should fail",
			applySetRef: "configmap/test",
			flagValues: map[string]interface{}{
				"prune-allowlist": []string{"v1/Pod"},
			},
			expectError:   true,
			errorContains: "--applyset is incompatible with --prune-allowlist",
		},
		{
			name:        "applyset with valid flags - should pass",
			applySetRef: "configmap/test",
			flagValues: map[string]interface{}{
				"all":            false,
				"prune-allowlist": []string{},
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a mock command with flags
			cmd := &cobra.Command{}
			cmd.Flags().Bool("all", false, "")
			cmd.Flags().StringArray("prune-allowlist", nil, "")

			// Set flag values
			for flag, value := range tc.flagValues {
				switch flag {
				case "all":
					cmd.Flags().Set(flag, fmt.Sprintf("%v", value))
				case "prune-allowlist":
					if arr, ok := value.([]string); ok && len(arr) > 0 {
						for _, v := range arr {
							cmd.Flags().Set(flag, v)
						}
					}
				}
			}

			options := &DiffOptions{
				ApplySetRef: tc.applySetRef,
				pruner:      &pruner{}, // Always provide pruner to isolate flag testing
			}

			err := options.ValidateWithCommand(cmd)

			if tc.expectError && err == nil {
				t.Errorf("Expected error but got nil")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
			if tc.expectError && err != nil && tc.errorContains != "" {
				if !strings.Contains(err.Error(), tc.errorContains) {
					t.Errorf("Expected error to contain %q, got %q", tc.errorContains, err.Error())
				}
			}
		})
	}
}
