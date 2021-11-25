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
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
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

	if oriLang := os.Getenv("LANG"); oriLang != "C" {
		os.Setenv("LANG", "C")
		defer os.Setenv("LANG", oriLang)
	}

	for i, c := range externalDiffCommands {
		os.Setenv("KUBECTL_EXTERNAL_DIFF", c)
		streams, _, stdout, _ := genericclioptions.NewTestIOStreams()
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
	fcontent, err := ioutil.ReadFile(path.Join(diff.Dir.Name, obj.Name()))
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
	entries, err := ioutil.ReadDir(dir.Name)
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
	entries, err = ioutil.ReadDir(dir.Name)
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
	err = diff.Diff(&obj, Printer{})
	if err != nil {
		t.Fatal(err)
	}
	fcontent, err := ioutil.ReadFile(path.Join(diff.From.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent := "live: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
	}

	fcontent, err = ioutil.ReadFile(path.Join(diff.To.Dir.Name, obj.Name()))
	if err != nil {
		t.Fatal(err)
	}
	econtent = "merged: true\n"
	if string(fcontent) != econtent {
		t.Fatalf("File has %q, expected %q", string(fcontent), econtent)
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
