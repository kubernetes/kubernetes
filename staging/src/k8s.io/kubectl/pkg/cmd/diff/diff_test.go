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
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
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

func TestDiffValidateOptions(t *testing.T) {
	testCases := []struct {
		name        string
		options     *DiffOptions
		expectedErr string
	}{
		{
			name: "valid: prune with --all",
			options: &DiffOptions{
				Prune: true,
				All:   true,
			},
			expectedErr: "",
		},
		{
			name: "valid: prune with --selector",
			options: &DiffOptions{
				Prune:    true,
				Selector: "app=test",
			},
			expectedErr: "",
		},
		{
			name: "valid: prune with --applyset",
			options: &DiffOptions{
				Prune:       true,
				ApplySetRef: "secret/my-applyset",
			},
			expectedErr: "",
		},
		{
			name: "valid: no prune flags",
			options: &DiffOptions{
				Prune: false,
			},
			expectedErr: "",
		},
		{
			name: "invalid: --applyset without --prune",
			options: &DiffOptions{
				Prune:       false,
				ApplySetRef: "secret/my-applyset",
			},
			expectedErr: "--applyset requires --prune",
		},
		{
			name: "invalid: --prune with --applyset and --all",
			options: &DiffOptions{
				Prune:       true,
				ApplySetRef: "secret/my-applyset",
				All:         true,
			},
			expectedErr: "--all is incompatible with --applyset",
		},
		{
			name: "invalid: --prune with --applyset and --selector",
			options: &DiffOptions{
				Prune:       true,
				ApplySetRef: "secret/my-applyset",
				Selector:    "app=test",
			},
			expectedErr: "--selector is incompatible with --applyset",
		},
		{
			name: "invalid: --prune with --applyset and --prune-allowlist",
			options: &DiffOptions{
				Prune:          true,
				ApplySetRef:    "secret/my-applyset",
				PruneAllowlist: []string{"v1/pods"},
			},
			expectedErr: "--prune-allowlist is incompatible with --applyset",
		},
		{
			name: "invalid: --prune without --all or --selector",
			options: &DiffOptions{
				Prune: true,
			},
			expectedErr: "all resources selected for prune without explicitly passing --all. To prune all resources, pass the --all flag. If you did not mean to prune all resources, specify a label selector",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.options.Validate()
			if tc.expectedErr == "" {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("expected error %q, but got none", tc.expectedErr)
				} else if err.Error() != tc.expectedErr {
					t.Errorf("expected error %q, but got %q", tc.expectedErr, err.Error())
				}
			}
		})
	}
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
	err = diff.Diff(&obj, Printer{}, true)
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

			err = diff.Diff(&obj, Printer{}, tc.showManagedFields)
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

func fatalNoExit(t *testing.T, ioStreams genericiooptions.IOStreams) func(msg string, code int) {
	return func(msg string, code int) {
		if len(msg) > 0 {
			// add newline if needed
			if !strings.HasSuffix(msg, "\n") {
				msg += "\n"
			}
			fmt.Fprint(ioStreams.ErrOut, msg)
		}
	}
}

func TestDiffWithPruneV2(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	tf := cmdtesting.NewTestFactory().WithNamespace("default")
	defer tf.Cleanup()

	liveNamespace := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Namespace",
			"metadata": map[string]interface{}{
				"name": "test-prune-simple-namespace",
				"uid":  "ns-uid-12345",
			},
		},
	}
	liveSecret := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Secret",
			"metadata": map[string]interface{}{
				"name":      "test-prune-simple-secret-1",
				"namespace": "default",
				"uid":       "secret-uid-67890",
				"labels": map[string]interface{}{
					"applyset.kubernetes.io/part-of": "simple", // Crucial label
				},
			},
		},
	}

	secretList := &unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "SecretList",
		},
		Items: []unstructured.Unstructured{*liveSecret},
	}

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			codec := scheme.Codecs.LegacyCodec(schema.GroupVersion{Version: "v1"})
			switch p, m := req.URL.Path, req.Method; {

			case p == "/namespaces/foo" && m == http.MethodGet:
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     cmdtesting.DefaultHeader(),
					Body: cmdtesting.ObjBody(codec, &unstructured.Unstructured{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Namespace",
							"metadata": map[string]interface{}{
								"name": "foo",
								"uid":  "some-fake-uid",
							},
						},
					}),
				}, nil

			case p == "/namespaces/foo" && m == http.MethodPatch:
				if req.URL.Query().Get("dryRun") == "All" {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     cmdtesting.DefaultHeader(),
						Body: cmdtesting.ObjBody(codec, &unstructured.Unstructured{
							Object: map[string]interface{}{
								"apiVersion": "v1",
								"kind":       "Namespace",
								"metadata": map[string]interface{}{
									"name":        "foo",
									"annotations": map[string]interface{}{"diff": "was-here"},
								},
							},
						}),
					}, nil
				}

			case p == "/namespaces/test-prune-simple-namespace" && m == http.MethodGet:
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, liveNamespace)}, nil

			case p == "/api/v1/namespaces/default/secrets" && m == http.MethodGet:
				if req.URL.Query().Get("labelSelector") == "applyset.kubernetes.io/part-of=simple" {
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, secretList)}, nil
				}

			case m == http.MethodGet && req.URL.Query().Has("labelSelector"):
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &unstructured.UnstructuredList{})}, nil
			}

			t.Fatalf("unexpected request: %s %s", req.Method, req.URL.Path)
			return nil, nil
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	cmdtesting.WithAlphaEnvs([]cmdutil.FeatureGate{cmdutil.ApplySet}, t, func(t *testing.T) {
		testdir := "testdata/prune/"
		ioStreams, _, outBuf, _ := genericiooptions.NewTestIOStreams()
		cmdutil.BehaviorOnFatal(fatalNoExit(t, ioStreams))
		defer cmdutil.DefaultBehaviorOnFatal()

		cmd := NewCmdDiff(tf, ioStreams)
		cmd.Flags().Set("filename", filepath.Join(testdir, "sample_manifest.yaml"))
		cmd.Flags().Set("applyset", "simple")
		cmd.Flags().Set("prune", "true")

		cmd.Run(cmd, []string{})

		got := outBuf.String()

		expectedPattern := `(?m)metadata:\n\+\s+annotations:\n\+\s+diff: was-here\n\s+name: foo\n-\s+uid: some-fake-uid`

		re := regexp.MustCompile(expectedPattern)

		if !re.MatchString(got) {
			t.Errorf("Diff output did not contain the expected changes.\nExpected pattern:\n%s\n\nActual output:\n%s", expectedPattern, got)
		}
	})
}
