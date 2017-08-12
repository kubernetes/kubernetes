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

package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type EditTestCase struct {
	Description string `yaml:"description"`
	// create or edit
	Mode             string   `yaml:"mode"`
	Args             []string `yaml:"args"`
	Filename         string   `yaml:"filename"`
	Output           string   `yaml:"outputFormat"`
	OutputPatch      string   `yaml:"outputPatch"`
	SaveConfig       string   `yaml:"saveConfig"`
	Namespace        string   `yaml:"namespace"`
	ExpectedStdout   []string `yaml:"expectedStdout"`
	ExpectedStderr   []string `yaml:"expectedStderr"`
	ExpectedExitCode int      `yaml:"expectedExitCode"`

	Steps []EditStep `yaml:"steps"`
}

type EditStep struct {
	// edit or request
	StepType string `yaml:"type"`

	// only applies to request
	RequestMethod      string `yaml:"expectedMethod,omitempty"`
	RequestPath        string `yaml:"expectedPath,omitempty"`
	RequestContentType string `yaml:"expectedContentType,omitempty"`
	Input              string `yaml:"expectedInput"`

	// only applies to request
	ResponseStatusCode int `yaml:"resultingStatusCode,omitempty"`

	Output string `yaml:"resultingOutput"`
}

func TestEdit(t *testing.T) {
	var (
		name     string
		testcase EditTestCase
		i        int
		err      error
	)

	const updateEnvVar = "UPDATE_EDIT_FIXTURE_DATA"
	updateInputFixtures := os.Getenv(updateEnvVar) == "true"

	reqResp := func(req *http.Request) (*http.Response, error) {
		defer func() { i++ }()
		if i > len(testcase.Steps)-1 {
			t.Fatalf("%s, step %d: more requests than steps, got %s %s", name, i, req.Method, req.URL.Path)
		}
		step := testcase.Steps[i]

		body := []byte{}
		if req.Body != nil {
			body, err = ioutil.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("%s, step %d: %v", name, i, err)
			}
		}

		inputFile := filepath.Join("testdata/edit", "testcase-"+name, step.Input)
		expectedInput, err := ioutil.ReadFile(inputFile)
		if err != nil {
			t.Fatalf("%s, step %d: %v", name, i, err)
		}

		outputFile := filepath.Join("testdata/edit", "testcase-"+name, step.Output)
		resultingOutput, err := ioutil.ReadFile(outputFile)
		if err != nil {
			t.Fatalf("%s, step %d: %v", name, i, err)
		}

		if req.Method == "POST" && req.URL.Path == "/callback" {
			if step.StepType != "edit" {
				t.Fatalf("%s, step %d: expected edit step, got %s %s", name, i, req.Method, req.URL.Path)
			}
			if !bytes.Equal(body, expectedInput) {
				if updateInputFixtures {
					// Convenience to allow recapturing the input and persisting it here
					ioutil.WriteFile(inputFile, body, os.FileMode(0644))
				} else {
					t.Errorf("%s, step %d: diff in edit content:\n%s", name, i, diff.StringDiff(string(body), string(expectedInput)))
					t.Logf("If the change in input is expected, rerun tests with %s=true to update input fixtures", updateEnvVar)
				}
			}
			return &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(resultingOutput))}, nil
		} else {
			if step.StepType != "request" {
				t.Fatalf("%s, step %d: expected request step, got %s %s", name, i, req.Method, req.URL.Path)
			}
			body = tryIndent(body)
			expectedInput = tryIndent(expectedInput)
			if req.Method != step.RequestMethod || req.URL.Path != step.RequestPath || req.Header.Get("Content-Type") != step.RequestContentType {
				t.Fatalf(
					"%s, step %d: expected \n%s %s (content-type=%s)\ngot\n%s %s (content-type=%s)", name, i,
					step.RequestMethod, step.RequestPath, step.RequestContentType,
					req.Method, req.URL.Path, req.Header.Get("Content-Type"),
				)
			}
			if !bytes.Equal(body, expectedInput) {
				if updateInputFixtures {
					// Convenience to allow recapturing the input and persisting it here
					ioutil.WriteFile(inputFile, body, os.FileMode(0644))
				} else {
					t.Errorf("%s, step %d: diff in edit content:\n%s", name, i, diff.StringDiff(string(body), string(expectedInput)))
					t.Logf("If the change in input is expected, rerun tests with %s=true to update input fixtures", updateEnvVar)
				}
			}
			return &http.Response{StatusCode: step.ResponseStatusCode, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(resultingOutput))}, nil
		}
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		resp, _ := reqResp(req)
		for k, vs := range resp.Header {
			w.Header().Del(k)
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	})

	server := httptest.NewServer(handler)
	defer server.Close()

	os.Setenv("KUBE_EDITOR", "testdata/edit/test_editor.sh")
	os.Setenv("KUBE_EDITOR_CALLBACK", server.URL+"/callback")

	testcases := sets.NewString()
	filepath.Walk("testdata/edit", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == "testdata/edit" {
			return nil
		}
		name := filepath.Base(path)
		if info.IsDir() {
			if strings.HasPrefix(name, "testcase-") {
				testcases.Insert(strings.TrimPrefix(name, "testcase-"))
			}
			return filepath.SkipDir
		}
		return nil
	})
	// sanity check that we found the right folder
	if !testcases.Has("create-list") {
		t.Fatalf("Error locating edit testcases")
	}

	for _, testcaseName := range testcases.List() {
		t.Logf("Running testcase: %s", testcaseName)
		i = 0
		name = testcaseName
		testcase = EditTestCase{}
		testcaseDir := filepath.Join("testdata", "edit", "testcase-"+name)
		testcaseData, err := ioutil.ReadFile(filepath.Join(testcaseDir, "test.yaml"))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := yaml.Unmarshal(testcaseData, &testcase); err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		f, tf, _, _ := cmdtesting.NewAPIFactory()
		tf.Printer = &testPrinter{}
		tf.UnstructuredClientForMappingFunc = func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
			versionedAPIPath := ""
			if mapping.GroupVersionKind.Group == "" {
				versionedAPIPath = "/api/" + mapping.GroupVersionKind.Version
			} else {
				versionedAPIPath = "/apis/" + mapping.GroupVersionKind.Group + "/" + mapping.GroupVersionKind.Version
			}
			return &fake.RESTClient{
				APIRegistry:          api.Registry,
				VersionedAPIPath:     versionedAPIPath,
				NegotiatedSerializer: unstructuredSerializer,
				Client:               fake.CreateHTTPClient(reqResp),
			}, nil
		}

		if len(testcase.Namespace) > 0 {
			tf.Namespace = testcase.Namespace
		}
		tf.ClientConfig = defaultClientConfig()
		tf.Command = "edit test cmd invocation"
		buf := bytes.NewBuffer([]byte{})
		errBuf := bytes.NewBuffer([]byte{})

		var cmd *cobra.Command
		switch testcase.Mode {
		case "edit":
			cmd = NewCmdEdit(f, buf, errBuf)
		case "create":
			cmd = NewCmdCreate(f, buf, errBuf)
			cmd.Flags().Set("edit", "true")
		case "edit-last-applied":
			cmd = NewCmdApplyEditLastApplied(f, buf, errBuf)
		default:
			t.Errorf("%s: unexpected mode %s", name, testcase.Mode)
			continue
		}
		if len(testcase.Filename) > 0 {
			cmd.Flags().Set("filename", filepath.Join(testcaseDir, testcase.Filename))
		}
		if len(testcase.Output) > 0 {
			cmd.Flags().Set("output", testcase.Output)
		}
		if len(testcase.OutputPatch) > 0 {
			cmd.Flags().Set("output-patch", testcase.OutputPatch)
		}
		if len(testcase.SaveConfig) > 0 {
			cmd.Flags().Set("save-config", testcase.SaveConfig)
		}

		cmdutil.BehaviorOnFatal(func(str string, code int) {
			errBuf.WriteString(str)
			if testcase.ExpectedExitCode != code {
				t.Errorf("%s: expected exit code %d, got %d: %s", name, testcase.ExpectedExitCode, code, str)
			}
		})

		cmd.Run(cmd, testcase.Args)

		stdout := buf.String()
		stderr := errBuf.String()

		for _, s := range testcase.ExpectedStdout {
			if !strings.Contains(stdout, s) {
				t.Errorf("%s: expected to see '%s' in stdout\n\nstdout:\n%s\n\nstderr:\n%s", name, s, stdout, stderr)
			}
		}
		for _, s := range testcase.ExpectedStderr {
			if !strings.Contains(stderr, s) {
				t.Errorf("%s: expected to see '%s' in stderr\n\nstdout:\n%s\n\nstderr:\n%s", name, s, stdout, stderr)
			}
		}
		if i < len(testcase.Steps) {
			t.Errorf("%s: saw %d steps, testcase included %d additional steps that were not exercised", name, i, len(testcase.Steps)-i)
		}
	}
}

func tryIndent(data []byte) []byte {
	indented := &bytes.Buffer{}
	if err := json.Indent(indented, data, "", "\t"); err == nil {
		return indented.Bytes()
	}
	return data
}
