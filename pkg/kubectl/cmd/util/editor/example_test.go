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

package editor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
	"k8s.io/kubernetes/staging/src/k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/staging/src/k8s.io/client-go/dynamic"
)

type editTestCase struct {
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

	Steps []editStep `yaml:"steps"`
}

type editStep struct {
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

var unstructuredSerializer = dynamic.ContentConfig().NegotiatedSerializer

func defaultHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

func defaultClientConfig() *restclient.Config {
	return &restclient.Config{
		APIPath: "/api",
		ContentConfig: restclient.ContentConfig{
			NegotiatedSerializer: scheme.Codecs,
			ContentType:          runtime.ContentTypeJSON,
			GroupVersion:         &schema.GroupVersion{Version: "v1"},
		},
	}
}

func tryIndent(data []byte) []byte {
	indented := &bytes.Buffer{}
	if err := json.Indent(indented, data, "", "\t"); err == nil {
		return indented.Bytes()
	}
	return data
}

func ExampleEditOption() {
	var (
		//name     string
		testcase editTestCase
		i        int
		err      error
	)

	const updateEnvVar = "UPDATE_EDIT_FIXTURE_DATA"
	updateInputFixtures := os.Getenv(updateEnvVar) == "true"

	reqResp := func(req *http.Request) (*http.Response, error) {
		defer func() { i++ }()
		if i > len(testcase.Steps)-1 {
			fmt.Errorf("step %d: more requests than steps, got %s %s", i, req.Method, req.URL.Path)
		}

		step := testcase.Steps[i]

		body := []byte{}
		if req.Body != nil {
			body, err = ioutil.ReadAll(req.Body)
			if err != nil {
				fmt.Errorf("step %d: %v", i, err)
			}
		}

		inputFile := filepath.Join("testdata/edit", "testcase-single-service", step.Input)
		expectedInput, err := ioutil.ReadFile(inputFile)
		if err != nil {
			fmt.Errorf("step %d: %v", i, err)
		}

		outputFile := filepath.Join("testdata/edit", "testcase-single-service", step.Output)
		resultingOutput, err := ioutil.ReadFile(outputFile)
		if err != nil {
			fmt.Errorf("step %d: %v", i, err)
		}

		if req.Method == "POST" && req.URL.Path == "/callback" {
			if step.StepType != "edit" {
				fmt.Errorf("step %d: expected edit step, got %s %s", i, req.Method, req.URL.Path)
			}
			if !bytes.Equal(body, expectedInput) {
				if updateInputFixtures {
					// Convenience to allow recapturing the input and persisting it here
					ioutil.WriteFile(inputFile, body, os.FileMode(0644))
				} else {
					fmt.Errorf("step %d: diff in edit content:\n%s", i, diff.StringDiff(string(body), string(expectedInput)))
					fmt.Printf("If the change in input is expected, return tests with %s=true to update input fixtures", updateEnvVar)
				}
			}
			return &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(resultingOutput))}, nil
		} else {
			if step.StepType != "request" {
				fmt.Errorf("step %d: expected request step, got %s %s", i, req.Method, req.URL.Path)
			}
			body = tryIndent(body)
			expectedInput = tryIndent(expectedInput)
			if req.Method != step.RequestMethod || req.URL.Path != step.RequestPath || req.Header.Get("Content-Type") != step.RequestContentType {
				fmt.Errorf(
					"step %d: expected \n%s %s (content-type=%s)\ngot\n%s %s (content-type=%s)", i,
					step.RequestMethod, step.RequestPath, step.RequestContentType,
					req.Method, req.URL.Path, req.Header.Get("Content-Type"),
				)
			}
			if !bytes.Equal(body, expectedInput) {
				if updateInputFixtures {
					// Convenience to allow recapturing the input and persisting it here
					ioutil.WriteFile(inputFile, body, os.FileMode(0644))
				} else {
					fmt.Errorf("step %d: diff in edit content:\n%s", i, diff.StringDiff(string(body), string(expectedInput)))
					fmt.Printf("If the change in input is expected, return tests with %s=true to update input fixtures", updateEnvVar)
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

	pwd := os.Getenv("PWD")
	if err := os.Chdir("../../"); err != nil {
		fmt.Printf("err: %q\n", err)
	}

	testcaseDir := filepath.Join("testdata", "edit", "testcase-single-service")
	testcaseData, err := ioutil.ReadFile(filepath.Join(testcaseDir, "test.yaml"))

	if err != nil {
		fmt.Printf("got err: %v\n", err)
	}
	if err := yaml.Unmarshal(testcaseData, &testcase); err != nil {
		fmt.Printf("got err: %v\n", err)
	}

	f, tf, _, _ := cmdtesting.NewAPIFactory()

	tf.UnstructuredClientForMappingFunc = func(mapping *meta.RESTMapping) (resource.RESTClient, error) {
		versionedAPIPath := "/api/" + mapping.GroupVersionKind.Version
		return &fake.RESTClient{
			VersionedAPIPath:     versionedAPIPath,
			NegotiatedSerializer: unstructuredSerializer,
			Client:               fake.CreateHTTPClient(reqResp),
		}, nil
	}

	tf.Namespace = testcase.Namespace
	tf.ClientConfig = defaultClientConfig()
	tf.Command = "edit test cmd invocation"

	// set edit mode
	options := &EditOptions{
		EditMode: NormalEditMode,
	}
	cmd := &cobra.Command{
		Use: "edit (RESOURCE/NAME | -f FILENAME)",
	}
	cmd.Flags().Set("output", "yaml")
	testcase.Args = []string{"service", "svc1"}

	buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

	err = options.Complete(f, buf, errBuf, testcase.Args, cmd)
	if err != nil {
		fmt.Printf("commplete unexpected error: %v", err)
	}

	err = options.Run()
	if err != nil {
		fmt.Printf("run unexpected error: %v\n", err)
	}

	if err := os.Chdir(pwd); err != nil {
		fmt.Printf("err: %v\n", err)
	}

	fmt.Println(buf.String())
	// Output:
	// service "svc1" edited
}
