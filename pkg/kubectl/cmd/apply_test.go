/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	kubeerr "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/printers"
)

func TestApplyExtraArgsFail(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	f, _, _, _ := cmdtesting.NewAPIFactory()
	c := NewCmdApply("kubectl", f, buf, errBuf)
	if validateApplyArgs(c, []string{"rc"}) == nil {
		t.Fatalf("unexpected non-error")
	}
}

func validateApplyArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageErrorf(cmd, "Unexpected args: %v", args)
	}
	return nil
}

const (
	filenameRC             = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc.yaml"
	filenameRCNoAnnotation = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc-no-annotation.yaml"
	filenameRCLASTAPPLIED  = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc-lastapplied.yaml"
	filenameSVC            = "../../../test/fixtures/pkg/kubectl/cmd/apply/service.yaml"
	filenameRCSVC          = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc-service.yaml"
	filenameNoExistRC      = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc-noexist.yaml"
	filenameRCPatchTest    = "../../../test/fixtures/pkg/kubectl/cmd/apply/patch.json"
	dirName                = "../../../test/fixtures/pkg/kubectl/cmd/apply/testdir"
	filenameRCJSON         = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc.json"

	filenameWidgetClientside = "../../../test/fixtures/pkg/kubectl/cmd/apply/widget-clientside.yaml"
	filenameWidgetServerside = "../../../test/fixtures/pkg/kubectl/cmd/apply/widget-serverside.yaml"
)

func readBytesFromFile(t *testing.T, filename string) []byte {
	file, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		t.Fatal(err)
	}

	return data
}

func readReplicationController(t *testing.T, filenameRC string) (string, []byte) {
	rcObj := readReplicationControllerFromFile(t, filenameRC)
	metaAccessor, err := meta.Accessor(rcObj)
	if err != nil {
		t.Fatal(err)
	}
	rcBytes, err := runtime.Encode(testapi.Default.Codec(), rcObj)
	if err != nil {
		t.Fatal(err)
	}

	return metaAccessor.GetName(), rcBytes
}

func readReplicationControllerFromFile(t *testing.T, filename string) *api.ReplicationController {
	data := readBytesFromFile(t, filename)
	rc := api.ReplicationController{}
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &rc); err != nil {
		t.Fatal(err)
	}

	return &rc
}

func readUnstructuredFromFile(t *testing.T, filename string) *unstructured.Unstructured {
	data := readBytesFromFile(t, filename)
	unst := unstructured.Unstructured{}
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &unst); err != nil {
		t.Fatal(err)
	}
	return &unst
}

func readServiceFromFile(t *testing.T, filename string) *api.Service {
	data := readBytesFromFile(t, filename)
	svc := api.Service{}
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &svc); err != nil {
		t.Fatal(err)
	}

	return &svc
}

func annotateRuntimeObject(t *testing.T, originalObj, currentObj runtime.Object, kind string) (string, []byte) {
	originalAccessor, err := meta.Accessor(originalObj)
	if err != nil {
		t.Fatal(err)
	}

	// The return value of this function is used in the body of the GET
	// request in the unit tests. Here we are adding a misc label to the object.
	// In tests, the validatePatchApplication() gets called in PATCH request
	// handler in fake round tripper. validatePatchApplication call
	// checks that this DELETE_ME label was deleted by the apply implementation in
	// kubectl.
	originalLabels := originalAccessor.GetLabels()
	originalLabels["DELETE_ME"] = "DELETE_ME"
	originalAccessor.SetLabels(originalLabels)
	original, err := runtime.Encode(testapi.Default.Codec(), originalObj)
	if err != nil {
		t.Fatal(err)
	}

	currentAccessor, err := meta.Accessor(currentObj)
	if err != nil {
		t.Fatal(err)
	}

	currentAnnotations := currentAccessor.GetAnnotations()
	if currentAnnotations == nil {
		currentAnnotations = make(map[string]string)
	}
	currentAnnotations[api.LastAppliedConfigAnnotation] = string(original)
	currentAccessor.SetAnnotations(currentAnnotations)
	current, err := runtime.Encode(testapi.Default.Codec(), currentObj)
	if err != nil {
		t.Fatal(err)
	}

	return currentAccessor.GetName(), current
}

func readAndAnnotateReplicationController(t *testing.T, filename string) (string, []byte) {
	rc1 := readReplicationControllerFromFile(t, filename)
	rc2 := readReplicationControllerFromFile(t, filename)
	return annotateRuntimeObject(t, rc1, rc2, "ReplicationController")
}

func readAndAnnotateService(t *testing.T, filename string) (string, []byte) {
	svc1 := readServiceFromFile(t, filename)
	svc2 := readServiceFromFile(t, filename)
	return annotateRuntimeObject(t, svc1, svc2, "Service")
}

func readAndAnnotateUnstructured(t *testing.T, filename string) (string, []byte) {
	obj1 := readUnstructuredFromFile(t, filename)
	obj2 := readUnstructuredFromFile(t, filename)
	return annotateRuntimeObject(t, obj1, obj2, "Widget")
}

func validatePatchApplication(t *testing.T, req *http.Request) {
	patch, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Fatal(err)
	}

	patchMap := map[string]interface{}{}
	if err := json.Unmarshal(patch, &patchMap); err != nil {
		t.Fatal(err)
	}

	annotationsMap := walkMapPath(t, patchMap, []string{"metadata", "annotations"})
	if _, ok := annotationsMap[api.LastAppliedConfigAnnotation]; !ok {
		t.Fatalf("patch does not contain annotation:\n%s\n", patch)
	}

	labelMap := walkMapPath(t, patchMap, []string{"metadata", "labels"})
	if deleteMe, ok := labelMap["DELETE_ME"]; !ok || deleteMe != nil {
		t.Fatalf("patch does not remove deleted key: DELETE_ME:\n%s\n", patch)
	}
}

func walkMapPath(t *testing.T, start map[string]interface{}, path []string) map[string]interface{} {
	finish := start
	for i := 0; i < len(path); i++ {
		var ok bool
		finish, ok = finish[path[i]].(map[string]interface{})
		if !ok {
			t.Fatalf("key:%s of path:%v not found in map:%v", path[i], path, start)
		}
	}

	return finish
}

func TestRunApplyViewLastApplied(t *testing.T) {
	_, rcBytesWithConfig := readReplicationController(t, filenameRCLASTAPPLIED)
	nameRC, rcBytes := readReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	tests := []struct {
		name, nameRC, pathRC, filePath, outputFormat, expectedErr, expectedOut, selector string
		args                                                                             []string
		respBytes                                                                        []byte
	}{
		{
			name:         "view with file",
			filePath:     filenameRC,
			outputFormat: "",
			expectedErr:  "",
			expectedOut:  "test: 1234\n",
			selector:     "",
			args:         []string{},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view with file json format",
			filePath:     filenameRC,
			outputFormat: "json",
			expectedErr:  "",
			expectedOut:  "{\n  \"test\": 1234\n}\n",
			selector:     "",
			args:         []string{},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view resource/name invalid format",
			filePath:     "",
			outputFormat: "wide",
			expectedErr:  "error: Unexpected -o output mode: wide, the flag 'output' must be one of yaml|json\nSee 'view-last-applied -h' for help and examples.",
			expectedOut:  "",
			selector:     "",
			args:         []string{"rc", "test-rc"},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view resource with label",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "",
			expectedOut:  "test: 1234\n",
			selector:     "name=test-rc",
			args:         []string{"rc"},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view resource without annotations",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "error: no last-applied-configuration annotation found on resource: test-rc",
			expectedOut:  "",
			selector:     "",
			args:         []string{"rc", "test-rc"},
			respBytes:    rcBytes,
		},
		{
			name:         "view resource no match",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "Error from server (NotFound): the server could not find the requested resource (get replicationcontrollers no-match)",
			expectedOut:  "",
			selector:     "",
			args:         []string{"rc", "no-match"},
			respBytes:    nil,
		},
	}
	for _, test := range tests {
		f, tf, codec, _ := cmdtesting.NewAPIFactory()
		tf.Printer = &testPrinter{}
		tf.UnstructuredClient = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: unstructuredSerializer,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == pathRC && m == "GET":
					bodyRC := ioutil.NopCloser(bytes.NewReader(test.respBytes))
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
				case p == "/namespaces/test/replicationcontrollers" && m == "GET":
					bodyRC := ioutil.NopCloser(bytes.NewReader(test.respBytes))
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
				case p == "/namespaces/test/replicationcontrollers/no-match" && m == "GET":
					return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &api.Pod{})}, nil
				case p == "/api/v1/namespaces/test" && m == "GET":
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()
		buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

		cmdutil.BehaviorOnFatal(func(str string, code int) {
			if str != test.expectedErr {
				t.Errorf("%s: unexpected error: %s\nexpected: %s", test.name, str, test.expectedErr)
			}
		})

		cmd := NewCmdApplyViewLastApplied(f, buf, errBuf)
		if test.filePath != "" {
			cmd.Flags().Set("filename", test.filePath)
		}
		if test.outputFormat != "" {
			cmd.Flags().Set("output", test.outputFormat)
		}
		if test.selector != "" {
			cmd.Flags().Set("selector", test.selector)
		}

		cmd.Run(cmd, test.args)
		if buf.String() != test.expectedOut {
			t.Fatalf("%s: unexpected output: %s\nexpected: %s", test.name, buf.String(), test.expectedOut)
		}
	}
}

func TestApplyObjectWithoutAnnotation(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, rcBytes := readReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(rcBytes))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				bodyRC := ioutil.NopCloser(bytes.NewReader(rcBytes))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	expectRC := "replicationcontroller/" + nameRC + "\n"
	expectWarning := fmt.Sprintf(warningNoLastAppliedConfigAnnotation, "kubectl")
	if errBuf.String() != expectWarning {
		t.Fatalf("unexpected non-warning: %s\nexpected: %s", errBuf.String(), expectWarning)
	}
	if buf.String() != expectRC {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
}

func TestApplyObject(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	expectRC := "replicationcontroller/" + nameRC + "\n"
	if buf.String() != expectRC {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
}

func TestApplyObjectOutput(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	// Add some extra data to the post-patch object
	postPatchObj := &unstructured.Unstructured{}
	if err := json.Unmarshal(currentRC, &postPatchObj.Object); err != nil {
		t.Fatal(err)
	}
	postPatchLabels := postPatchObj.GetLabels()
	if postPatchLabels == nil {
		postPatchLabels = map[string]string{}
	}
	postPatchLabels["post-patch"] = "value"
	postPatchObj.SetLabels(postPatchLabels)
	postPatchData, err := json.Marshal(postPatchObj)
	if err != nil {
		t.Fatal(err)
	}

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &printers.YAMLPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(postPatchData))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "yaml")
	cmd.Run(cmd, []string{})

	if !strings.Contains(buf.String(), "name: test-rc") {
		t.Fatalf("unexpected output: %s\nexpected to contain: %s", buf.String(), "name: test-rc")
	}
	if !strings.Contains(buf.String(), "post-patch: value") {
		t.Fatalf("unexpected output: %s\nexpected to contain: %s", buf.String(), "post-patch: value")
	}
}

func TestApplyRetry(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	firstPatch := true
	retry := false
	getCount := 0
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				getCount++
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				if firstPatch {
					firstPatch = false
					statusErr := kubeerr.NewConflict(schema.GroupResource{Group: "", Resource: "rc"}, "test-rc", fmt.Errorf("the object has been modified. Please apply at first."))
					bodyBytes, _ := json.Marshal(statusErr)
					bodyErr := ioutil.NopCloser(bytes.NewReader(bodyBytes))
					return &http.Response{StatusCode: http.StatusConflict, Header: defaultHeader(), Body: bodyErr}, nil
				}
				retry = true
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if !retry || getCount != 2 {
		t.Fatalf("apply didn't retry when get conflict error")
	}

	// uses the name from the file, not the response
	expectRC := "replicationcontroller/" + nameRC + "\n"
	if buf.String() != expectRC {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
}

func TestApplyNonExistObject(t *testing.T) {
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers"
	pathNameRC := pathRC + "/" + nameRC

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == "GET":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathNameRC && m == "GET":
				return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathRC && m == "POST":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	expectRC := "replicationcontroller/" + nameRC + "\n"
	if buf.String() != expectRC {
		t.Errorf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
}

func TestApplyMultipleObjectsAsList(t *testing.T) {
	testApplyMultipleObjects(t, true)
}

func TestApplyMultipleObjectsAsFiles(t *testing.T) {
	testApplyMultipleObjects(t, false)
}

func testApplyMultipleObjects(t *testing.T, asList bool) {
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	nameSVC, currentSVC := readAndAnnotateService(t, filenameSVC)
	pathSVC := "/namespaces/test/services/" + nameSVC

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case p == pathSVC && m == "GET":
				bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodySVC}, nil
			case p == pathSVC && m == "PATCH":
				validatePatchApplication(t, req)
				bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodySVC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	if asList {
		cmd.Flags().Set("filename", filenameRCSVC)
	} else {
		cmd.Flags().Set("filename", filenameRC)
		cmd.Flags().Set("filename", filenameSVC)
	}
	cmd.Flags().Set("output", "name")

	cmd.Run(cmd, []string{})

	// Names should come from the REST response, NOT the files
	expectRC := "replicationcontroller/" + nameRC + "\n"
	expectSVC := "service/" + nameSVC + "\n"
	// Test both possible orders since output is non-deterministic.
	expectOne := expectRC + expectSVC
	expectTwo := expectSVC + expectRC
	if buf.String() != expectOne && buf.String() != expectTwo {
		t.Fatalf("unexpected output: %s\nexpected: %s OR %s", buf.String(), expectOne, expectTwo)
	}
}

const (
	filenameDeployObjServerside = "../../../test/fixtures/pkg/kubectl/cmd/apply/deploy-serverside.yaml"
	filenameDeployObjClientside = "../../../test/fixtures/pkg/kubectl/cmd/apply/deploy-clientside.yaml"
)

func readDeploymentFromFile(t *testing.T, file string) []byte {
	raw := readBytesFromFile(t, file)
	obj := &extensions.Deployment{}
	if err := runtime.DecodeInto(testapi.Extensions.Codec(), raw, obj); err != nil {
		t.Fatal(err)
	}
	objJSON, err := runtime.Encode(testapi.Extensions.Codec(), obj)
	if err != nil {
		t.Fatal(err)
	}
	return objJSON
}

func TestApplyNULLPreservation(t *testing.T) {
	initTestErrorHandler(t)
	deploymentName := "nginx-deployment"
	deploymentPath := "/namespaces/test/deployments/" + deploymentName

	verifiedPatch := false
	deploymentBytes := readDeploymentFromFile(t, filenameDeployObjServerside)

	f, tf, _, _ := cmdtesting.NewTestFactory()
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == deploymentPath && m == "GET":
				body := ioutil.NopCloser(bytes.NewReader(deploymentBytes))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			case p == deploymentPath && m == "PATCH":
				patch, err := ioutil.ReadAll(req.Body)
				if err != nil {
					t.Fatal(err)
				}

				patchMap := map[string]interface{}{}
				if err := json.Unmarshal(patch, &patchMap); err != nil {
					t.Fatal(err)
				}
				annotationMap := walkMapPath(t, patchMap, []string{"metadata", "annotations"})
				if _, ok := annotationMap[api.LastAppliedConfigAnnotation]; !ok {
					t.Fatalf("patch does not contain annotation:\n%s\n", patch)
				}
				strategy := walkMapPath(t, patchMap, []string{"spec", "strategy"})
				if value, ok := strategy["rollingUpdate"]; !ok || value != nil {
					t.Fatalf("patch did not retain null value in key: rollingUpdate:\n%s\n", patch)
				}
				verifiedPatch = true

				// The real API server would had returned the patched object but Kubectl
				// is ignoring the actual return object.
				// TODO: Make this match actual server behavior by returning the patched object.
				body := ioutil.NopCloser(bytes.NewReader(deploymentBytes))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = defaultClientConfig()
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameDeployObjClientside)
	cmd.Flags().Set("output", "name")

	cmd.Run(cmd, []string{})

	expected := "deployment/" + deploymentName + "\n"
	if buf.String() != expected {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
	}
	if !verifiedPatch {
		t.Fatal("No server-side patch call detected")
	}
}

// TestUnstructuredApply checks apply operations on an unstructured object
func TestUnstructuredApply(t *testing.T) {
	initTestErrorHandler(t)
	name, curr := readAndAnnotateUnstructured(t, filenameWidgetClientside)
	path := "/namespaces/test/widgets/" + name

	verifiedPatch := false

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == path && m == "GET":
				body := ioutil.NopCloser(bytes.NewReader(curr))
				return &http.Response{
					StatusCode: 200,
					Header:     defaultHeader(),
					Body:       body}, nil
			case p == path && m == "PATCH":
				contentType := req.Header.Get("Content-Type")
				if contentType != "application/merge-patch+json" {
					t.Fatalf("Unexpected Content-Type: %s", contentType)
				}
				validatePatchApplication(t, req)
				verifiedPatch = true

				body := ioutil.NopCloser(bytes.NewReader(curr))
				return &http.Response{
					StatusCode: 200,
					Header:     defaultHeader(),
					Body:       body}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameWidgetClientside)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	expected := "widget/" + name + "\n"
	if buf.String() != expected {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
	}
	if !verifiedPatch {
		t.Fatal("No server-side patch call detected")
	}
}

// TestUnstructuredIdempotentApply checks repeated apply operation on an unstructured object
func TestUnstructuredIdempotentApply(t *testing.T) {
	initTestErrorHandler(t)

	serversideObject := readUnstructuredFromFile(t, filenameWidgetServerside)
	serversideData, err := runtime.Encode(testapi.Default.Codec(), serversideObject)
	if err != nil {
		t.Fatal(err)
	}
	path := "/namespaces/test/widgets/widget"

	verifiedPatch := false

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == path && m == "GET":
				body := ioutil.NopCloser(bytes.NewReader(serversideData))
				return &http.Response{
					StatusCode: 200,
					Header:     defaultHeader(),
					Body:       body}, nil
			case p == path && m == "PATCH":
				// In idempotent updates, kubectl sends a logically empty
				// request body with the PATCH request.
				// Should look like this:
				// Request Body: {"metadata":{"annotations":{}}}

				patch, err := ioutil.ReadAll(req.Body)
				if err != nil {
					t.Fatal(err)
				}

				contentType := req.Header.Get("Content-Type")
				if contentType != "application/merge-patch+json" {
					t.Fatalf("Unexpected Content-Type: %s", contentType)
				}

				patchMap := map[string]interface{}{}
				if err := json.Unmarshal(patch, &patchMap); err != nil {
					t.Fatal(err)
				}
				if len(patchMap) != 1 {
					t.Fatalf("Unexpected Patch. Has more than 1 entry. path: %s", patch)
				}

				annotationsMap := walkMapPath(t, patchMap, []string{"metadata", "annotations"})
				if len(annotationsMap) != 0 {
					t.Fatalf("Unexpected Patch. Found unexpected annotation: %s", patch)
				}

				verifiedPatch = true

				body := ioutil.NopCloser(bytes.NewReader(serversideData))
				return &http.Response{
					StatusCode: 200,
					Header:     defaultHeader(),
					Body:       body}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}

	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameWidgetClientside)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	expected := "widget/widget\n"
	if buf.String() != expected {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
	}
	if !verifiedPatch {
		t.Fatal("No server-side patch call detected")
	}
}

func TestRunApplySetLastApplied(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	noExistRC, _ := readAndAnnotateReplicationController(t, filenameNoExistRC)
	noExistPath := "/namespaces/test/replicationcontrollers/" + noExistRC

	noAnnotationName, noAnnotationRC := readReplicationController(t, filenameRCNoAnnotation)
	noAnnotationPath := "/namespaces/test/replicationcontrollers/" + noAnnotationName

	tests := []struct {
		name, nameRC, pathRC, filePath, expectedErr, expectedOut, output string
	}{
		{
			name:        "set with exist object",
			filePath:    filenameRC,
			expectedErr: "",
			expectedOut: "replicationcontroller/test-rc\n",
			output:      "name",
		},
		{
			name:        "set with no-exist object",
			filePath:    filenameNoExistRC,
			expectedErr: "Error from server (NotFound): the server could not find the requested resource (get replicationcontrollers no-exist)",
			expectedOut: "",
			output:      "name",
		},
		{
			name:        "set for the annotation does not exist on the live object",
			filePath:    filenameRCNoAnnotation,
			expectedErr: "error: no last-applied-configuration annotation found on resource: no-annotation, to create the annotation, run the command with --create-annotation\nSee 'set-last-applied -h' for help and examples.",
			expectedOut: "",
			output:      "name",
		},
		{
			name:        "set with exist object output json",
			filePath:    filenameRCJSON,
			expectedErr: "",
			expectedOut: "replicationcontroller/test-rc\n",
			output:      "name",
		},
		{
			name:        "set test for a directory of files",
			filePath:    dirName,
			expectedErr: "",
			expectedOut: "replicationcontroller/test-rc\nreplicationcontroller/test-rc\n",
			output:      "name",
		},
	}
	for _, test := range tests {
		f, tf, codec, _ := cmdtesting.NewAPIFactory()
		tf.Printer = &testPrinter{}
		tf.UnstructuredClient = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: unstructuredSerializer,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == pathRC && m == "GET":
					bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
				case p == noAnnotationPath && m == "GET":
					bodyRC := ioutil.NopCloser(bytes.NewReader(noAnnotationRC))
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
				case p == noExistPath && m == "GET":
					return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: objBody(codec, &api.Pod{})}, nil
				case p == pathRC && m == "PATCH":
					checkPatchString(t, req)
					bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
				case p == "/api/v1/namespaces/test" && m == "GET":
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.Namespace{})}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()
		buf, errBuf := bytes.NewBuffer([]byte{}), bytes.NewBuffer([]byte{})

		cmdutil.BehaviorOnFatal(func(str string, code int) {
			if str != test.expectedErr {
				t.Errorf("%s: unexpected error: %s\nexpected: %s", test.name, str, test.expectedErr)
			}
		})

		cmd := NewCmdApplySetLastApplied(f, buf, errBuf)
		cmd.Flags().Set("filename", test.filePath)
		cmd.Flags().Set("output", test.output)
		cmd.Run(cmd, []string{})

		if buf.String() != test.expectedOut {
			t.Fatalf("%s: unexpected output: %s\nexpected: %s", test.name, buf.String(), test.expectedOut)
		}
	}
	cmdutil.BehaviorOnFatal(func(str string, code int) {})
}

func checkPatchString(t *testing.T, req *http.Request) {
	checkString := string(readBytesFromFile(t, filenameRCPatchTest))
	patch, err := ioutil.ReadAll(req.Body)
	if err != nil {
		t.Fatal(err)
	}

	patchMap := map[string]interface{}{}
	if err := json.Unmarshal(patch, &patchMap); err != nil {
		t.Fatal(err)
	}

	annotationsMap := walkMapPath(t, patchMap, []string{"metadata", "annotations"})
	if _, ok := annotationsMap[api.LastAppliedConfigAnnotation]; !ok {
		t.Fatalf("patch does not contain annotation:\n%s\n", patch)
	}

	resultString := annotationsMap["kubectl.kubernetes.io/last-applied-configuration"]
	if resultString != checkString {
		t.Fatalf("patch annotation is not correct, expect:%s\n but got:%s\n", checkString, resultString)
	}
}

func TestForceApply(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC
	pathRCList := "/namespaces/test/replicationcontrollers"
	deleted := false
	counts := map[string]int{}
	expected := map[string]int{
		"getOk":       9,
		"getNotFound": 1,
		"getList":     1,
		"patch":       6,
		"delete":      1,
		"put":         1,
		"post":        1,
	}

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.UnstructuredClient = &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: unstructuredSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasSuffix(p, pathRC) && m == "GET":
				if deleted {
					counts["getNotFound"]++
					return &http.Response{StatusCode: 404, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte{}))}, nil
				}
				counts["getOk"]++
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case strings.HasSuffix(p, pathRCList) && m == "GET":
				counts["getList"]++
				rcObj := readUnstructuredFromFile(t, filenameRC)
				list := &unstructured.UnstructuredList{
					Object: map[string]interface{}{
						"apiVersion": "v1",
						"kind":       "ReplicationControllerList",
					},
					Items: []unstructured.Unstructured{*rcObj},
				}
				listBytes, err := runtime.Encode(testapi.Default.Codec(), list)
				if err != nil {
					t.Fatal(err)
				}
				bodyRCList := ioutil.NopCloser(bytes.NewReader(listBytes))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRCList}, nil
			case strings.HasSuffix(p, pathRC) && m == "PATCH":
				counts["patch"]++
				if counts["patch"] <= 6 {
					statusErr := kubeerr.NewConflict(schema.GroupResource{Group: "", Resource: "rc"}, "test-rc", fmt.Errorf("the object has been modified. Please apply at first."))
					bodyBytes, _ := json.Marshal(statusErr)
					bodyErr := ioutil.NopCloser(bytes.NewReader(bodyBytes))
					return &http.Response{StatusCode: http.StatusConflict, Header: defaultHeader(), Body: bodyErr}, nil
				}
				t.Fatalf("unexpected request: %#v after %v tries\n%#v", req.URL, counts["patch"], req)
				return nil, nil
			case strings.HasSuffix(p, pathRC) && m == "DELETE":
				counts["delete"]++
				deleted = true
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte{}))}, nil
			case strings.HasSuffix(p, pathRC) && m == "PUT":
				counts["put"]++
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			case strings.HasSuffix(p, pathRCList) && m == "POST":
				counts["post"]++
				deleted = false
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Client = tf.UnstructuredClient
	tf.ClientConfig = &restclient.Config{}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply("kubectl", f, buf, errBuf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("force", "true")
	cmd.Run(cmd, []string{})

	for method, exp := range expected {
		if exp != counts[method] {
			t.Errorf("Unexpected amount of %q API calls, wanted %v got %v", method, exp, counts[method])
		}
	}

	if expected := "replicationcontroller/" + nameRC + "\n"; buf.String() != expected {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
	}
}
