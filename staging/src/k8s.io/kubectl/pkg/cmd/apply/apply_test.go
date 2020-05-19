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

package apply

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	sptest "k8s.io/apimachinery/pkg/util/strategicpatch/testing"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	dynamicfakeclient "k8s.io/client-go/dynamic/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	clienttesting "k8s.io/client-go/testing"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/openapi"
	utilpointer "k8s.io/utils/pointer"
)

var (
	fakeSchema                 = sptest.Fake{Path: filepath.Join("..", "..", "..", "testdata", "openapi", "swagger.json")}
	testingOpenAPISchemaFns    = []func() (openapi.Resources, error){nil, AlwaysErrorOpenAPISchemaFn, openAPISchemaFn}
	AlwaysErrorOpenAPISchemaFn = func() (openapi.Resources, error) {
		return nil, errors.New("cannot get openapi spec")
	}
	openAPISchemaFn = func() (openapi.Resources, error) {
		s, err := fakeSchema.OpenAPISchema()
		if err != nil {
			return nil, err
		}
		return openapi.NewOpenAPIData(s)
	}
	codec = scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
)

func TestApplyExtraArgsFail(t *testing.T) {
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	c := NewCmdApply("kubectl", f, genericclioptions.NewTestIOStreamsDiscard())
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
	filenameCM                = "../../../testdata/apply/cm.yaml"
	filenameRC                = "../../../testdata/apply/rc.yaml"
	filenameRCArgs            = "../../../testdata/apply/rc-args.yaml"
	filenameRCLastAppliedArgs = "../../../testdata/apply/rc-lastapplied-args.yaml"
	filenameRCNoAnnotation    = "../../../testdata/apply/rc-no-annotation.yaml"
	filenameRCLASTAPPLIED     = "../../../testdata/apply/rc-lastapplied.yaml"
	filenameSVC               = "../../../testdata/apply/service.yaml"
	filenameRCSVC             = "../../../testdata/apply/rc-service.yaml"
	filenameNoExistRC         = "../../../testdata/apply/rc-noexist.yaml"
	filenameRCPatchTest       = "../../../testdata/apply/patch.json"
	dirName                   = "../../../testdata/apply/testdir"
	filenameRCJSON            = "../../../testdata/apply/rc.json"

	filenameWidgetClientside    = "../../../testdata/apply/widget-clientside.yaml"
	filenameWidgetServerside    = "../../../testdata/apply/widget-serverside.yaml"
	filenameDeployObjServerside = "../../../testdata/apply/deploy-serverside.yaml"
	filenameDeployObjClientside = "../../../testdata/apply/deploy-clientside.yaml"
)

func readConfigMapList(t *testing.T, filename string) [][]byte {
	data := readBytesFromFile(t, filename)
	cmList := corev1.ConfigMapList{}
	if err := runtime.DecodeInto(codec, data, &cmList); err != nil {
		t.Fatal(err)
	}

	var listCmBytes [][]byte

	for _, cm := range cmList.Items {
		cmBytes, err := runtime.Encode(codec, &cm)
		if err != nil {
			t.Fatal(err)
		}
		listCmBytes = append(listCmBytes, cmBytes)
	}

	return listCmBytes
}

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
	rcBytes, err := runtime.Encode(codec, rcObj)
	if err != nil {
		t.Fatal(err)
	}

	return metaAccessor.GetName(), rcBytes
}

func readReplicationControllerFromFile(t *testing.T, filename string) *corev1.ReplicationController {
	data := readBytesFromFile(t, filename)
	rc := corev1.ReplicationController{}
	if err := runtime.DecodeInto(codec, data, &rc); err != nil {
		t.Fatal(err)
	}

	return &rc
}

func readUnstructuredFromFile(t *testing.T, filename string) *unstructured.Unstructured {
	data := readBytesFromFile(t, filename)
	unst := unstructured.Unstructured{}
	if err := runtime.DecodeInto(codec, data, &unst); err != nil {
		t.Fatal(err)
	}
	return &unst
}

func readServiceFromFile(t *testing.T, filename string) *corev1.Service {
	data := readBytesFromFile(t, filename)
	svc := corev1.Service{}
	if err := runtime.DecodeInto(codec, data, &svc); err != nil {
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
	original, err := runtime.Encode(unstructured.NewJSONFallbackEncoder(codec), originalObj)
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
	currentAnnotations[corev1.LastAppliedConfigAnnotation] = string(original)
	currentAccessor.SetAnnotations(currentAnnotations)
	current, err := runtime.Encode(unstructured.NewJSONFallbackEncoder(codec), currentObj)
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
	if _, ok := annotationsMap[corev1.LastAppliedConfigAnnotation]; !ok {
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

func TestRunApplyPrintsValidObjectList(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	configMapList := readConfigMapList(t, filenameCM)
	pathCM := "/namespaces/test/configmaps"

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case strings.HasPrefix(p, pathCM) && m == "GET":
				fallthrough
			case strings.HasPrefix(p, pathCM) && m == "PATCH":
				var body io.ReadCloser

				switch p {
				case pathCM + "/test0":
					body = ioutil.NopCloser(bytes.NewReader(configMapList[0]))
				case pathCM + "/test1":
					body = ioutil.NopCloser(bytes.NewReader(configMapList[1]))
				default:
					t.Errorf("unexpected request to %s", p)
				}

				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdApply("kubectl", tf, ioStreams)
	cmd.Flags().Set("filename", filenameCM)
	cmd.Flags().Set("output", "json")
	cmd.Flags().Set("dry-run", "true")
	cmd.Run(cmd, []string{})

	// ensure that returned list can be unmarshaled back into a configmap list
	cmList := corev1.List{}
	if err := runtime.DecodeInto(codec, buf.Bytes(), &cmList); err != nil {
		t.Fatal(err)
	}

	if len(cmList.Items) != 2 {
		t.Fatalf("Expected 2 items in the result; got %d", len(cmList.Items))
	}
	if !strings.Contains(string(cmList.Items[0].Raw), "key1") {
		t.Fatalf("Did not get first ConfigMap at the first position")
	}
	if !strings.Contains(string(cmList.Items[1].Raw), "key2") {
		t.Fatalf("Did not get second ConfigMap at the second position")
	}
}

func TestRunApplyViewLastApplied(t *testing.T) {
	_, rcBytesWithConfig := readReplicationController(t, filenameRCLASTAPPLIED)
	_, rcBytesWithArgs := readReplicationController(t, filenameRCLastAppliedArgs)
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
			name:         "test with file include `%s` in arguments",
			filePath:     filenameRCArgs,
			outputFormat: "",
			expectedErr:  "",
			expectedOut:  "args: -random_flag=%s@domain.com\n",
			selector:     "",
			args:         []string{},
			respBytes:    rcBytesWithArgs,
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
			expectedErr:  "error: Unexpected -o output mode: wide, the flag 'output' must be one of yaml|json\nSee 'view-last-applied -h' for help and examples",
			expectedOut:  "",
			selector:     "",
			args:         []string{"replicationcontroller", "test-rc"},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view resource with label",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "",
			expectedOut:  "test: 1234\n",
			selector:     "name=test-rc",
			args:         []string{"replicationcontroller"},
			respBytes:    rcBytesWithConfig,
		},
		{
			name:         "view resource without annotations",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "error: no last-applied-configuration annotation found on resource: test-rc",
			expectedOut:  "",
			selector:     "",
			args:         []string{"replicationcontroller", "test-rc"},
			respBytes:    rcBytes,
		},
		{
			name:         "view resource no match",
			filePath:     "",
			outputFormat: "",
			expectedErr:  "Error from server (NotFound): the server could not find the requested resource (get replicationcontrollers no-match)",
			expectedOut:  "",
			selector:     "",
			args:         []string{"replicationcontroller", "no-match"},
			respBytes:    nil,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "v1"},
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(test.respBytes))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == "/namespaces/test/replicationcontrollers" && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(test.respBytes))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == "/namespaces/test/replicationcontrollers/no-match" && m == "GET":
						return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Pod{})}, nil
					case p == "/api/v1/namespaces/test" && m == "GET":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			cmdutil.BehaviorOnFatal(func(str string, code int) {
				if str != test.expectedErr {
					t.Errorf("%s: unexpected error: %s\nexpected: %s", test.name, str, test.expectedErr)
				}
			})

			ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApplyViewLastApplied(tf, ioStreams)
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
		})
	}
}

func TestApplyObjectWithoutAnnotation(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	nameRC, rcBytes := readReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(rcBytes))
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				bodyRC := ioutil.NopCloser(bytes.NewReader(rcBytes))
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
	cmd := NewCmdApply("kubectl", tf, ioStreams)
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
	cmdtesting.InitTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply when a local object is specified", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathRC && m == "PATCH":
						validatePatchApplication(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameRC)
			cmd.Flags().Set("output", "name")
			cmd.Run(cmd, []string{})

			// uses the name from the file, not the response
			expectRC := "replicationcontroller/" + nameRC + "\n"
			if buf.String() != expectRC {
				t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func TestApplyPruneObjects(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply returns correct output", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathRC && m == "PATCH":
						validatePatchApplication(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameRC)
			cmd.Flags().Set("prune", "true")
			cmd.Flags().Set("namespace", "test")
			cmd.Flags().Set("output", "yaml")
			cmd.Flags().Set("all", "true")
			cmd.Run(cmd, []string{})

			if !strings.Contains(buf.String(), "test-rc") {
				t.Fatalf("unexpected output: %s\nexpected to contain: %s", buf.String(), "test-rc")
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func TestApplyObjectOutput(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
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

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply returns correct output", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathRC && m == "PATCH":
						validatePatchApplication(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(postPatchData))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameRC)
			cmd.Flags().Set("output", "yaml")
			cmd.Run(cmd, []string{})

			if !strings.Contains(buf.String(), "test-rc") {
				t.Fatalf("unexpected output: %s\nexpected to contain: %s", buf.String(), "test-rc")
			}
			if !strings.Contains(buf.String(), "post-patch: value") {
				t.Fatalf("unexpected output: %s\nexpected to contain: %s", buf.String(), "post-patch: value")
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func TestApplyRetry(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply retries on conflict error", func(t *testing.T) {
			firstPatch := true
			retry := false
			getCount := 0
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						getCount++
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathRC && m == "PATCH":
						if firstPatch {
							firstPatch = false
							statusErr := apierrors.NewConflict(schema.GroupResource{Group: "", Resource: "rc"}, "test-rc", fmt.Errorf("the object has been modified. Please apply at first"))
							bodyBytes, _ := json.Marshal(statusErr)
							bodyErr := ioutil.NopCloser(bytes.NewReader(bodyBytes))
							return &http.Response{StatusCode: http.StatusConflict, Header: cmdtesting.DefaultHeader(), Body: bodyErr}, nil
						}
						retry = true
						validatePatchApplication(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
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
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func TestApplyNonExistObject(t *testing.T) {
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers"
	pathNameRC := pathRC + "/" + nameRC

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == "GET":
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathNameRC && m == "GET":
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathRC && m == "POST":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdApply("kubectl", tf, ioStreams)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	// uses the name from the file, not the response
	expectRC := "replicationcontroller/" + nameRC + "\n"
	if buf.String() != expectRC {
		t.Errorf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
}

func TestApplyEmptyPatch(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	nameRC, _ := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers"
	pathNameRC := pathRC + "/" + nameRC

	verifyPost := false

	var body []byte

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	tf.UnstructuredClient = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/api/v1/namespaces/test" && m == "GET":
				return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathNameRC && m == "GET":
				if body == nil {
					return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
				}
				bodyRC := ioutil.NopCloser(bytes.NewReader(body))
				return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
			case p == pathRC && m == "POST":
				body, _ = ioutil.ReadAll(req.Body)
				verifyPost = true
				bodyRC := ioutil.NopCloser(bytes.NewReader(body))
				return &http.Response{StatusCode: http.StatusCreated, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	// 1. apply non exist object
	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdApply("kubectl", tf, ioStreams)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	expectRC := "replicationcontroller/" + nameRC + "\n"
	if buf.String() != expectRC {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
	}
	if !verifyPost {
		t.Fatal("No server-side post call detected")
	}

	// 2. test apply already exist object, will not send empty patch request
	ioStreams, _, buf, _ = genericclioptions.NewTestIOStreams()
	cmd = NewCmdApply("kubectl", tf, ioStreams)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

	if buf.String() != expectRC {
		t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expectRC)
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

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply on multiple objects", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathRC && m == "PATCH":
						validatePatchApplication(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == pathSVC && m == "GET":
						bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodySVC}, nil
					case p == pathSVC && m == "PATCH":
						validatePatchApplication(t, req)
						bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodySVC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
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
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func readDeploymentFromFile(t *testing.T, file string) []byte {
	raw := readBytesFromFile(t, file)
	obj := &appsv1.Deployment{}
	if err := runtime.DecodeInto(codec, raw, obj); err != nil {
		t.Fatal(err)
	}
	objJSON, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Fatal(err)
	}
	return objJSON
}

func TestApplyNULLPreservation(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	deploymentName := "nginx-deployment"
	deploymentPath := "/namespaces/test/deployments/" + deploymentName

	verifiedPatch := false
	deploymentBytes := readDeploymentFromFile(t, filenameDeployObjServerside)

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply preserves NULL fields", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == deploymentPath && m == "GET":
						body := ioutil.NopCloser(bytes.NewReader(deploymentBytes))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
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
						if _, ok := annotationMap[corev1.LastAppliedConfigAnnotation]; !ok {
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
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameDeployObjClientside)
			cmd.Flags().Set("output", "name")

			cmd.Run(cmd, []string{})

			expected := "deployment.apps/" + deploymentName + "\n"
			if buf.String() != expected {
				t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
			if !verifiedPatch {
				t.Fatal("No server-side patch call detected")
			}
		})
	}
}

// TestUnstructuredApply checks apply operations on an unstructured object
func TestUnstructuredApply(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	name, curr := readAndAnnotateUnstructured(t, filenameWidgetClientside)
	path := "/namespaces/test/widgets/" + name

	verifiedPatch := false

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply works correctly with unstructured objects", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == path && m == "GET":
						body := ioutil.NopCloser(bytes.NewReader(curr))
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     cmdtesting.DefaultHeader(),
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
							StatusCode: http.StatusOK,
							Header:     cmdtesting.DefaultHeader(),
							Body:       body}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameWidgetClientside)
			cmd.Flags().Set("output", "name")
			cmd.Run(cmd, []string{})

			expected := "widget.unit-test.test.com/" + name + "\n"
			if buf.String() != expected {
				t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
			if !verifiedPatch {
				t.Fatal("No server-side patch call detected")
			}
		})
	}
}

// TestUnstructuredIdempotentApply checks repeated apply operation on an unstructured object
func TestUnstructuredIdempotentApply(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	serversideObject := readUnstructuredFromFile(t, filenameWidgetServerside)
	serversideData, err := runtime.Encode(unstructured.NewJSONFallbackEncoder(codec), serversideObject)
	if err != nil {
		t.Fatal(err)
	}
	path := "/namespaces/test/widgets/widget"

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test repeated apply operations on an unstructured object", func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == path && m == "GET":
						body := ioutil.NopCloser(bytes.NewReader(serversideData))
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     cmdtesting.DefaultHeader(),
							Body:       body}, nil
					case p == path && m == "PATCH":
						// In idempotent updates, kubectl will resolve to an empty patch and not send anything to the server
						// Thus, if we reach this branch, kubectl is unnecessarily sending a patch.

						patch, err := ioutil.ReadAll(req.Body)
						if err != nil {
							t.Fatal(err)
						}
						t.Fatalf("Unexpected Patch: %s", patch)
						return nil, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.OpenAPISchemaFunc = fn
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
			cmd.Flags().Set("filename", filenameWidgetClientside)
			cmd.Flags().Set("output", "name")
			cmd.Run(cmd, []string{})

			expected := "widget.unit-test.test.com/widget\n"
			if buf.String() != expected {
				t.Fatalf("unexpected output: %s\nexpected: %s", buf.String(), expected)
			}
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}

func TestRunApplySetLastApplied(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
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
			expectedErr: "error: no last-applied-configuration annotation found on resource: no-annotation, to create the annotation, run the command with --create-annotation",
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
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.UnstructuredClient = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "v1"},
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == pathRC && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == noAnnotationPath && m == "GET":
						bodyRC := ioutil.NopCloser(bytes.NewReader(noAnnotationRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == noExistPath && m == "GET":
						return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Pod{})}, nil
					case p == pathRC && m == "PATCH":
						checkPatchString(t, req)
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case p == "/api/v1/namespaces/test" && m == "GET":
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Namespace{})}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			cmdutil.BehaviorOnFatal(func(str string, code int) {
				if str != test.expectedErr {
					t.Errorf("%s: unexpected error: %s\nexpected: %s", test.name, str, test.expectedErr)
				}
			})

			ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApplySetLastApplied(tf, ioStreams)
			cmd.Flags().Set("filename", test.filePath)
			cmd.Flags().Set("output", test.output)
			cmd.Run(cmd, []string{})

			if buf.String() != test.expectedOut {
				t.Fatalf("%s: unexpected output: %s\nexpected: %s", test.name, buf.String(), test.expectedOut)
			}
		})
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
	if _, ok := annotationsMap[corev1.LastAppliedConfigAnnotation]; !ok {
		t.Fatalf("patch does not contain annotation:\n%s\n", patch)
	}

	resultString := annotationsMap["kubectl.kubernetes.io/last-applied-configuration"]
	if resultString != checkString {
		t.Fatalf("patch annotation is not correct, expect:%s\n but got:%s\n", checkString, resultString)
	}
}

func TestForceApply(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)
	scheme := runtime.NewScheme()
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC
	pathRCList := "/namespaces/test/replicationcontrollers"
	expected := map[string]int{
		"getOk":       6,
		"getNotFound": 1,
		"getList":     0,
		"patch":       6,
		"delete":      1,
		"post":        1,
	}

	for _, fn := range testingOpenAPISchemaFns {
		t.Run("test apply with --force", func(t *testing.T) {
			deleted := false
			isScaledDownToZero := false
			counts := map[string]int{}
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case strings.HasSuffix(p, pathRC) && m == "GET":
						if deleted {
							counts["getNotFound"]++
							return &http.Response{StatusCode: http.StatusNotFound, Header: cmdtesting.DefaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte{}))}, nil
						}
						counts["getOk"]++
						var bodyRC io.ReadCloser
						if isScaledDownToZero {
							rcObj := readReplicationControllerFromFile(t, filenameRC)
							rcObj.Spec.Replicas = utilpointer.Int32Ptr(0)
							rcBytes, err := runtime.Encode(codec, rcObj)
							if err != nil {
								t.Fatal(err)
							}
							bodyRC = ioutil.NopCloser(bytes.NewReader(rcBytes))
						} else {
							bodyRC = ioutil.NopCloser(bytes.NewReader(currentRC))
						}
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
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
						listBytes, err := runtime.Encode(codec, list)
						if err != nil {
							t.Fatal(err)
						}
						bodyRCList := ioutil.NopCloser(bytes.NewReader(listBytes))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRCList}, nil
					case strings.HasSuffix(p, pathRC) && m == "PATCH":
						counts["patch"]++
						if counts["patch"] <= 6 {
							statusErr := apierrors.NewConflict(schema.GroupResource{Group: "", Resource: "rc"}, "test-rc", fmt.Errorf("the object has been modified. Please apply at first"))
							bodyBytes, _ := json.Marshal(statusErr)
							bodyErr := ioutil.NopCloser(bytes.NewReader(bodyBytes))
							return &http.Response{StatusCode: http.StatusConflict, Header: cmdtesting.DefaultHeader(), Body: bodyErr}, nil
						}
						t.Fatalf("unexpected request: %#v after %v tries\n%#v", req.URL, counts["patch"], req)
						return nil, nil
					case strings.HasSuffix(p, pathRC) && m == "PUT":
						counts["put"]++
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						isScaledDownToZero = true
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					case strings.HasSuffix(p, pathRCList) && m == "POST":
						counts["post"]++
						deleted = false
						isScaledDownToZero = false
						bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: bodyRC}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}
			fakeDynamicClient := dynamicfakeclient.NewSimpleDynamicClient(scheme)
			fakeDynamicClient.PrependReactor("delete", "replicationcontrollers", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if deleteAction, ok := action.(clienttesting.DeleteAction); ok {
					if deleteAction.GetName() == nameRC {
						counts["delete"]++
						deleted = true
						return true, nil, nil
					}
				}
				return false, nil, nil
			})
			tf.FakeDynamicClient = fakeDynamicClient
			tf.OpenAPISchemaFunc = fn
			tf.Client = tf.UnstructuredClient
			tf.ClientConfigVal = &restclient.Config{}

			ioStreams, _, buf, errBuf := genericclioptions.NewTestIOStreams()
			cmd := NewCmdApply("kubectl", tf, ioStreams)
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
			if errBuf.String() != "" {
				t.Fatalf("unexpected error output: %s", errBuf.String())
			}
		})
	}
}
