/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"net/http"
	"os"
	"testing"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestApplyExtraArgsFail(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})

	f, _, _ := NewAPIFactory()
	c := NewCmdApply(f, buf)
	if validateApplyArgs(c, []string{"rc"}) == nil {
		t.Fatalf("unexpected non-error")
	}
}

func validateApplyArgs(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return cmdutil.UsageError(cmd, "Unexpected args: %v", args)
	}
	return nil
}

const (
	filenameRC    = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc.yaml"
	filenameSVC   = "../../../test/fixtures/pkg/kubectl/cmd/apply/service.yaml"
	filenameRCSVC = "../../../test/fixtures/pkg/kubectl/cmd/apply/rc-service.yaml"
)

func readBytesFromFile(t *testing.T, filename string) []byte {
	file, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		t.Fatal(err)
	}

	return data
}

func readReplicationControllerFromFile(t *testing.T, filename string) *api.ReplicationController {
	data := readBytesFromFile(t, filename)
	rc := api.ReplicationController{}
	// TODO(jackgr): Replace with a call to testapi.Codec().Decode().
	if err := yaml.Unmarshal(data, &rc); err != nil {
		t.Fatal(err)
	}

	return &rc
}

func readServiceFromFile(t *testing.T, filename string) *api.Service {
	data := readBytesFromFile(t, filename)
	svc := api.Service{}
	// TODO(jackgr): Replace with a call to testapi.Codec().Decode().
	if err := yaml.Unmarshal(data, &svc); err != nil {
		t.Fatal(err)
	}

	return &svc
}

func annotateRuntimeObject(t *testing.T, originalObj, currentObj runtime.Object, kind string) (string, []byte) {
	originalMeta, err := api.ObjectMetaFor(originalObj)
	if err != nil {
		t.Fatal(err)
	}

	originalMeta.Labels["DELETE_ME"] = "DELETE_ME"
	original, err := json.Marshal(originalObj)
	if err != nil {
		t.Fatal(err)
	}

	currentMeta, err := api.ObjectMetaFor(currentObj)
	if err != nil {
		t.Fatal(err)
	}

	if currentMeta.Annotations == nil {
		currentMeta.Annotations = map[string]string{}
	}

	currentMeta.Annotations[kubectl.LastAppliedConfigAnnotation] = string(original)
	current, err := json.Marshal(currentObj)
	if err != nil {
		t.Fatal(err)
	}

	return currentMeta.Name, current
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
	if _, ok := annotationsMap[kubectl.LastAppliedConfigAnnotation]; !ok {
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

func TestApplyObject(t *testing.T) {
	initTestErrorHandler(t)
	nameRC, currentRC := readAndAnnotateReplicationController(t, filenameRC)
	pathRC := "/namespaces/test/replicationcontrollers/" + nameRC

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply(f, buf)
	cmd.Flags().Set("filename", filenameRC)
	cmd.Flags().Set("output", "name")
	cmd.Run(cmd, []string{})

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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathNameRC && m == "GET":
				return &http.Response{StatusCode: 404, Body: ioutil.NopCloser(bytes.NewReader(nil))}, nil
			case p == pathRC && m == "POST":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 201, Body: bodyRC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply(f, buf)
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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		Codec: codec,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == pathRC && m == "GET":
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Body: bodyRC}, nil
			case p == pathRC && m == "PATCH":
				validatePatchApplication(t, req)
				bodyRC := ioutil.NopCloser(bytes.NewReader(currentRC))
				return &http.Response{StatusCode: 200, Body: bodyRC}, nil
			case p == pathSVC && m == "GET":
				bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
				return &http.Response{StatusCode: 200, Body: bodySVC}, nil
			case p == pathSVC && m == "PATCH":
				validatePatchApplication(t, req)
				bodySVC := ioutil.NopCloser(bytes.NewReader(currentSVC))
				return &http.Response{StatusCode: 200, Body: bodySVC}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdApply(f, buf)
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
