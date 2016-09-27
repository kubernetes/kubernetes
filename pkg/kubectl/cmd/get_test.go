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
	encjson "encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/runtime/serializer/streaming"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

func testData() (*api.PodList, *api.ServiceList, *api.ReplicationControllerList) {
	pods := &api.PodList{
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "15",
		},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}
	svc := &api.ServiceList{
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "16",
		},
		Items: []api.Service{
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					SessionAffinity: "None",
					Type:            api.ServiceTypeClusterIP,
				},
			},
		},
	}
	rc := &api.ReplicationControllerList{
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "17",
		},
		Items: []api.ReplicationController{
			{
				ObjectMeta: api.ObjectMeta{Name: "rc1", Namespace: "test", ResourceVersion: "18"},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
				},
			},
		},
	}
	return pods, svc, rc
}

func testDynamicResources() []*discovery.APIGroupResources {
	return []*discovery.APIGroupResources{
		{
			Group: unversioned.APIGroup{
				Versions: []unversioned.GroupVersionForDiscovery{
					{Version: "v1"},
				},
				PreferredVersion: unversioned.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]unversioned.APIResource{
				"v1": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
					{Name: "services", Namespaced: true, Kind: "Service"},
					{Name: "replicationcontrollers", Namespaced: true, Kind: "ReplicationController"},
				},
			},
		},
	}
}

func testComponentStatusData() *api.ComponentStatusList {
	good := api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionTrue, Message: "ok"},
		},
		ObjectMeta: api.ObjectMeta{Name: "servergood"},
	}

	bad := api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionFalse, Message: "", Error: "bad status: 500"},
		},
		ObjectMeta: api.ObjectMeta{Name: "serverbad"},
	}

	unknown := api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionUnknown, Message: "", Error: "fizzbuzz error"},
		},
		ObjectMeta: api.ObjectMeta{Name: "serverunknown"},
	}

	return &api.ComponentStatusList{
		Items: []api.ComponentStatus{good, bad, unknown},
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestGetUnknownSchemaObject(t *testing.T) {
	f, tf, codec, ns := NewTestFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &internalType{Name: "foo"})},
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"type", "foo"})

	expected := &internalType{Name: "foo"}
	actual := tf.Printer.(*testPrinter).Objects[0]
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if buf.String() != fmt.Sprintf("%#v", expected) {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
// Because api.List is part of the Kube API, resource.Builder has to perform a conversion on
// api.Scheme, which may not have access to all objects, and not all objects are at the same
// internal versioning scheme. This test verifies that two isolated schemes (Test, and api.Scheme)
// can be conjoined into a single output object.
//
// The expected behavior of the `kubectl get` command is:
// 1. objects using unrecognized schemes will always be returned using that scheme/version, "unlikelyversion" in this test;
// 2. if the specified output-version is a recognized, valid Scheme, then the list should use that scheme, and otherwise it will default to the client version, testapi.Default.GroupVersion().String() in this test;
// 3a. if the specified output-version is a recognized, valid Scheme, in which the requested object (replicationcontroller) can be represented, then the object should be returned using that version;
// 3b. otherwise if the specified output-version is unrecognized, but the requested object (replicationcontroller) is recognized by the client's codec, then it will be converted to the client version, testapi.Default.GroupVersion().String() in this test.
func TestGetUnknownSchemaObjectListGeneric(t *testing.T) {
	testCases := map[string]struct {
		outputVersion   string
		listVersion     string
		testtypeVersion string
		rcVersion       string
	}{
		"handles specific version": {
			outputVersion:   testapi.Default.GroupVersion().String(),
			listVersion:     testapi.Default.GroupVersion().String(),
			testtypeVersion: unlikelyGV.String(),
			rcVersion:       testapi.Default.GroupVersion().String(),
		},
		"handles second specific version": {
			outputVersion:   "unlikely.group/unlikelyversion",
			listVersion:     testapi.Default.GroupVersion().String(),
			testtypeVersion: unlikelyGV.String(),
			rcVersion:       testapi.Default.GroupVersion().String(), // see expected behavior 3b
		},
		"handles common version": {
			outputVersion:   testapi.Default.GroupVersion().String(),
			listVersion:     testapi.Default.GroupVersion().String(),
			testtypeVersion: unlikelyGV.String(),
			rcVersion:       testapi.Default.GroupVersion().String(),
		},
	}
	for k, test := range testCases {
		apiCodec := testapi.Default.Codec()
		apiNegotiatedSerializer := testapi.Default.NegotiatedSerializer()
		regularClient := &fake.RESTClient{
			NegotiatedSerializer: apiNegotiatedSerializer,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(apiCodec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})}, nil
			}),
		}

		f, tf, codec := NewMixedFactory(regularClient)
		negotiatedSerializer := serializer.NegotiatedSerializerWrapper(
			runtime.SerializerInfo{Serializer: codec},
			runtime.StreamSerializerInfo{})
		tf.Printer = &testPrinter{}
		tf.Client = &fake.RESTClient{
			NegotiatedSerializer: negotiatedSerializer,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &internalType{Name: "foo"})}, nil
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
		buf := bytes.NewBuffer([]byte{})
		errBuf := bytes.NewBuffer([]byte{})
		cmd := NewCmdGet(f, buf, errBuf)
		cmd.SetOutput(buf)
		cmd.Flags().Set("output", "json")

		cmd.Flags().Set("output-version", test.outputVersion)
		err := RunGet(f, buf, errBuf, cmd, []string{"type/foo", "replicationcontrollers/foo"}, &GetOptions{})
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}
		out := make(map[string]interface{})
		if err := encjson.Unmarshal(buf.Bytes(), &out); err != nil {
			t.Errorf("%s: unexpected error: %v\n%s", k, err, buf.String())
			continue
		}
		if out["apiVersion"] != test.listVersion {
			t.Errorf("%s: unexpected list: %#v", k, out)
		}
		arr := out["items"].([]interface{})
		if arr[0].(map[string]interface{})["apiVersion"] != test.testtypeVersion {
			t.Errorf("%s: unexpected list: %#v", k, out)
		}
		if arr[1].(map[string]interface{})["apiVersion"] != test.rcVersion {
			t.Errorf("%s: unexpected list: %#v", k, out)
		}
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestGetSchemaObject(t *testing.T) {
	f, tf, _, _ := NewTestFactory()
	tf.Mapper = testapi.Default.RESTMapper()
	tf.Typer = api.Scheme
	codec := testapi.Default.Codec()
	ns := testapi.Default.NegotiatedSerializer()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})},
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.Run(cmd, []string{"replicationcontrollers", "foo"})

	if !strings.Contains(buf.String(), "\"foo\"") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestGetObjects(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := []runtime.Object{&pods.Items[0]}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetSortedObjects(t *testing.T) {
	pods := &api.PodList{
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "15",
		},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "c", Namespace: "test", ResourceVersion: "10"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "b", Namespace: "test", ResourceVersion: "11"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "a", Namespace: "test", ResourceVersion: "9"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: "v1"}}}

	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	// sorting with metedata.name
	cmd.Flags().Set("sort-by", ".metadata.name")
	cmd.Run(cmd, []string{"pods"})

	// expect sorted: a,b,c
	expected := []runtime.Object{&pods.Items[2], &pods.Items[1], &pods.Items[0]}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}

}

func TestGetObjectsIdentifiedByFile(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("filename", "../../../examples/storage/cassandra/cassandra-controller.yaml")
	cmd.Run(cmd, []string{})

	expected := []runtime.Object{&pods.Items[0]}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetListObjects(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods"})

	expected, err := extractResourceList([]runtime.Object{pods})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: expected %#v, got %#v", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func extractResourceList(objs []runtime.Object) ([]runtime.Object, error) {
	finalObjs := []runtime.Object{}
	for _, obj := range objs {
		items, err := meta.ExtractList(obj)
		if err != nil {
			return nil, err
		}
		for _, item := range items {
			finalObjs = append(finalObjs, item)
		}
	}
	return finalObjs, nil
}

func TestGetAllListObjects(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("show-all", "true")
	cmd.Run(cmd, []string{"pods"})

	expected, err := extractResourceList([]runtime.Object{pods})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v %#v", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetListComponentStatus(t *testing.T) {
	statuses := testComponentStatusData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, statuses)},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"componentstatuses"})

	expected, err := extractResourceList([]runtime.Object{statuses})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: expected %#v, got %#v", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetMultipleTypeObjects(t *testing.T) {
	pods, svc, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods,services"})

	expected, err := extractResourceList([]runtime.Object{pods, svc})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetMultipleTypeObjectsAsList(t *testing.T) {
	pods, svc, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("output", "json")
	cmd.Run(cmd, []string{"pods,services"})

	if tf.Printer.(*testPrinter).Objects != nil {
		t.Errorf("unexpected print to default printer")
	}

	out, err := runtime.Decode(codec, buf.Bytes())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	list, err := meta.ExtractList(out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if errs := runtime.DecodeList(list, codec); len(errs) > 0 {
		t.Fatalf("unexpected error: %v", errs)
	}
	if err := meta.SetList(out, list); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &api.List{
		Items: []runtime.Object{
			&pods.Items[0],
			&pods.Items[1],
			&svc.Items[0],
		},
	}
	if !reflect.DeepEqual(expected, out) {
		t.Errorf("unexpected output: %#v", out)
	}
}

func TestGetMultipleTypeObjectsWithSelector(t *testing.T) {
	pods, svc, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods,services"})

	expected, err := extractResourceList([]runtime.Object{pods, svc})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetMultipleTypeObjectsWithDirectReference(t *testing.T) {
	_, svc, _ := testData()
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.NodeSpec{
			ExternalID: "ext",
		},
	}

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/nodes/foo":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, node)}, nil
			case "/namespaces/test/services/bar":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Run(cmd, []string{"services/bar", "node/foo"})

	expected := []runtime.Object{&svc.Items[0], node}
	actual := tf.Printer.(*testPrinter).Objects
	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %s", diff.ObjectDiff(expected, actual))
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetByNameForcesFlag(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Resp:                 &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods.Items[0])},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods", "foo"})

	showAllFlag, _ := cmd.Flags().GetBool("show-all")
	if !showAllFlag {
		t.Errorf("expected showAll to be true when getting resource by name")
	}
}

func watchTestData() ([]api.Pod, []watch.Event) {
	pods := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				Name:            "bar",
				Namespace:       "test",
				ResourceVersion: "9",
			},
			Spec: apitesting.DeepEqualSafePodSpec(),
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       "test",
				ResourceVersion: "10",
			},
			Spec: apitesting.DeepEqualSafePodSpec(),
		},
	}
	events := []watch.Event{
		// current state events
		{
			Type: watch.Added,
			Object: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:            "bar",
					Namespace:       "test",
					ResourceVersion: "9",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
		{
			Type: watch.Added,
			Object: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "10",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
		// resource events
		{
			Type: watch.Modified,
			Object: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "11",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
		{
			Type: watch.Deleted,
			Object: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "12",
				},
				Spec: apitesting.DeepEqualSafePodSpec(),
			},
		},
	}
	return pods, events
}

func TestWatchSelector(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	podList := &api.PodList{
		Items: pods,
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "10",
		},
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(unversioned.LabelSelectorQueryParam(testapi.Default.GroupVersion().String())) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, podList)}, nil
			case "/watch/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: watchBody(codec, events[2:])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods"})

	expected := []runtime.Object{podList, events[2].Object, events[3].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object:\nExpected: %#v\n\nGot: %#v\n\n", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchResource(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods[1])}, nil
			case "/watch/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: watchBody(codec, events[1:])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := []runtime.Object{&pods[1], events[2].Object, events[3].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object:\nExpected: %#v\n\nGot: %#v\n\n", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchResourceIdentifiedByFile(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/replicationcontrollers/cassandra":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods[1])}, nil
			case "/watch/namespaces/test/replicationcontrollers/cassandra":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: watchBody(codec, events[1:])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("filename", "../../../examples/storage/cassandra/cassandra-controller.yaml")
	cmd.Run(cmd, []string{})

	expected := []runtime.Object{&pods[1], events[2].Object, events[3].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected object: %#v unexpected object: %#v", expected, actual)
	}

	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchOnlyResource(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, &pods[1])}, nil
			case "/watch/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: watchBody(codec, events[1:])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch-only", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := []runtime.Object{events[2].Object, events[3].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchOnlyList(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec, ns := NewAPIFactory()
	tf.Printer = &testPrinter{}
	podList := &api.PodList{
		Items: pods,
		ListMeta: unversioned.ListMeta{
			ResourceVersion: "10",
		},
	}
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, podList)}, nil
			case "/watch/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: watchBody(codec, events[2:])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	errBuf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf, errBuf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch-only", "true")
	cmd.Run(cmd, []string{"pods"})

	expected := []runtime.Object{events[2].Object, events[3].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func watchBody(codec runtime.Codec, events []watch.Event) io.ReadCloser {
	buf := bytes.NewBuffer([]byte{})
	enc := versioned.NewEncoder(streaming.NewEncoder(buf, codec), codec)
	for i := range events {
		enc.Encode(&events[i])
	}
	return json.Framer.NewFrameReader(ioutil.NopCloser(buf))
}
