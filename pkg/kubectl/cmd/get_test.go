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
	encjson "encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/watch/json"
)

func testData() (*api.PodList, *api.ServiceList, *api.ReplicationControllerList) {
	pods := &api.PodList{
		ListMeta: api.ListMeta{
			ResourceVersion: "15",
		},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
	svc := &api.ServiceList{
		ListMeta: api.ListMeta{
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
		ListMeta: api.ListMeta{
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

func testComponentStatusData() *api.ComponentStatusList {
	good := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionTrue, Message: "ok", Error: "nil"},
		},
	}
	good.Name = "servergood"

	bad := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionFalse, Message: "", Error: "bad status: 500"},
		},
	}
	bad.Name = "serverbad"

	unknown := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{
			{Type: api.ComponentHealthy, Status: api.ConditionUnknown, Message: "", Error: "fizzbuzz error"},
		},
	}
	unknown.Name = "serverunknown"

	return &api.ComponentStatusList{
		Items: []api.ComponentStatus{*good, *bad, *unknown},
	}
}

// Verifies that schemas that are not in the master tree of Kubernetes can be retrieved via Get.
func TestGetUnknownSchemaObject(t *testing.T) {
	f, tf, codec := NewTestFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, &internalType{Name: "foo"})},
	}
	tf.Namespace = "test"
	tf.ClientConfig = &client.Config{Version: latest.Version}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
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
// 2. if the specified output-version is a recognized, valid Scheme, then the list should use that scheme, and otherwise it will default to the client version, latest.Version in this test;
// 3a. if the specified output-version is a recognized, valid Scheme, in which the requested object (replicationcontroller) can be represented, then the object should be returned using that version;
// 3b. otherwise if the specified output-version is unrecognized, but the requested object (replicationcontroller) is recognized by the client's codec, then it will be converted to the client version, latest.Version in this test.
func TestGetUnknownSchemaObjectListGeneric(t *testing.T) {
	testCases := map[string]struct {
		outputVersion   string
		listVersion     string
		testtypeVersion string
		rcVersion       string
	}{
		"handles specific version": {
			outputVersion:   latest.Version,
			listVersion:     latest.Version,
			testtypeVersion: "unlikelyversion",
			rcVersion:       latest.Version,
		},
		"handles second specific version": {
			outputVersion:   "unlikelyversion",
			listVersion:     latest.Version,
			testtypeVersion: "unlikelyversion",
			rcVersion:       latest.Version, // see expected behavior 3b
		},
		"handles common version": {
			outputVersion:   testapi.Version(),
			listVersion:     testapi.Version(),
			testtypeVersion: "unlikelyversion",
			rcVersion:       testapi.Version(),
		},
	}
	for k, test := range testCases {
		apiCodec := runtime.CodecFor(api.Scheme, testapi.Version())
		regularClient := &client.FakeRESTClient{
			Codec: apiCodec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: objBody(apiCodec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})}, nil
			}),
		}

		f, tf, codec := NewMixedFactory(regularClient)
		tf.Printer = &testPrinter{}
		tf.Client = &client.FakeRESTClient{
			Codec: codec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: objBody(codec, &internalType{Name: "foo"})}, nil
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &client.Config{Version: latest.Version}
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdGet(f, buf)
		cmd.SetOutput(buf)
		cmd.Flags().Set("output", "json")
		cmd.Flags().Set("output-version", test.outputVersion)
		err := RunGet(f, buf, cmd, []string{"type/foo", "replicationcontrollers/foo"})
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
	f, tf, _ := NewTestFactory()
	tf.Mapper = latest.RESTMapper
	tf.Typer = api.Scheme
	codec := latest.Codec
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}})},
	}
	tf.Namespace = "test"
	tf.ClientConfig = &client.Config{Version: "v1"}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.Run(cmd, []string{"replicationcontrollers", "foo"})

	if !strings.Contains(buf.String(), "\"foo\"") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}

func TestGetObjects(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, &pods.Items[0])},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
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

func TestGetListObjects(t *testing.T) {
	pods, _, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, pods)},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods"})

	expected := []runtime.Object{pods}
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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, statuses)},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"componentstatuses"})

	expected := []runtime.Object{statuses}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v %#v", expected, actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestGetMultipleTypeObjects(t *testing.T) {
	pods, svc, _ := testData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)
	cmd.Run(cmd, []string{"pods,services"})

	expected := []runtime.Object{pods, svc}
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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	tf.ClientConfig = &client.Config{Version: testapi.Version()}
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("output", "json")
	cmd.Run(cmd, []string{"pods,services"})

	if tf.Printer.(*testPrinter).Objects != nil {
		t.Errorf("unexpected print to default printer")
	}

	out, err := codec.Decode(buf.Bytes())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	list, err := runtime.ExtractList(out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if errs := runtime.DecodeList(list, api.Scheme); len(errs) > 0 {
		t.Fatalf("unexpected error: %v", errs)
	}
	if err := runtime.SetList(out, list); err != nil {
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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(api.LabelSelectorQueryParam(testapi.Version())) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Body: objBody(codec, pods)}, nil
			case "/namespaces/test/services":
				return &http.Response{StatusCode: 200, Body: objBody(codec, svc)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods,services"})

	expected := []runtime.Object{pods, svc}
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

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/nodes/foo":
				return &http.Response{StatusCode: 200, Body: objBody(codec, node)}, nil
			case "/namespaces/test/services/bar":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Run(cmd, []string{"services/bar", "node/foo"})

	expected := []runtime.Object{&svc.Items[0], node}
	actual := tf.Printer.(*testPrinter).Objects
	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, actual))
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}
func watchTestData() ([]api.Pod, []watch.Event) {
	pods := []api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       "test",
				ResourceVersion: "10",
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
	}
	events := []watch.Event{
		{
			Type: watch.Modified,
			Object: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:            "foo",
					Namespace:       "test",
					ResourceVersion: "11",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
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
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
	return pods, events
}

func TestWatchSelector(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.Query().Get(api.LabelSelectorQueryParam(testapi.Version())) != "a=b" {
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
			}
			switch req.URL.Path {
			case "/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &api.PodList{Items: pods})}, nil
			case "/watch/namespaces/test/pods":
				return &http.Response{StatusCode: 200, Body: watchBody(codec, events)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Flags().Set("selector", "a=b")
	cmd.Run(cmd, []string{"pods"})

	expected := []runtime.Object{&api.PodList{Items: pods}, events[0].Object, events[1].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v %#v", expected[0], actual[0])
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchResource(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods[0])}, nil
			case "/watch/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Body: watchBody(codec, events)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := []runtime.Object{&pods[0], events[0].Object, events[1].Object}
	actual := tf.Printer.(*testPrinter).Objects
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("unexpected object: %#v", actual)
	}
	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
}

func TestWatchOnlyResource(t *testing.T) {
	pods, events := watchTestData()

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch req.URL.Path {
			case "/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Body: objBody(codec, &pods[0])}, nil
			case "/watch/namespaces/test/pods/foo":
				return &http.Response{StatusCode: 200, Body: watchBody(codec, events)}, nil
			default:
				t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
				return nil, nil
			}
		}),
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdGet(f, buf)
	cmd.SetOutput(buf)

	cmd.Flags().Set("watch-only", "true")
	cmd.Run(cmd, []string{"pods", "foo"})

	expected := []runtime.Object{events[0].Object, events[1].Object}
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
	enc := json.NewEncoder(buf, codec)
	for i := range events {
		enc.Encode(&events[i])
	}
	return ioutil.NopCloser(buf)
}
