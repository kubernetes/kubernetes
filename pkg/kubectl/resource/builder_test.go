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

package resource

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/watch"
	watchjson "k8s.io/kubernetes/pkg/watch/json"
)

func stringBody(body string) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(body)))
}

func watchBody(events ...watch.Event) string {
	buf := &bytes.Buffer{}
	enc := watchjson.NewEncoder(buf, testapi.Default.Codec())
	for _, e := range events {
		enc.Encode(&e)
	}
	return buf.String()
}

func fakeClient() ClientMapper {
	return ClientMapperFunc(func(*meta.RESTMapping) (RESTClient, error) {
		return &fake.RESTClient{}, nil
	})
}

func fakeClientWith(testName string, t *testing.T, data map[string]string) ClientMapper {
	return ClientMapperFunc(func(*meta.RESTMapping) (RESTClient, error) {
		return &fake.RESTClient{
			Codec: testapi.Default.Codec(),
			Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				p := req.URL.Path
				q := req.URL.RawQuery
				if len(q) != 0 {
					p = p + "?" + q
				}
				body, ok := data[p]
				if !ok {
					t.Fatalf("%s: unexpected request: %s (%s)\n%#v", testName, p, req.URL, req)
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       stringBody(body),
				}, nil
			}),
		}, nil
	})
}

func testData() (*api.PodList, *api.ServiceList) {
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
					Type:            "ClusterIP",
					SessionAffinity: "None",
				},
			},
		},
	}
	return pods, svc
}

func streamTestData() (io.Reader, *api.PodList, *api.ServiceList) {
	pods, svc := testData()
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), pods)))
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), svc)))
	}()
	return r, pods, svc
}

func JSONToYAMLOrDie(in []byte) []byte {
	data, err := yaml.JSONToYAML(in)
	if err != nil {
		panic(err)
	}
	return data
}

func streamYAMLTestData() (io.Reader, *api.PodList, *api.ServiceList) {
	pods, svc := testData()
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write(JSONToYAMLOrDie([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), pods))))
		w.Write([]byte("\n---\n"))
		w.Write(JSONToYAMLOrDie([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), svc))))
	}()
	return r, pods, svc
}

func streamTestObject(obj runtime.Object) io.Reader {
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), obj)))
	}()
	return r
}

type testVisitor struct {
	InjectErr error
	Infos     []*Info
}

func (v *testVisitor) Handle(info *Info, err error) error {
	if err != nil {
		return err
	}
	v.Infos = append(v.Infos, info)
	return v.InjectErr
}

func (v *testVisitor) Objects() []runtime.Object {
	objects := []runtime.Object{}
	for i := range v.Infos {
		objects = append(objects, v.Infos[i].Object)
	}
	return objects
}

func TestPathBuilder(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, "../../../examples/guestbook/redis-master-controller.yaml")

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || !singular || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}

	info := test.Infos[0]
	if info.Name != "redis-master" || info.Namespace != "" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}
}

func TestNodeBuilder(t *testing.T) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "node1", Namespace: "should-not-have", ResourceVersion: "10"},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1000m"),
				api.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
	}
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), node)))
	}()

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").Stream(r, "STDIN")

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}
	info := test.Infos[0]
	if info.Name != "node1" || info.Namespace != "" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}
}

func TestPathBuilderWithMultiple(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, "../../../examples/guestbook/redis-master-controller.yaml").
		FilenameParam(false, "../../../examples/pod").
		NamespaceParam("test").DefaultNamespace()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}

	info := test.Infos[0]
	if _, ok := info.Object.(*api.ReplicationController); !ok || info.Name != "redis-master" || info.Namespace != "test" {
		t.Errorf("unexpected info: %#v", info)
	}
	info = test.Infos[1]
	if _, ok := info.Object.(*api.Pod); !ok || info.Name != "nginx" || info.Namespace != "test" {
		t.Errorf("unexpected info: %#v", info)
	}
}

func TestDirectoryBuilder(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, "../../../examples/guestbook").
		NamespaceParam("test").DefaultNamespace()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) < 4 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}

	found := false
	for _, info := range test.Infos {
		if info.Name == "redis-master" && info.Namespace == "test" && info.Object != nil {
			found = true
		}
	}
	if !found {
		t.Errorf("unexpected responses: %#v", test.Infos)
	}
}

func TestNamespaceOverride(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "foo", Name: "test"}})))
	}))
	defer s.Close()

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, s.URL).
		NamespaceParam("test")

	test := &testVisitor{}

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 1 && test.Infos[0].Namespace != "foo" {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}

	b = NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(true, s.URL).
		NamespaceParam("test")

	test = &testVisitor{}

	err = b.Do().Visit(test.Handle)
	if err == nil {
		t.Fatalf("expected namespace error. got: %#v", test.Infos)
	}
}

func TestURLBuilder(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "foo", Name: "test"}})))
	}))
	defer s.Close()

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, s.URL).
		NamespaceParam("test")

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || !singular || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	info := test.Infos[0]
	if info.Name != "test" || info.Namespace != "foo" || info.Object == nil {
		t.Errorf("unexpected info: %#v", info)
	}
}

func TestURLBuilderRequireNamespace(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "foo", Name: "test"}})))
	}))
	defer s.Close()

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		FilenameParam(false, s.URL).
		NamespaceParam("test").RequireNamespace()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err == nil || !singular || len(test.Infos) != 0 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
}

func TestResourceByName(t *testing.T) {
	pods, _ := testData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
	})).
		NamespaceParam("test")

	test := &testVisitor{}
	singular := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods", "foo")

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || !singular || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !reflect.DeepEqual(&pods.Items[0], test.Objects()[0]) {
		t.Errorf("unexpected object: %#v", test.Objects()[0])
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "pods" {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestMultipleResourceByTheSameName(t *testing.T) {
	pods, svcs := testData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
		"/namespaces/test/pods/baz":     runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[1]),
		"/namespaces/test/services/foo": runtime.EncodeOrDie(testapi.Default.Codec(), &svcs.Items[0]),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(testapi.Default.Codec(), &svcs.Items[0]),
	})).
		NamespaceParam("test")

	test := &testVisitor{}
	singular := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods,services", "foo", "baz")

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 4 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !api.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &svcs.Items[0], &svcs.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}

	if _, err := b.Do().ResourceMapping(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestResourceNames(t *testing.T) {
	pods, svc := testData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(testapi.Default.Codec(), &svc.Items[0]),
	})).
		NamespaceParam("test")

	test := &testVisitor{}

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceNames("pods", "foo", "services/baz")

	err := b.Do().Visit(test.Handle)
	if err != nil || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %#v", err, test.Infos)
	}
	if !reflect.DeepEqual(&pods.Items[0], test.Objects()[0]) {
		t.Errorf("unexpected object: \n%#v, expected: \n%#v", test.Objects()[0], &pods.Items[0])
	}
	if !reflect.DeepEqual(&svc.Items[0], test.Objects()[1]) {
		t.Errorf("unexpected object: \n%#v, expected: \n%#v", test.Objects()[1], &svc.Items[0])
	}
}

func TestResourceByNameWithoutRequireObject(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{})).
		NamespaceParam("test")

	test := &testVisitor{}
	singular := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods", "foo").RequireObject(false)

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || !singular || len(test.Infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if test.Infos[0].Name != "foo" {
		t.Errorf("unexpected name: %#v", test.Infos[0].Name)
	}
	if test.Infos[0].Object != nil {
		t.Errorf("unexpected object: %#v", test.Infos[0].Object)
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Kind != "Pod" || mapping.Resource != "pods" {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestResourceByNameAndEmptySelector(t *testing.T) {
	pods, _ := testData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo": runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
	})).
		NamespaceParam("test").
		SelectorParam("").
		ResourceTypeOrNameArgs(true, "pods", "foo")

	singular := false
	infos, err := b.Do().IntoSingular(&singular).Infos()
	if err != nil || !singular || len(infos) != 1 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, infos)
	}
	if !reflect.DeepEqual(&pods.Items[0], infos[0].Object) {
		t.Errorf("unexpected object: %#v", infos[0])
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "pods" {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestSelector(t *testing.T) {
	pods, svc := testData()
	labelKey := api.LabelSelectorQueryParam(testapi.Default.Version())
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db":     runtime.EncodeOrDie(testapi.Default.Codec(), pods),
		"/namespaces/test/services?" + labelKey + "=a%3Db": runtime.EncodeOrDie(testapi.Default.Codec(), svc),
	})).
		SelectorParam("a=b").
		NamespaceParam("test").
		Flatten()

	test := &testVisitor{}
	singular := false

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}

	b.ResourceTypeOrNameArgs(true, "pods,service")

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !api.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &svc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}

	if _, err := b.Do().ResourceMapping(); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSelectorRequiresKnownTypes(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		SelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypes("unknown")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestSingleResourceType(t *testing.T) {
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		SelectorParam("a=b").
		SingleResourceType().
		ResourceTypeOrNameArgs(true, "pods,services")

	if b.Do().Err() == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestHasNamesArg(t *testing.T) {
	testCases := map[string]struct {
		args     []string
		expected bool
	}{
		"resource/name": {
			args:     []string{"pods/foo"},
			expected: true,
		},
		"resource name": {
			args:     []string{"pods", "foo"},
			expected: true,
		},
		"resource1,resource2 name": {
			args:     []string{"pods,rc", "foo"},
			expected: true,
		},
		"resource1,group2/resource2 name": {
			args:     []string{"pods,experimental/deployments", "foo"},
			expected: true,
		},
		"group/resource name": {
			args:     []string{"experimental/deployments", "foo"},
			expected: true,
		},
		"group/resource/name": {
			args:     []string{"experimental/deployments/foo"},
			expected: true,
		},
		"group1/resource1,group2/resource2": {
			args:     []string{"experimental/daemonsets,experimental/deployments"},
			expected: false,
		},
		"resource1,group2/resource2": {
			args:     []string{"pods,experimental/deployments"},
			expected: false,
		},
		"group/resource/name,group2/resource2": {
			args:     []string{"experimental/deployments/foo,controller/deamonset"},
			expected: false,
		},
	}
	for k, testCase := range testCases {
		b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient())
		if testCase.expected != b.hasNamesArg(testCase.args) {
			t.Errorf("%s: unexpected argument - expected: %v", k, testCase.expected)
		}
	}
}

func TestSplitGroupResourceTypeName(t *testing.T) {
	expectNoErr := func(err error) bool { return err == nil }
	expectErr := func(err error) bool { return err != nil }
	testCases := map[string]struct {
		arg           string
		expectedTuple resourceTuple
		expectedOK    bool
		errFn         func(error) bool
	}{
		"group/type/name": {
			arg:           "experimental/deployments/foo",
			expectedTuple: resourceTuple{Resource: "experimental/deployments", Name: "foo"},
			expectedOK:    true,
			errFn:         expectNoErr,
		},
		"type/name": {
			arg:           "pods/foo",
			expectedTuple: resourceTuple{Resource: "pods", Name: "foo"},
			expectedOK:    true,
			errFn:         expectNoErr,
		},
		"type": {
			arg:        "pods",
			expectedOK: false,
			errFn:      expectNoErr,
		},
		"": {
			arg:        "",
			expectedOK: false,
			errFn:      expectNoErr,
		},
		"/": {
			arg:        "/",
			expectedOK: false,
			errFn:      expectErr,
		},
		"group/type/name/something": {
			arg:        "experimental/deployments/foo/something",
			expectedOK: false,
			errFn:      expectErr,
		},
	}
	for k, testCase := range testCases {
		tuple, ok, err := splitGroupResourceTypeName(testCase.arg)
		if !testCase.errFn(err) {
			t.Errorf("%s: unexpected error: %v", k, err)
		}
		if ok != testCase.expectedOK {
			t.Errorf("%s: unexpected ok: %v", k, ok)
		}
		if testCase.expectedOK && !reflect.DeepEqual(tuple, testCase.expectedTuple) {
			t.Errorf("%s: unexpected tuple - expected: %v, got: %v", k, testCase.expectedTuple, tuple)
		}
	}
}

func TestResourceTuple(t *testing.T) {
	expectNoErr := func(err error) bool { return err == nil }
	expectErr := func(err error) bool { return err != nil }
	testCases := map[string]struct {
		args  []string
		errFn func(error) bool
	}{
		"valid": {
			args:  []string{"pods/foo"},
			errFn: expectNoErr,
		},
		"valid multiple with name indirection": {
			args:  []string{"pods/foo", "pod/bar"},
			errFn: expectNoErr,
		},
		"valid multiple with namespaced and non-namespaced types": {
			args:  []string{"nodes/foo", "pod/bar"},
			errFn: expectNoErr,
		},
		"mixed arg types": {
			args:  []string{"pods/foo", "bar"},
			errFn: expectErr,
		},
		/*"missing resource": {
			args:  []string{"pods/foo2"},
			errFn: expectNoErr, // not an error because resources are lazily visited
		},*/
		"comma in resource": {
			args:  []string{",pods/foo"},
			errFn: expectErr,
		},
		"multiple types in resource": {
			args:  []string{"pods,services/foo"},
			errFn: expectErr,
		},
		"unknown resource type": {
			args:  []string{"unknown/foo"},
			errFn: expectErr,
		},
		"leading slash": {
			args:  []string{"/bar"},
			errFn: expectErr,
		},
		"trailing slash": {
			args:  []string{"bar/"},
			errFn: expectErr,
		},
	}
	for k, testCase := range testCases {
		for _, requireObject := range []bool{true, false} {
			expectedRequests := map[string]string{}
			if requireObject {
				pods, _ := testData()
				expectedRequests = map[string]string{
					"/namespaces/test/pods/foo": runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
					"/namespaces/test/pods/bar": runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0]),
					"/nodes/foo":                runtime.EncodeOrDie(testapi.Default.Codec(), &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
				}
			}

			b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith(k, t, expectedRequests)).
				NamespaceParam("test").DefaultNamespace().
				ResourceTypeOrNameArgs(true, testCase.args...).RequireObject(requireObject)

			r := b.Do()

			if !testCase.errFn(r.Err()) {
				t.Errorf("%s: unexpected error: %v", k, r.Err())
			}
			if r.Err() != nil {
				continue
			}
			switch {
			case (r.singular && len(testCase.args) != 1),
				(!r.singular && len(testCase.args) == 1):
				t.Errorf("%s: result had unexpected singular value", k)
			}
			info, err := r.Infos()
			if err != nil {
				// test error
				continue
			}
			if len(info) != len(testCase.args) {
				t.Errorf("%s: unexpected number of infos returned: %#v", k, info)
			}
		}
	}
}

func TestStream(t *testing.T) {
	r, pods, rc := streamTestData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").Stream(r, "STDIN").Flatten()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !api.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &rc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestYAMLStream(t *testing.T) {
	r, pods, rc := streamYAMLTestData()
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").Stream(r, "STDIN").Flatten()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !api.Semantic.DeepDerivative([]runtime.Object{&pods.Items[0], &pods.Items[1], &rc.Items[0]}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestMultipleObject(t *testing.T) {
	r, pods, svc := streamTestData()
	obj, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").Stream(r, "STDIN").Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := &api.List{
		Items: []runtime.Object{
			&pods.Items[0],
			&pods.Items[1],
			&svc.Items[0],
		},
	}
	if !api.Semantic.DeepDerivative(expected, obj) {
		t.Errorf("unexpected visited objects: %#v", obj)
	}
}

func TestContinueOnErrorVisitor(t *testing.T) {
	r, _, _ := streamTestData()
	req := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		ContinueOnError().
		NamespaceParam("test").Stream(r, "STDIN").Flatten().
		Do()
	count := 0
	testErr := fmt.Errorf("test error")
	err := req.Visit(func(_ *Info, _ error) error {
		count++
		if count > 1 {
			return testErr
		}
		return nil
	})
	if err == nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 3 {
		t.Fatalf("did not visit all infos: %d", count)
	}
	agg, ok := err.(errors.Aggregate)
	if !ok {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(agg.Errors()) != 2 || agg.Errors()[0] != testErr || agg.Errors()[1] != testErr {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSingularObject(t *testing.T) {
	obj, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, "../../../examples/guestbook/redis-master-controller.yaml").
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rc, ok := obj.(*api.ReplicationController)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if rc.Name != "redis-master" || rc.Namespace != "test" {
		t.Errorf("unexpected controller: %#v", rc)
	}
}

func TestSingularObjectNoExtension(t *testing.T) {
	obj, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, "../../../examples/pod").
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pod, ok := obj.(*api.Pod)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if pod.Name != "nginx" || pod.Namespace != "test" {
		t.Errorf("unexpected pod: %#v", pod)
	}
}

func TestSingularRootScopedObject(t *testing.T) {
	node := &api.Node{ObjectMeta: api.ObjectMeta{Name: "test"}, Spec: api.NodeSpec{ExternalID: "test"}}
	r := streamTestObject(node)
	infos, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").DefaultNamespace().
		Stream(r, "STDIN").
		Flatten().
		Do().Infos()

	if err != nil || len(infos) != 1 {
		t.Fatalf("unexpected error: %v", err)
	}

	if infos[0].Namespace != "" {
		t.Errorf("namespace should be empty: %#v", infos[0])
	}
	n, ok := infos[0].Object.(*api.Node)
	if !ok {
		t.Fatalf("unexpected object: %#v", infos[0].Object)
	}
	if n.Name != "test" || n.Namespace != "" {
		t.Errorf("unexpected object: %#v", n)
	}
}

func TestListObject(t *testing.T) {
	pods, _ := testData()
	labelKey := api.LabelSelectorQueryParam(testapi.Default.Version())
	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db": runtime.EncodeOrDie(testapi.Default.Codec(), pods),
	})).
		SelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypeOrNameArgs(true, "pods").
		Flatten()

	obj, err := b.Do().Object()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	list, ok := obj.(*api.List)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	if list.ResourceVersion != pods.ResourceVersion || len(list.Items) != 2 {
		t.Errorf("unexpected list: %#v", list)
	}

	mapping, err := b.Do().ResourceMapping()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mapping.Resource != "pods" {
		t.Errorf("unexpected resource mapping: %#v", mapping)
	}
}

func TestListObjectWithDifferentVersions(t *testing.T) {
	pods, svc := testData()
	labelKey := api.LabelSelectorQueryParam(testapi.Default.Version())
	obj, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods?" + labelKey + "=a%3Db":     runtime.EncodeOrDie(testapi.Default.Codec(), pods),
		"/namespaces/test/services?" + labelKey + "=a%3Db": runtime.EncodeOrDie(testapi.Default.Codec(), svc),
	})).
		SelectorParam("a=b").
		NamespaceParam("test").
		ResourceTypeOrNameArgs(true, "pods,services").
		Flatten().
		Do().Object()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	list, ok := obj.(*api.List)
	if !ok {
		t.Fatalf("unexpected object: %#v", obj)
	}
	// resource version differs between type lists, so it's not possible to get a single version.
	if list.ResourceVersion != "" || len(list.Items) != 3 {
		t.Errorf("unexpected list: %#v", list)
	}
}

func TestWatch(t *testing.T) {
	_, svc := testData()
	w, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/watch/namespaces/test/services/redis-master?resourceVersion=12": watchBody(watch.Event{
			Type:   watch.Added,
			Object: &svc.Items[0],
		}),
	})).
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, "../../../examples/guestbook/redis-master-service.yaml").Flatten().
		Do().Watch("12")

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	defer w.Stop()
	ch := w.ResultChan()
	select {
	case obj := <-ch:
		if obj.Type != watch.Added {
			t.Fatalf("unexpected watch event %#v", obj)
		}
		service, ok := obj.Object.(*api.Service)
		if !ok {
			t.Fatalf("unexpected object: %#v", obj)
		}
		if service.Name != "baz" || service.ResourceVersion != "12" {
			t.Errorf("unexpected service: %#v", service)
		}
	}
}

func TestWatchMultipleError(t *testing.T) {
	_, err := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		NamespaceParam("test").DefaultNamespace().
		FilenameParam(false, "../../../examples/guestbook/redis-master-controller.yaml").Flatten().
		FilenameParam(false, "../../../examples/guestbook/redis-master-controller.yaml").Flatten().
		Do().Watch("")

	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestLatest(t *testing.T) {
	r, _, _ := streamTestData()
	newPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "13"},
	}
	newPod2 := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "14"},
	}
	newSvc := &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "15"},
	}

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClientWith("", t, map[string]string{
		"/namespaces/test/pods/foo":     runtime.EncodeOrDie(testapi.Default.Codec(), newPod),
		"/namespaces/test/pods/bar":     runtime.EncodeOrDie(testapi.Default.Codec(), newPod2),
		"/namespaces/test/services/baz": runtime.EncodeOrDie(testapi.Default.Codec(), newSvc),
	})).
		NamespaceParam("other").Stream(r, "STDIN").Flatten().Latest()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err != nil || singular || len(test.Infos) != 3 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}
	if !api.Semantic.DeepDerivative([]runtime.Object{newPod, newPod2, newSvc}, test.Objects()) {
		t.Errorf("unexpected visited objects: %#v", test.Objects())
	}
}

func TestReceiveMultipleErrors(t *testing.T) {
	pods, svc := testData()

	r, w := io.Pipe()
	go func() {
		defer w.Close()
		w.Write([]byte(`{}`))
		w.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), &pods.Items[0])))
	}()

	r2, w2 := io.Pipe()
	go func() {
		defer w2.Close()
		w2.Write([]byte(`{}`))
		w2.Write([]byte(runtime.EncodeOrDie(testapi.Default.Codec(), &svc.Items[0])))
	}()

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient()).
		Stream(r, "1").Stream(r2, "2").
		ContinueOnError()

	test := &testVisitor{}
	singular := false

	err := b.Do().IntoSingular(&singular).Visit(test.Handle)
	if err == nil || singular || len(test.Infos) != 2 {
		t.Fatalf("unexpected response: %v %t %#v", err, singular, test.Infos)
	}

	errs, ok := err.(errors.Aggregate)
	if !ok {
		t.Fatalf("unexpected error: %v", reflect.TypeOf(err))
	}
	if len(errs.Errors()) != 2 {
		t.Errorf("unexpected errors %v", errs)
	}
}

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "rc,svc,pods,pvc",
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "rc,svc,pods,pvc,secrets",
		},
	}

	b := NewBuilder(testapi.Default.RESTMapper(), api.Scheme, fakeClient())

	for _, test := range tests {
		replaced := b.replaceAliases(test.arg)
		if replaced != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, replaced)
		}
	}
}
