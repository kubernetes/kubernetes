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

package resource

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/rest/fake"

	// TODO we need to remove this linkage and create our own scheme
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
)

func objBody(obj runtime.Object) io.ReadCloser {
	return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(corev1Codec, obj))))
}

func header() http.Header {
	header := http.Header{}
	header.Set("Content-Type", runtime.ContentTypeJSON)
	return header
}

// splitPath returns the segments for a URL path.
func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

// V1DeepEqualSafePodSpec returns a PodSpec which is ready to be used with apiequality.Semantic.DeepEqual
func V1DeepEqualSafePodSpec() corev1.PodSpec {
	grace := int64(30)
	return corev1.PodSpec{
		RestartPolicy:                 corev1.RestartPolicyAlways,
		DNSPolicy:                     corev1.DNSClusterFirst,
		TerminationGracePeriodSeconds: &grace,
		SecurityContext:               &corev1.PodSecurityContext{},
	}
}

func TestHelperDelete(t *testing.T) {
	tests := []struct {
		Err     bool
		Req     func(*http.Request) bool
		Resp    *http.Response
		HttpErr error
	}{
		{
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusSuccess}),
			},
			Req: func(req *http.Request) bool {
				if req.Method != "DELETE" {
					t.Errorf("unexpected method: %#v", req)
					return false
				}
				parts := splitPath(req.URL.Path)
				if len(parts) < 3 {
					t.Errorf("expected URL path to have 3 parts: %s", req.URL.Path)
					return false
				}
				if parts[1] != "bar" {
					t.Errorf("url doesn't contain namespace: %#v", req)
					return false
				}
				if parts[2] != "foo" {
					t.Errorf("url doesn't contain name: %#v", req)
					return false
				}
				return true
			},
		},
	}
	for _, test := range tests {
		client := &fake.RESTClient{
			NegotiatedSerializer: scheme.Codecs,
			Resp:                 test.Resp,
			Err:                  test.HttpErr,
		}
		modifier := &Helper{
			RESTClient:      client,
			NamespaceScoped: true,
		}
		_, err := modifier.Delete("bar", "foo")
		if (err != nil) != test.Err {
			t.Errorf("unexpected error: %t %v", test.Err, err)
		}
		if err != nil {
			continue
		}
		if test.Req != nil && !test.Req(client.Req) {
			t.Errorf("unexpected request: %#v", client.Req)
		}
	}
}

func TestHelperCreate(t *testing.T) {
	expectPost := func(req *http.Request) bool {
		if req.Method != "POST" {
			t.Errorf("unexpected method: %#v", req)
			return false
		}
		parts := splitPath(req.URL.Path)
		if parts[1] != "bar" {
			t.Errorf("url doesn't contain namespace: %#v", req)
			return false
		}
		return true
	}

	tests := []struct {
		Resp    *http.Response
		HttpErr error
		Modify  bool
		Object  runtime.Object

		ExpectObject runtime.Object
		Err          bool
		Req          func(*http.Request) bool
	}{
		{
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusSuccess}),
			},
			Object:       &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			ExpectObject: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			Req:          expectPost,
		},
		{
			Modify:       false,
			Object:       &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			ExpectObject: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			Resp:         &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})},
			Req:          expectPost,
		},
		{
			Modify: true,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
			ExpectObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
			Resp: &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})},
			Req:  expectPost,
		},
	}
	for i, test := range tests {
		client := &fake.RESTClient{
			GroupVersion:         corev1GV,
			NegotiatedSerializer: scheme.Codecs,
			Resp:                 test.Resp,
			Err:                  test.HttpErr,
		}
		modifier := &Helper{
			RESTClient:      client,
			NamespaceScoped: true,
		}
		_, err := modifier.Create("bar", test.Modify, test.Object)
		if (err != nil) != test.Err {
			t.Errorf("%d: unexpected error: %t %v", i, test.Err, err)
		}
		if err != nil {
			continue
		}
		if test.Req != nil && !test.Req(client.Req) {
			t.Errorf("%d: unexpected request: %#v", i, client.Req)
		}
		body, err := ioutil.ReadAll(client.Req.Body)
		if err != nil {
			t.Fatalf("%d: unexpected error: %#v", i, err)
		}
		t.Logf("got body: %s", string(body))
		expect := []byte{}
		if test.ExpectObject != nil {
			expect = []byte(runtime.EncodeOrDie(corev1Codec, test.ExpectObject))
		}
		if !reflect.DeepEqual(expect, body) {
			t.Errorf("%d: unexpected body: %s (expected %s)", i, string(body), string(expect))
		}

	}
}

func TestHelperGet(t *testing.T) {
	tests := []struct {
		Err     bool
		Req     func(*http.Request) bool
		Resp    *http.Response
		HttpErr error
	}{
		{
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     header(),
				Body:       objBody(&corev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "foo"}}),
			},
			Req: func(req *http.Request) bool {
				if req.Method != "GET" {
					t.Errorf("unexpected method: %#v", req)
					return false
				}
				parts := splitPath(req.URL.Path)
				if parts[1] != "bar" {
					t.Errorf("url doesn't contain namespace: %#v", req)
					return false
				}
				if parts[2] != "foo" {
					t.Errorf("url doesn't contain name: %#v", req)
					return false
				}
				return true
			},
		},
	}
	for i, test := range tests {
		client := &fake.RESTClient{
			GroupVersion:         corev1GV,
			NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
			Resp:                 test.Resp,
			Err:                  test.HttpErr,
		}
		modifier := &Helper{
			RESTClient:      client,
			NamespaceScoped: true,
		}
		obj, err := modifier.Get("bar", "foo", false)

		if (err != nil) != test.Err {
			t.Errorf("unexpected error: %d %t %v", i, test.Err, err)
		}
		if err != nil {
			continue
		}
		if obj.(*corev1.Pod).Name != "foo" {
			t.Errorf("unexpected object: %#v", obj)
		}
		if test.Req != nil && !test.Req(client.Req) {
			t.Errorf("unexpected request: %#v", client.Req)
		}
	}
}

func TestHelperList(t *testing.T) {
	tests := []struct {
		Err     bool
		Req     func(*http.Request) bool
		Resp    *http.Response
		HttpErr error
	}{
		{
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			Resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     header(),
				Body: objBody(&corev1.PodList{
					Items: []corev1.Pod{{
						ObjectMeta: metav1.ObjectMeta{Name: "foo"},
					},
					},
				}),
			},
			Req: func(req *http.Request) bool {
				if req.Method != "GET" {
					t.Errorf("unexpected method: %#v", req)
					return false
				}
				if req.URL.Path != "/namespaces/bar" {
					t.Errorf("url doesn't contain name: %#v", req.URL)
					return false
				}
				if req.URL.Query().Get(metav1.LabelSelectorQueryParam(corev1GV.String())) != labels.SelectorFromSet(labels.Set{"foo": "baz"}).String() {
					t.Errorf("url doesn't contain query parameters: %#v", req.URL)
					return false
				}
				return true
			},
		},
	}
	for _, test := range tests {
		client := &fake.RESTClient{
			GroupVersion:         corev1GV,
			NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
			Resp:                 test.Resp,
			Err:                  test.HttpErr,
		}
		modifier := &Helper{
			RESTClient:      client,
			NamespaceScoped: true,
		}
		obj, err := modifier.List("bar", corev1GV.String(), false, &metav1.ListOptions{LabelSelector: "foo=baz"})
		if (err != nil) != test.Err {
			t.Errorf("unexpected error: %t %v", test.Err, err)
		}
		if err != nil {
			continue
		}
		if obj.(*corev1.PodList).Items[0].Name != "foo" {
			t.Errorf("unexpected object: %#v", obj)
		}
		if test.Req != nil && !test.Req(client.Req) {
			t.Errorf("unexpected request: %#v", client.Req)
		}
	}
}

func TestHelperListSelectorCombination(t *testing.T) {
	tests := []struct {
		Name          string
		Err           bool
		ErrMsg        string
		FieldSelector string
		LabelSelector string
	}{
		{
			Name: "No selector",
			Err:  false,
		},
		{
			Name:          "Only Label Selector",
			Err:           false,
			LabelSelector: "foo=baz",
		},
		{
			Name:          "Only Field Selector",
			Err:           false,
			FieldSelector: "xyz=zyx",
		},
		{
			Name:          "Both Label and Field Selector",
			Err:           false,
			LabelSelector: "foo=baz",
			FieldSelector: "xyz=zyx",
		},
	}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     header(),
		Body: objBody(&corev1.PodList{
			Items: []corev1.Pod{{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			},
		}),
	}
	client := &fake.RESTClient{
		NegotiatedSerializer: scheme.Codecs,
		Resp:                 resp,
		Err:                  nil,
	}
	modifier := &Helper{
		RESTClient:      client,
		NamespaceScoped: true,
	}

	for _, test := range tests {
		_, err := modifier.List("bar",
			corev1GV.String(),
			false,
			&metav1.ListOptions{LabelSelector: test.LabelSelector, FieldSelector: test.FieldSelector})
		if test.Err {
			if err == nil {
				t.Errorf("%q expected error: %q", test.Name, test.ErrMsg)
			}
			if err != nil && err.Error() != test.ErrMsg {
				t.Errorf("%q expected error: %q", test.Name, test.ErrMsg)
			}
		}
	}
}

func TestHelperReplace(t *testing.T) {
	expectPut := func(path string, req *http.Request) bool {
		if req.Method != "PUT" {
			t.Errorf("unexpected method: %#v", req)
			return false
		}
		if req.URL.Path != path {
			t.Errorf("unexpected url: %v", req.URL)
			return false
		}
		return true
	}

	tests := []struct {
		Resp            *http.Response
		HTTPClient      *http.Client
		HttpErr         error
		Overwrite       bool
		Object          runtime.Object
		Namespace       string
		NamespaceScoped bool

		ExpectPath   string
		ExpectObject runtime.Object
		Err          bool
		Req          func(string, *http.Request) bool
	}{
		{
			Namespace:       "bar",
			NamespaceScoped: true,
			HttpErr:         errors.New("failure"),
			Err:             true,
		},
		{
			Namespace:       "bar",
			NamespaceScoped: true,
			Object:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			Namespace:       "bar",
			NamespaceScoped: true,
			Object:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			ExpectPath:      "/namespaces/bar/foo",
			ExpectObject:    &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			Resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusSuccess}),
			},
			Req: expectPut,
		},
		// namespace scoped resource
		{
			Namespace:       "bar",
			NamespaceScoped: true,
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
			ExpectPath: "/namespaces/bar/foo",
			ExpectObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
				Spec:       V1DeepEqualSafePodSpec(),
			},
			Overwrite: true,
			HTTPClient: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if req.Method == "PUT" {
					return &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})}, nil
				}
				return &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}})}, nil
			}),
			Req: expectPut,
		},
		// cluster scoped resource
		{
			Object: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			ExpectObject: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
			},
			Overwrite:  true,
			ExpectPath: "/foo",
			HTTPClient: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if req.Method == "PUT" {
					return &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})}, nil
				}
				return &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}})}, nil
			}),
			Req: expectPut,
		},
		{
			Namespace:       "bar",
			NamespaceScoped: true,
			Object:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			ExpectPath:      "/namespaces/bar/foo",
			ExpectObject:    &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			Resp:            &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})},
			Req:             expectPut,
		},
	}
	for i, test := range tests {
		client := &fake.RESTClient{
			GroupVersion:         corev1GV,
			NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
			Client:               test.HTTPClient,
			Resp:                 test.Resp,
			Err:                  test.HttpErr,
		}
		modifier := &Helper{
			RESTClient:      client,
			NamespaceScoped: test.NamespaceScoped,
		}
		_, err := modifier.Replace(test.Namespace, "foo", test.Overwrite, test.Object)
		if (err != nil) != test.Err {
			t.Errorf("%d: unexpected error: %t %v", i, test.Err, err)
		}
		if err != nil {
			continue
		}
		if test.Req != nil && !test.Req(test.ExpectPath, client.Req) {
			t.Errorf("%d: unexpected request: %#v", i, client.Req)
		}
		body, err := ioutil.ReadAll(client.Req.Body)
		if err != nil {
			t.Fatalf("%d: unexpected error: %#v", i, err)
		}
		expect := []byte{}
		if test.ExpectObject != nil {
			expect = []byte(runtime.EncodeOrDie(corev1Codec, test.ExpectObject))
		}
		if !reflect.DeepEqual(expect, body) {
			t.Errorf("%d: unexpected body: %s", i, string(body))
		}
	}
}
