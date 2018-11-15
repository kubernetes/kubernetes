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

package rest

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
	restclientwatch "k8s.io/client-go/rest/watch"
	"k8s.io/client-go/util/flowcontrol"
	utiltesting "k8s.io/client-go/util/testing"
)

func TestNewRequestSetsAccept(t *testing.T) {
	r := NewRequest(nil, "get", &url.URL{Path: "/path/"}, "", ContentConfig{}, Serializers{}, nil, nil, 0)
	if r.headers.Get("Accept") != "" {
		t.Errorf("unexpected headers: %#v", r.headers)
	}
	r = NewRequest(nil, "get", &url.URL{Path: "/path/"}, "", ContentConfig{ContentType: "application/other"}, Serializers{}, nil, nil, 0)
	if r.headers.Get("Accept") != "application/other, */*" {
		t.Errorf("unexpected headers: %#v", r.headers)
	}
}

type clientFunc func(req *http.Request) (*http.Response, error)

func (f clientFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestRequestSetsHeaders(t *testing.T) {
	server := clientFunc(func(req *http.Request) (*http.Response, error) {
		if req.Header.Get("Accept") != "application/other, */*" {
			t.Errorf("unexpected headers: %#v", req.Header)
		}
		return &http.Response{
			StatusCode: http.StatusForbidden,
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}, nil
	})
	config := defaultContentConfig()
	config.ContentType = "application/other"
	serializers := defaultSerializers(t)
	r := NewRequest(server, "get", &url.URL{Path: "/path"}, "", config, serializers, nil, nil, 0)

	// Check if all "issue" methods are setting headers.
	_ = r.Do()
	_, _ = r.Watch()
	_, _ = r.Stream()
}

func TestRequestWithErrorWontChange(t *testing.T) {
	gvCopy := v1.SchemeGroupVersion
	original := Request{
		err:     errors.New("test"),
		content: ContentConfig{GroupVersion: &gvCopy},
	}
	r := original
	changed := r.Param("foo", "bar").
		AbsPath("/abs").
		Prefix("test").
		Suffix("testing").
		Namespace("new").
		Resource("foos").
		Name("bars").
		Body("foo").
		Timeout(time.Millisecond)
	if changed != &r {
		t.Errorf("returned request should point to the same object")
	}
	if !reflect.DeepEqual(changed, &original) {
		t.Errorf("expected %#v, got %#v", &original, changed)
	}
}

func TestRequestPreservesBaseTrailingSlash(t *testing.T) {
	r := &Request{baseURL: &url.URL{}, pathPrefix: "/path/"}
	if s := r.URL().String(); s != "/path/" {
		t.Errorf("trailing slash should be preserved: %s", s)
	}
}

func TestRequestAbsPathPreservesTrailingSlash(t *testing.T) {
	r := (&Request{baseURL: &url.URL{}}).AbsPath("/foo/")
	if s := r.URL().String(); s != "/foo/" {
		t.Errorf("trailing slash should be preserved: %s", s)
	}

	r = (&Request{baseURL: &url.URL{}}).AbsPath("/foo/")
	if s := r.URL().String(); s != "/foo/" {
		t.Errorf("trailing slash should be preserved: %s", s)
	}
}

func TestRequestAbsPathJoins(t *testing.T) {
	r := (&Request{baseURL: &url.URL{}}).AbsPath("foo/bar", "baz")
	if s := r.URL().String(); s != "foo/bar/baz" {
		t.Errorf("trailing slash should be preserved: %s", s)
	}
}

func TestRequestSetsNamespace(t *testing.T) {
	r := (&Request{
		baseURL: &url.URL{
			Path: "/",
		},
	}).Namespace("foo")
	if r.namespace == "" {
		t.Errorf("namespace should be set: %#v", r)
	}

	if s := r.URL().String(); s != "namespaces/foo" {
		t.Errorf("namespace should be in path: %s", s)
	}
}

func TestRequestOrdersNamespaceInPath(t *testing.T) {
	r := (&Request{
		baseURL:    &url.URL{},
		pathPrefix: "/test/",
	}).Name("bar").Resource("baz").Namespace("foo")
	if s := r.URL().String(); s != "/test/namespaces/foo/baz/bar" {
		t.Errorf("namespace should be in order in path: %s", s)
	}
}

func TestRequestOrdersSubResource(t *testing.T) {
	r := (&Request{
		baseURL:    &url.URL{},
		pathPrefix: "/test/",
	}).Name("bar").Resource("baz").Namespace("foo").Suffix("test").SubResource("a", "b")
	if s := r.URL().String(); s != "/test/namespaces/foo/baz/bar/a/b/test" {
		t.Errorf("namespace should be in order in path: %s", s)
	}
}

func TestRequestSetTwiceError(t *testing.T) {
	if (&Request{}).Name("bar").Name("baz").err == nil {
		t.Errorf("setting name twice should result in error")
	}
	if (&Request{}).Namespace("bar").Namespace("baz").err == nil {
		t.Errorf("setting namespace twice should result in error")
	}
	if (&Request{}).Resource("bar").Resource("baz").err == nil {
		t.Errorf("setting resource twice should result in error")
	}
	if (&Request{}).SubResource("bar").SubResource("baz").err == nil {
		t.Errorf("setting subresource twice should result in error")
	}
}

func TestInvalidSegments(t *testing.T) {
	invalidSegments := []string{".", "..", "test/segment", "test%2bsegment"}
	setters := map[string]func(string, *Request){
		"namespace":   func(s string, r *Request) { r.Namespace(s) },
		"resource":    func(s string, r *Request) { r.Resource(s) },
		"name":        func(s string, r *Request) { r.Name(s) },
		"subresource": func(s string, r *Request) { r.SubResource(s) },
	}
	for _, invalidSegment := range invalidSegments {
		for setterName, setter := range setters {
			r := &Request{}
			setter(invalidSegment, r)
			if r.err == nil {
				t.Errorf("%s: %s: expected error, got none", setterName, invalidSegment)
			}
		}
	}
}

func TestRequestParam(t *testing.T) {
	r := (&Request{}).Param("foo", "a")
	if !reflect.DeepEqual(r.params, url.Values{"foo": []string{"a"}}) {
		t.Errorf("should have set a param: %#v", r)
	}

	r.Param("bar", "1")
	r.Param("bar", "2")
	if !reflect.DeepEqual(r.params, url.Values{"foo": []string{"a"}, "bar": []string{"1", "2"}}) {
		t.Errorf("should have set a param: %#v", r)
	}
}

func TestRequestVersionedParams(t *testing.T) {
	r := (&Request{content: ContentConfig{GroupVersion: &v1.SchemeGroupVersion}}).Param("foo", "a")
	if !reflect.DeepEqual(r.params, url.Values{"foo": []string{"a"}}) {
		t.Errorf("should have set a param: %#v", r)
	}
	r.VersionedParams(&v1.PodLogOptions{Follow: true, Container: "bar"}, scheme.ParameterCodec)

	if !reflect.DeepEqual(r.params, url.Values{
		"foo":       []string{"a"},
		"container": []string{"bar"},
		"follow":    []string{"true"},
	}) {
		t.Errorf("should have set a param: %#v", r)
	}
}

func TestRequestVersionedParamsFromListOptions(t *testing.T) {
	r := &Request{content: ContentConfig{GroupVersion: &v1.SchemeGroupVersion}}
	r.VersionedParams(&metav1.ListOptions{ResourceVersion: "1"}, scheme.ParameterCodec)
	if !reflect.DeepEqual(r.params, url.Values{
		"resourceVersion": []string{"1"},
	}) {
		t.Errorf("should have set a param: %#v", r)
	}

	var timeout int64 = 10
	r.VersionedParams(&metav1.ListOptions{ResourceVersion: "2", TimeoutSeconds: &timeout}, scheme.ParameterCodec)
	if !reflect.DeepEqual(r.params, url.Values{
		"resourceVersion": []string{"1", "2"},
		"timeoutSeconds":  []string{"10"},
	}) {
		t.Errorf("should have set a param: %#v %v", r.params, r.err)
	}
}

func TestRequestURI(t *testing.T) {
	r := (&Request{}).Param("foo", "a")
	r.Prefix("other")
	r.RequestURI("/test?foo=b&a=b&c=1&c=2")
	if r.pathPrefix != "/test" {
		t.Errorf("path is wrong: %#v", r)
	}
	if !reflect.DeepEqual(r.params, url.Values{"a": []string{"b"}, "foo": []string{"b"}, "c": []string{"1", "2"}}) {
		t.Errorf("should have set a param: %#v", r)
	}
}

type NotAnAPIObject struct{}

func (obj NotAnAPIObject) GroupVersionKind() *schema.GroupVersionKind       { return nil }
func (obj NotAnAPIObject) SetGroupVersionKind(gvk *schema.GroupVersionKind) {}

func defaultContentConfig() ContentConfig {
	gvCopy := v1.SchemeGroupVersion
	return ContentConfig{
		ContentType:          "application/json",
		GroupVersion:         &gvCopy,
		NegotiatedSerializer: serializer.DirectCodecFactory{CodecFactory: scheme.Codecs},
	}
}

func defaultSerializers(t *testing.T) Serializers {
	config := defaultContentConfig()
	serializers, err := createSerializers(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return *serializers
}

func TestRequestBody(t *testing.T) {
	// test unknown type
	r := (&Request{}).Body([]string{"test"})
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}

	// test error set when failing to read file
	f, err := ioutil.TempFile("", "test")
	if err != nil {
		t.Fatalf("unable to create temp file")
	}
	defer f.Close()
	os.Remove(f.Name())
	r = (&Request{}).Body(f.Name())
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}

	// test unencodable api object
	r = (&Request{content: defaultContentConfig()}).Body(&NotAnAPIObject{})
	if r.err == nil || r.body != nil {
		t.Errorf("should have set err and left body nil: %#v", r)
	}
}

func TestResultIntoWithErrReturnsErr(t *testing.T) {
	res := Result{err: errors.New("test")}
	if err := res.Into(&v1.Pod{}); err != res.err {
		t.Errorf("should have returned exact error from result")
	}
}

func TestResultIntoWithNoBodyReturnsErr(t *testing.T) {
	res := Result{
		body:    []byte{},
		decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),
	}
	if err := res.Into(&v1.Pod{}); err == nil || !strings.Contains(err.Error(), "0-length") {
		t.Errorf("should have complained about 0 length body")
	}
}

func TestURLTemplate(t *testing.T) {
	uri, _ := url.Parse("http://localhost/some/base/url/path")
	testCases := []struct {
		Request          *Request
		ExpectedFullURL  string
		ExpectedFinalURL string
	}{
		{
			// non dynamic client
			Request: NewRequest(nil, "POST", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("api", "v1").Resource("r1").Namespace("ns").Name("nm").Param("p0", "v0"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/api/v1/namespaces/ns/r1/nm?p0=v0",
			ExpectedFinalURL: "http://localhost/some/base/url/path/api/v1/namespaces/%7Bnamespace%7D/r1/%7Bname%7D?p0=%7Bvalue%7D",
		},
		{
			// non dynamic client with wrong api group
			Request: NewRequest(nil, "POST", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("pre1", "v1").Resource("r1").Namespace("ns").Name("nm").Param("p0", "v0"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/pre1/v1/namespaces/ns/r1/nm?p0=v0",
			ExpectedFinalURL: "http://localhost/%7Bprefix%7D",
		},
		{
			// dynamic client with core group + namespace + resourceResource (with name)
			// /api/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/api/v1/namespaces/ns/r1/name1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/api/v1/namespaces/ns/r1/name1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/api/v1/namespaces/%7Bnamespace%7D/r1/%7Bname%7D",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/g1/v1/namespaces/ns/r1/name1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/g1/v1/namespaces/ns/r1/name1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/g1/v1/namespaces/%7Bnamespace%7D/r1/%7Bname%7D",
		},
		{
			// dynamic client with core group + namespace + resourceResource (with NO name)
			// /api/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/api/v1/namespaces/ns/r1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/api/v1/namespaces/ns/r1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/api/v1/namespaces/%7Bnamespace%7D/r1",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with NO name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/g1/v1/namespaces/ns/r1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/g1/v1/namespaces/ns/r1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/g1/v1/namespaces/%7Bnamespace%7D/r1",
		},
		{
			// dynamic client with core group + resourceResource (with name)
			// /api/$RESOURCEVERSION/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/api/v1/r1/name1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/api/v1/r1/name1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/api/v1/r1/%7Bname%7D",
		},
		{
			// dynamic client with named group + resourceResource (with name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/g1/v1/r1/name1"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/g1/v1/r1/name1",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/g1/v1/r1/%7Bname%7D",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME/$SUBRESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces/finalize"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces/finalize",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bnamespace%7D/namespaces/%7Bname%7D/finalize",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bnamespace%7D/namespaces/%7Bname%7D",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with NO name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%SUBRESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/namespaces/finalize"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/namespaces/finalize",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bnamespace%7D/namespaces/finalize",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with NO name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%SUBRESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/namespaces/status"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/namespaces/status",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bnamespace%7D/namespaces/status",
		},
		{
			// dynamic client with named group + namespace + resourceResource (with no name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/namespaces"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/namespaces",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bnamespace%7D/namespaces",
		},
		{
			// dynamic client with named group + resourceResource (with name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/finalize"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/finalize",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bname%7D/finalize",
		},
		{
			// dynamic client with named group + resourceResource (with name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces/status"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces/status",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bname%7D/status",
		},
		{
			// dynamic client with named group + resourceResource (with name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces/namespaces"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/namespaces",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces/%7Bname%7D",
		},
		{
			// dynamic client with named group + resourceResource (with no name)
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/$RESOURCE/%NAME
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/apis/namespaces/namespaces/namespaces"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces",
			ExpectedFinalURL: "http://localhost/some/base/url/path/apis/namespaces/namespaces/namespaces",
		},
		{
			// dynamic client with wrong api group + namespace + resourceResource (with name) + subresource
			// /apis/$NAMEDGROUPNAME/$RESOURCEVERSION/namespaces/$NAMESPACE/$RESOURCE/%NAME/$SUBRESOURCE
			Request: NewRequest(nil, "DELETE", uri, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).
				Prefix("/pre1/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces/finalize"),
			ExpectedFullURL:  "http://localhost/some/base/url/path/pre1/namespaces/namespaces/namespaces/namespaces/namespaces/namespaces/finalize",
			ExpectedFinalURL: "http://localhost/%7Bprefix%7D",
		},
	}
	for i, testCase := range testCases {
		r := testCase.Request
		full := r.URL()
		if full.String() != testCase.ExpectedFullURL {
			t.Errorf("%d: unexpected initial URL: %s %s", i, full, testCase.ExpectedFullURL)
		}
		actualURL := r.finalURLTemplate()
		actual := actualURL.String()
		if actual != testCase.ExpectedFinalURL {
			t.Errorf("%d: unexpected URL template: %s %s", i, actual, testCase.ExpectedFinalURL)
		}
		if r.URL().String() != full.String() {
			t.Errorf("%d, creating URL template changed request: %s -> %s", i, full.String(), r.URL().String())
		}
	}
}

func TestTransformResponse(t *testing.T) {
	invalid := []byte("aaaaa")
	uri, _ := url.Parse("http://localhost")
	testCases := []struct {
		Response *http.Response
		Data     []byte
		Created  bool
		Error    bool
		ErrFn    func(err error) bool
	}{
		{Response: &http.Response{StatusCode: 200}, Data: []byte{}},
		{Response: &http.Response{StatusCode: 201}, Data: []byte{}, Created: true},
		{Response: &http.Response{StatusCode: 199}, Error: true},
		{Response: &http.Response{StatusCode: 500}, Error: true},
		{Response: &http.Response{StatusCode: 422}, Error: true},
		{Response: &http.Response{StatusCode: 409}, Error: true},
		{Response: &http.Response{StatusCode: 404}, Error: true},
		{Response: &http.Response{StatusCode: 401}, Error: true},
		{
			Response: &http.Response{
				StatusCode: 401,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Error: true,
			ErrFn: func(err error) bool {
				return err.Error() != "aaaaa" && apierrors.IsUnauthorized(err)
			},
		},
		{
			Response: &http.Response{
				StatusCode: 401,
				Header:     http.Header{"Content-Type": []string{"text/any"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Error: true,
			ErrFn: func(err error) bool {
				return strings.Contains(err.Error(), "server has asked for the client to provide") && apierrors.IsUnauthorized(err)
			},
		},
		{Response: &http.Response{StatusCode: 403}, Error: true},
		{Response: &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(invalid))}, Data: invalid},
		{Response: &http.Response{StatusCode: 200, Body: ioutil.NopCloser(bytes.NewReader(invalid))}, Data: invalid},
	}
	for i, test := range testCases {
		r := NewRequest(nil, "", uri, "", defaultContentConfig(), defaultSerializers(t), nil, nil, 0)
		if test.Response.Body == nil {
			test.Response.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))
		}
		result := r.transformResponse(test.Response, &http.Request{})
		response, created, err := result.body, result.statusCode == http.StatusCreated, result.err
		hasErr := err != nil
		if hasErr != test.Error {
			t.Errorf("%d: unexpected error: %t %v", i, test.Error, err)
		} else if hasErr && test.Response.StatusCode > 399 {
			status, ok := err.(apierrors.APIStatus)
			if !ok {
				t.Errorf("%d: response should have been transformable into APIStatus: %v", i, err)
				continue
			}
			if int(status.Status().Code) != test.Response.StatusCode {
				t.Errorf("%d: status code did not match response: %#v", i, status.Status())
			}
		}
		if test.ErrFn != nil && !test.ErrFn(err) {
			t.Errorf("%d: error function did not match: %v", i, err)
		}
		if !(test.Data == nil && response == nil) && !apiequality.Semantic.DeepDerivative(test.Data, response) {
			t.Errorf("%d: unexpected response: %#v %#v", i, test.Data, response)
		}
		if test.Created != created {
			t.Errorf("%d: expected created %t, got %t", i, test.Created, created)
		}
	}
}

type renegotiator struct {
	called      bool
	contentType string
	params      map[string]string
	decoder     runtime.Decoder
	err         error
}

func (r *renegotiator) invoke(contentType string, params map[string]string) (runtime.Decoder, error) {
	r.called = true
	r.contentType = contentType
	r.params = params
	return r.decoder, r.err
}

func TestTransformResponseNegotiate(t *testing.T) {
	invalid := []byte("aaaaa")
	uri, _ := url.Parse("http://localhost")
	testCases := []struct {
		Response *http.Response
		Data     []byte
		Created  bool
		Error    bool
		ErrFn    func(err error) bool

		ContentType       string
		Called            bool
		ExpectContentType string
		Decoder           runtime.Decoder
		NegotiateErr      error
	}{
		{
			ContentType: "application/json",
			Response: &http.Response{
				StatusCode: 401,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Error: true,
			ErrFn: func(err error) bool {
				return err.Error() != "aaaaa" && apierrors.IsUnauthorized(err)
			},
		},
		{
			ContentType: "application/json",
			Response: &http.Response{
				StatusCode: 401,
				Header:     http.Header{"Content-Type": []string{"application/protobuf"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),

			Called:            true,
			ExpectContentType: "application/protobuf",

			Error: true,
			ErrFn: func(err error) bool {
				return err.Error() != "aaaaa" && apierrors.IsUnauthorized(err)
			},
		},
		{
			ContentType: "application/json",
			Response: &http.Response{
				StatusCode: 500,
				Header:     http.Header{"Content-Type": []string{"application/,others"}},
			},
			Decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),

			Error: true,
			ErrFn: func(err error) bool {
				return err.Error() == "Internal error occurred: mime: expected token after slash" && err.(apierrors.APIStatus).Status().Code == 500
			},
		},
		{
			// no negotiation when no content type specified
			Response: &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": []string{"text/any"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),
		},
		{
			// no negotiation when no response content type specified
			ContentType: "text/any",
			Response: &http.Response{
				StatusCode: 200,
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),
		},
		{
			// unrecognized content type is not handled
			ContentType: "application/json",
			Response: &http.Response{
				StatusCode: 404,
				Header:     http.Header{"Content-Type": []string{"application/unrecognized"}},
				Body:       ioutil.NopCloser(bytes.NewReader(invalid)),
			},
			Decoder: scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion),

			NegotiateErr:      fmt.Errorf("aaaa"),
			Called:            true,
			ExpectContentType: "application/unrecognized",

			Error: true,
			ErrFn: func(err error) bool {
				return err.Error() != "aaaaa" && apierrors.IsNotFound(err)
			},
		},
	}
	for i, test := range testCases {
		serializers := defaultSerializers(t)
		negotiator := &renegotiator{
			decoder: test.Decoder,
			err:     test.NegotiateErr,
		}
		serializers.RenegotiatedDecoder = negotiator.invoke
		contentConfig := defaultContentConfig()
		contentConfig.ContentType = test.ContentType
		r := NewRequest(nil, "", uri, "", contentConfig, serializers, nil, nil, 0)
		if test.Response.Body == nil {
			test.Response.Body = ioutil.NopCloser(bytes.NewReader([]byte{}))
		}
		result := r.transformResponse(test.Response, &http.Request{})
		_, err := result.body, result.err
		hasErr := err != nil
		if hasErr != test.Error {
			t.Errorf("%d: unexpected error: %t %v", i, test.Error, err)
			continue
		} else if hasErr && test.Response.StatusCode > 399 {
			status, ok := err.(apierrors.APIStatus)
			if !ok {
				t.Errorf("%d: response should have been transformable into APIStatus: %v", i, err)
				continue
			}
			if int(status.Status().Code) != test.Response.StatusCode {
				t.Errorf("%d: status code did not match response: %#v", i, status.Status())
			}
		}
		if test.ErrFn != nil && !test.ErrFn(err) {
			t.Errorf("%d: error function did not match: %v", i, err)
		}
		if negotiator.called != test.Called {
			t.Errorf("%d: negotiator called %t != %t", i, negotiator.called, test.Called)
		}
		if !test.Called {
			continue
		}
		if negotiator.contentType != test.ExpectContentType {
			t.Errorf("%d: unexpected content type: %s", i, negotiator.contentType)
		}
	}
}

func TestTransformUnstructuredError(t *testing.T) {
	testCases := []struct {
		Req *http.Request
		Res *http.Response

		Resource string
		Name     string

		ErrFn       func(error) bool
		Transformed error
	}{
		{
			Resource: "foo",
			Name:     "bar",
			Req: &http.Request{
				Method: "POST",
			},
			Res: &http.Response{
				StatusCode: http.StatusConflict,
				Body:       ioutil.NopCloser(bytes.NewReader(nil)),
			},
			ErrFn: apierrors.IsAlreadyExists,
		},
		{
			Resource: "foo",
			Name:     "bar",
			Req: &http.Request{
				Method: "PUT",
			},
			Res: &http.Response{
				StatusCode: http.StatusConflict,
				Body:       ioutil.NopCloser(bytes.NewReader(nil)),
			},
			ErrFn: apierrors.IsConflict,
		},
		{
			Resource: "foo",
			Name:     "bar",
			Req:      &http.Request{},
			Res: &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       ioutil.NopCloser(bytes.NewReader(nil)),
			},
			ErrFn: apierrors.IsNotFound,
		},
		{
			Req: &http.Request{},
			Res: &http.Response{
				StatusCode: http.StatusBadRequest,
				Body:       ioutil.NopCloser(bytes.NewReader(nil)),
			},
			ErrFn: apierrors.IsBadRequest,
		},
		{
			// status in response overrides transformed result
			Req:   &http.Request{},
			Res:   &http.Response{StatusCode: http.StatusBadRequest, Body: ioutil.NopCloser(bytes.NewReader([]byte(`{"kind":"Status","apiVersion":"v1","status":"Failure","code":404}`)))},
			ErrFn: apierrors.IsBadRequest,
			Transformed: &apierrors.StatusError{
				ErrStatus: metav1.Status{Status: metav1.StatusFailure, Code: http.StatusNotFound},
			},
		},
		{
			// successful status is ignored
			Req:   &http.Request{},
			Res:   &http.Response{StatusCode: http.StatusBadRequest, Body: ioutil.NopCloser(bytes.NewReader([]byte(`{"kind":"Status","apiVersion":"v1","status":"Success","code":404}`)))},
			ErrFn: apierrors.IsBadRequest,
		},
		{
			// empty object does not change result
			Req:   &http.Request{},
			Res:   &http.Response{StatusCode: http.StatusBadRequest, Body: ioutil.NopCloser(bytes.NewReader([]byte(`{}`)))},
			ErrFn: apierrors.IsBadRequest,
		},
		{
			// we default apiVersion for backwards compatibility with old clients
			// TODO: potentially remove in 1.7
			Req:   &http.Request{},
			Res:   &http.Response{StatusCode: http.StatusBadRequest, Body: ioutil.NopCloser(bytes.NewReader([]byte(`{"kind":"Status","status":"Failure","code":404}`)))},
			ErrFn: apierrors.IsBadRequest,
			Transformed: &apierrors.StatusError{
				ErrStatus: metav1.Status{Status: metav1.StatusFailure, Code: http.StatusNotFound},
			},
		},
		{
			// we do not default kind
			Req:   &http.Request{},
			Res:   &http.Response{StatusCode: http.StatusBadRequest, Body: ioutil.NopCloser(bytes.NewReader([]byte(`{"status":"Failure","code":404}`)))},
			ErrFn: apierrors.IsBadRequest,
		},
	}

	for i, testCase := range testCases {
		r := &Request{
			content:      defaultContentConfig(),
			serializers:  defaultSerializers(t),
			resourceName: testCase.Name,
			resource:     testCase.Resource,
		}
		result := r.transformResponse(testCase.Res, testCase.Req)
		err := result.err
		if !testCase.ErrFn(err) {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !apierrors.IsUnexpectedServerError(err) {
			t.Errorf("%d: unexpected error type: %v", i, err)
		}
		if len(testCase.Name) != 0 && !strings.Contains(err.Error(), testCase.Name) {
			t.Errorf("unexpected error string: %s", err)
		}
		if len(testCase.Resource) != 0 && !strings.Contains(err.Error(), testCase.Resource) {
			t.Errorf("unexpected error string: %s", err)
		}

		// verify Error() properly transforms the error
		transformed := result.Error()
		expect := testCase.Transformed
		if expect == nil {
			expect = err
		}
		if !reflect.DeepEqual(expect, transformed) {
			t.Errorf("%d: unexpected Error(): %s", i, diff.ObjectReflectDiff(expect, transformed))
		}

		// verify result.Get properly transforms the error
		if _, err := result.Get(); !reflect.DeepEqual(expect, err) {
			t.Errorf("%d: unexpected error on Get(): %s", i, diff.ObjectReflectDiff(expect, err))
		}

		// verify result.Into properly handles the error
		if err := result.Into(&v1.Pod{}); !reflect.DeepEqual(expect, err) {
			t.Errorf("%d: unexpected error on Into(): %s", i, diff.ObjectReflectDiff(expect, err))
		}

		// verify result.Raw leaves the error in the untransformed state
		if _, err := result.Raw(); !reflect.DeepEqual(result.err, err) {
			t.Errorf("%d: unexpected error on Raw(): %s", i, diff.ObjectReflectDiff(expect, err))
		}
	}
}

func TestRequestWatch(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
		ErrFn   func(error) bool
		Empty   bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, pathPrefix: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
		{
			Request: &Request{
				content:     defaultContentConfig(),
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusForbidden,
						Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
					}, nil
				}),
				baseURL: &url.URL{},
			},
			Err: true,
			ErrFn: func(err error) bool {
				return apierrors.IsForbidden(err)
			},
		},
		{
			Request: &Request{
				content:     defaultContentConfig(),
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusUnauthorized,
						Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
					}, nil
				}),
				baseURL: &url.URL{},
			},
			Err: true,
			ErrFn: func(err error) bool {
				return apierrors.IsUnauthorized(err)
			},
		},
		{
			Request: &Request{
				content:     defaultContentConfig(),
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusUnauthorized,
						Body: ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &metav1.Status{
							Status: metav1.StatusFailure,
							Reason: metav1.StatusReasonUnauthorized,
						})))),
					}, nil
				}),
				baseURL: &url.URL{},
			},
			Err: true,
			ErrFn: func(err error) bool {
				return apierrors.IsUnauthorized(err)
			},
		},
		{
			Request: &Request{
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, io.EOF
				}),
				baseURL: &url.URL{},
			},
			Empty: true,
		},
		{
			Request: &Request{
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, &url.Error{Err: io.EOF}
				}),
				baseURL: &url.URL{},
			},
			Empty: true,
		},
		{
			Request: &Request{
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("http: can't write HTTP request on broken connection")
				}),
				baseURL: &url.URL{},
			},
			Empty: true,
		},
		{
			Request: &Request{
				serializers: defaultSerializers(t),
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("foo: connection reset by peer")
				}),
				baseURL: &url.URL{},
			},
			Empty: true,
		},
	}
	for i, testCase := range testCases {
		t.Logf("testcase %v", testCase.Request)
		testCase.Request.backoffMgr = &NoBackoff{}
		watch, err := testCase.Request.Watch()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %t, got %t: %v", i, testCase.Err, hasErr, err)
			continue
		}
		if testCase.ErrFn != nil && !testCase.ErrFn(err) {
			t.Errorf("%d: error not valid: %v", i, err)
		}
		if hasErr && watch != nil {
			t.Errorf("%d: watch should be nil when error is returned", i)
			continue
		}
		if testCase.Empty {
			_, ok := <-watch.ResultChan()
			if ok {
				t.Errorf("%d: expected the watch to be empty: %#v", i, watch)
			}
		}
	}
}

func TestRequestStream(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
		ErrFn   func(error) bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, pathPrefix: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusUnauthorized,
						Body: ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &metav1.Status{
							Status: metav1.StatusFailure,
							Reason: metav1.StatusReasonUnauthorized,
						})))),
					}, nil
				}),
				content:     defaultContentConfig(),
				serializers: defaultSerializers(t),
				baseURL:     &url.URL{},
			},
			Err: true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusBadRequest,
						Body:       ioutil.NopCloser(bytes.NewReader([]byte(`{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure","message":"a container name must be specified for pod kube-dns-v20-mz5cv, choose one of: [kubedns dnsmasq healthz]","reason":"BadRequest","code":400}`))),
					}, nil
				}),
				content:     defaultContentConfig(),
				serializers: defaultSerializers(t),
				baseURL:     &url.URL{},
			},
			Err: true,
			ErrFn: func(err error) bool {
				if err.Error() == "a container name must be specified for pod kube-dns-v20-mz5cv, choose one of: [kubedns dnsmasq healthz]" {
					return true
				}
				return false
			},
		},
	}
	for i, testCase := range testCases {
		testCase.Request.backoffMgr = &NoBackoff{}
		body, err := testCase.Request.Stream()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %t, got %t: %v", i, testCase.Err, hasErr, err)
		}
		if hasErr && body != nil {
			t.Errorf("%d: body should be nil when error is returned", i)
		}

		if hasErr {
			if testCase.ErrFn != nil && !testCase.ErrFn(err) {
				t.Errorf("unexpected error: %v", err)
			}
		}
	}
}

type fakeUpgradeConnection struct{}

func (c *fakeUpgradeConnection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	return nil, nil
}
func (c *fakeUpgradeConnection) Close() error {
	return nil
}
func (c *fakeUpgradeConnection) CloseChan() <-chan bool {
	return make(chan bool)
}
func (c *fakeUpgradeConnection) SetIdleTimeout(timeout time.Duration) {
}

type fakeUpgradeRoundTripper struct {
	req  *http.Request
	conn httpstream.Connection
}

func (f *fakeUpgradeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	f.req = req
	b := []byte{}
	body := ioutil.NopCloser(bytes.NewReader(b))
	resp := &http.Response{
		StatusCode: 101,
		Body:       body,
	}
	return resp, nil
}

func (f *fakeUpgradeRoundTripper) NewConnection(resp *http.Response) (httpstream.Connection, error) {
	return f.conn, nil
}

func TestRequestDo(t *testing.T) {
	testCases := []struct {
		Request *Request
		Err     bool
	}{
		{
			Request: &Request{err: errors.New("bail")},
			Err:     true,
		},
		{
			Request: &Request{baseURL: &url.URL{}, pathPrefix: "%"},
			Err:     true,
		},
		{
			Request: &Request{
				client: clientFunc(func(req *http.Request) (*http.Response, error) {
					return nil, errors.New("err")
				}),
				baseURL: &url.URL{},
			},
			Err: true,
		},
	}
	for i, testCase := range testCases {
		testCase.Request.backoffMgr = &NoBackoff{}
		body, err := testCase.Request.Do().Raw()
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Errorf("%d: expected %t, got %t: %v", i, testCase.Err, hasErr, err)
		}
		if hasErr && body != nil {
			t.Errorf("%d: body should be nil when error is returned", i)
		}
	}
}

func TestDoRequestNewWay(t *testing.T) {
	reqBody := "request body"
	expectedObj := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{
		Protocol:   "TCP",
		Port:       12345,
		TargetPort: intstr.FromInt(12345),
	}}}}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expectedObj)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := testRESTClient(t, testServer)
	obj, err := c.Verb("POST").
		Prefix("foo", "bar").
		Suffix("baz").
		Timeout(time.Second).
		Body([]byte(reqBody)).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !apiequality.Semantic.DeepDerivative(expectedObj, obj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	requestURL := defaultResourcePathWithPrefix("foo/bar", "", "", "baz")
	requestURL += "?timeout=1s"
	fakeHandler.ValidateRequest(t, requestURL, "POST", &reqBody)
}

// This test assumes that the client implementation backs off exponentially, for an individual request.
func TestBackoffLifecycle(t *testing.T) {
	count := 0
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		count++
		t.Logf("Attempt %d", count)
		if count == 5 || count == 9 {
			w.WriteHeader(http.StatusOK)
			return
		} else {
			w.WriteHeader(http.StatusGatewayTimeout)
			return
		}
	}))
	defer testServer.Close()
	c := testRESTClient(t, testServer)

	// Test backoff recovery and increase.  This correlates to the constants
	// which are used in the server implementation returning StatusOK above.
	seconds := []int{0, 1, 2, 4, 8, 0, 1, 2, 4, 0}
	request := c.Verb("POST").Prefix("backofftest").Suffix("abc")
	clock := clock.FakeClock{}
	request.backoffMgr = &URLBackoff{
		// Use a fake backoff here to avoid flakes and speed the test up.
		Backoff: flowcontrol.NewFakeBackOff(
			time.Duration(1)*time.Second,
			time.Duration(200)*time.Second,
			&clock,
		)}

	for _, sec := range seconds {
		thisBackoff := request.backoffMgr.CalculateBackoff(request.URL())
		t.Logf("Current backoff %v", thisBackoff)
		if thisBackoff != time.Duration(sec)*time.Second {
			t.Errorf("Backoff is %v instead of %v", thisBackoff, sec)
		}
		now := clock.Now()
		request.DoRaw()
		elapsed := clock.Since(now)
		if clock.Since(now) != thisBackoff {
			t.Errorf("CalculatedBackoff not honored by clock: Expected time of %v, but got %v ", thisBackoff, elapsed)
		}
	}
}

type testBackoffManager struct {
	sleeps []time.Duration
}

func (b *testBackoffManager) UpdateBackoff(actualUrl *url.URL, err error, responseCode int) {
}

func (b *testBackoffManager) CalculateBackoff(actualUrl *url.URL) time.Duration {
	return time.Duration(0)
}

func (b *testBackoffManager) Sleep(d time.Duration) {
	b.sleeps = append(b.sleeps, d)
}

func TestCheckRetryClosesBody(t *testing.T) {
	count := 0
	ch := make(chan struct{})
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		count++
		t.Logf("attempt %d", count)
		if count >= 5 {
			w.WriteHeader(http.StatusOK)
			close(ch)
			return
		}
		w.Header().Set("Retry-After", "1")
		http.Error(w, "Too many requests, please try again later.", http.StatusTooManyRequests)
	}))
	defer testServer.Close()

	backoffMgr := &testBackoffManager{}
	expectedSleeps := []time.Duration{0, time.Second, 0, time.Second, 0, time.Second, 0, time.Second, 0}

	c := testRESTClient(t, testServer)
	c.createBackoffMgr = func() BackoffManager { return backoffMgr }
	_, err := c.Verb("POST").
		Prefix("foo", "bar").
		Suffix("baz").
		Timeout(time.Second).
		Body([]byte(strings.Repeat("abcd", 1000))).
		DoRaw()
	if err != nil {
		t.Fatalf("Unexpected error: %v %#v", err, err)
	}
	<-ch
	if count != 5 {
		t.Errorf("unexpected retries: %d", count)
	}
	if !reflect.DeepEqual(backoffMgr.sleeps, expectedSleeps) {
		t.Errorf("unexpected sleeps, expected: %v, got: %v", expectedSleeps, backoffMgr.sleeps)
	}
}

func TestConnectionResetByPeerIsRetried(t *testing.T) {
	count := 0
	backoff := &testBackoffManager{}
	req := &Request{
		verb: "GET",
		client: clientFunc(func(req *http.Request) (*http.Response, error) {
			count++
			if count >= 3 {
				return &http.Response{
					StatusCode: 200,
					Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
				}, nil
			}
			return nil, &net.OpError{Err: syscall.ECONNRESET}
		}),
		backoffMgr: backoff,
	}
	// We expect two retries of "connection reset by peer" and the success.
	_, err := req.Do().Raw()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	// We have a sleep before each retry (including the initial one) and for
	// every "retry-after" call - thus 5 together.
	if len(backoff.sleeps) != 5 {
		t.Errorf("Expected 5 retries, got: %d", len(backoff.sleeps))
	}
}

func TestCheckRetryHandles429And5xx(t *testing.T) {
	count := 0
	ch := make(chan struct{})
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		data, err := ioutil.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("unable to read request body: %v", err)
		}
		if !bytes.Equal(data, []byte(strings.Repeat("abcd", 1000))) {
			t.Fatalf("retry did not send a complete body: %s", data)
		}
		t.Logf("attempt %d", count)
		if count >= 4 {
			w.WriteHeader(http.StatusOK)
			close(ch)
			return
		}
		w.Header().Set("Retry-After", "0")
		w.WriteHeader([]int{http.StatusTooManyRequests, 500, 501, 504}[count])
		count++
	}))
	defer testServer.Close()

	c := testRESTClient(t, testServer)
	_, err := c.Verb("POST").
		Prefix("foo", "bar").
		Suffix("baz").
		Timeout(time.Second).
		Body([]byte(strings.Repeat("abcd", 1000))).
		DoRaw()
	if err != nil {
		t.Fatalf("Unexpected error: %v %#v", err, err)
	}
	<-ch
	if count != 4 {
		t.Errorf("unexpected retries: %d", count)
	}
}

func BenchmarkCheckRetryClosesBody(b *testing.B) {
	count := 0
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		count++
		if count%3 == 0 {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer testServer.Close()

	c := testRESTClient(b, testServer)
	r := c.Verb("POST").
		Prefix("foo", "bar").
		Suffix("baz").
		Timeout(time.Second).
		Body([]byte(strings.Repeat("abcd", 1000)))

	for i := 0; i < b.N; i++ {
		if _, err := r.DoRaw(); err != nil {
			b.Fatalf("Unexpected error: %v %#v", err, err)
		}
	}
}

func TestDoRequestNewWayReader(t *testing.T) {
	reqObj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	reqBodyExpected, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), reqObj)
	expectedObj := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{
		Protocol:   "TCP",
		Port:       12345,
		TargetPort: intstr.FromInt(12345),
	}}}}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expectedObj)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := testRESTClient(t, testServer)
	obj, err := c.Verb("POST").
		Resource("bar").
		Name("baz").
		Prefix("foo").
		Timeout(time.Second).
		Body(bytes.NewBuffer(reqBodyExpected)).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !apiequality.Semantic.DeepDerivative(expectedObj, obj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	tmpStr := string(reqBodyExpected)
	requestURL := defaultResourcePathWithPrefix("foo", "bar", "", "baz")
	requestURL += "?timeout=1s"
	fakeHandler.ValidateRequest(t, requestURL, "POST", &tmpStr)
}

func TestDoRequestNewWayObj(t *testing.T) {
	reqObj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	reqBodyExpected, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), reqObj)
	expectedObj := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{
		Protocol:   "TCP",
		Port:       12345,
		TargetPort: intstr.FromInt(12345),
	}}}}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expectedObj)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := testRESTClient(t, testServer)
	obj, err := c.Verb("POST").
		Suffix("baz").
		Name("bar").
		Resource("foo").
		Timeout(time.Second).
		Body(reqObj).
		Do().Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !apiequality.Semantic.DeepDerivative(expectedObj, obj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	tmpStr := string(reqBodyExpected)
	requestURL := defaultResourcePathWithPrefix("", "foo", "", "bar/baz")
	requestURL += "?timeout=1s"
	fakeHandler.ValidateRequest(t, requestURL, "POST", &tmpStr)
}

func TestDoRequestNewWayFile(t *testing.T) {
	reqObj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	reqBodyExpected, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), reqObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	file, err := ioutil.TempFile("", "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer file.Close()
	defer os.Remove(file.Name())

	_, err = file.Write(reqBodyExpected)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedObj := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{
		Protocol:   "TCP",
		Port:       12345,
		TargetPort: intstr.FromInt(12345),
	}}}}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expectedObj)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := testRESTClient(t, testServer)
	wasCreated := true
	obj, err := c.Verb("POST").
		Prefix("foo/bar", "baz").
		Timeout(time.Second).
		Body(file.Name()).
		Do().WasCreated(&wasCreated).Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !apiequality.Semantic.DeepDerivative(expectedObj, obj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	if wasCreated {
		t.Errorf("expected object was created")
	}
	tmpStr := string(reqBodyExpected)
	requestURL := defaultResourcePathWithPrefix("foo/bar/baz", "", "", "")
	requestURL += "?timeout=1s"
	fakeHandler.ValidateRequest(t, requestURL, "POST", &tmpStr)
}

func TestWasCreated(t *testing.T) {
	reqObj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	reqBodyExpected, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), reqObj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedObj := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{
		Protocol:   "TCP",
		Port:       12345,
		TargetPort: intstr.FromInt(12345),
	}}}}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expectedObj)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   201,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()
	c := testRESTClient(t, testServer)
	wasCreated := false
	obj, err := c.Verb("PUT").
		Prefix("foo/bar", "baz").
		Timeout(time.Second).
		Body(reqBodyExpected).
		Do().WasCreated(&wasCreated).Get()
	if err != nil {
		t.Errorf("Unexpected error: %v %#v", err, err)
		return
	}
	if obj == nil {
		t.Error("nil obj")
	} else if !apiequality.Semantic.DeepDerivative(expectedObj, obj) {
		t.Errorf("Expected: %#v, got %#v", expectedObj, obj)
	}
	if !wasCreated {
		t.Errorf("Expected object was created")
	}

	tmpStr := string(reqBodyExpected)
	requestURL := defaultResourcePathWithPrefix("foo/bar/baz", "", "", "")
	requestURL += "?timeout=1s"
	fakeHandler.ValidateRequest(t, requestURL, "PUT", &tmpStr)
}

func TestVerbs(t *testing.T) {
	c := testRESTClient(t, nil)
	if r := c.Post(); r.verb != "POST" {
		t.Errorf("Post verb is wrong")
	}
	if r := c.Put(); r.verb != "PUT" {
		t.Errorf("Put verb is wrong")
	}
	if r := c.Get(); r.verb != "GET" {
		t.Errorf("Get verb is wrong")
	}
	if r := c.Delete(); r.verb != "DELETE" {
		t.Errorf("Delete verb is wrong")
	}
}

func TestAbsPath(t *testing.T) {
	for i, tc := range []struct {
		configPrefix   string
		resourcePrefix string
		absPath        string
		wantsAbsPath   string
	}{
		{"/", "", "", "/"},
		{"", "", "/", "/"},
		{"", "", "/api", "/api"},
		{"", "", "/api/", "/api/"},
		{"", "", "/apis", "/apis"},
		{"", "/foo", "/bar/foo", "/bar/foo"},
		{"", "/api/foo/123", "/bar/foo", "/bar/foo"},
		{"/p1", "", "", "/p1"},
		{"/p1", "", "/", "/p1/"},
		{"/p1", "", "/api", "/p1/api"},
		{"/p1", "", "/apis", "/p1/apis"},
		{"/p1", "/r1", "/apis", "/p1/apis"},
		{"/p1", "/api/r1", "/apis", "/p1/apis"},
		{"/p1/api/p2", "", "", "/p1/api/p2"},
		{"/p1/api/p2", "", "/", "/p1/api/p2/"},
		{"/p1/api/p2", "", "/api", "/p1/api/p2/api"},
		{"/p1/api/p2", "", "/api/", "/p1/api/p2/api/"},
		{"/p1/api/p2", "/r1", "/api/", "/p1/api/p2/api/"},
		{"/p1/api/p2", "/api/r1", "/api/", "/p1/api/p2/api/"},
	} {
		u, _ := url.Parse("http://localhost:123" + tc.configPrefix)
		r := NewRequest(nil, "POST", u, "", ContentConfig{GroupVersion: &schema.GroupVersion{Group: "test"}}, Serializers{}, nil, nil, 0).Prefix(tc.resourcePrefix).AbsPath(tc.absPath)
		if r.pathPrefix != tc.wantsAbsPath {
			t.Errorf("test case %d failed, unexpected path: %q, expected %q", i, r.pathPrefix, tc.wantsAbsPath)
		}
	}
}

func TestUnacceptableParamNames(t *testing.T) {
	table := []struct {
		name          string
		testVal       string
		expectSuccess bool
	}{
		// timeout is no longer "protected"
		{"timeout", "42", true},
	}

	for _, item := range table {
		c := testRESTClient(t, nil)
		r := c.Get().setParam(item.name, item.testVal)
		if e, a := item.expectSuccess, r.err == nil; e != a {
			t.Errorf("expected %v, got %v (%v)", e, a, r.err)
		}
	}
}

func TestBody(t *testing.T) {
	const data = "test payload"

	obj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	bodyExpected, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), obj)

	f, err := ioutil.TempFile("", "test_body")
	if err != nil {
		t.Fatalf("TempFile error: %v", err)
	}
	if _, err := f.WriteString(data); err != nil {
		t.Fatalf("TempFile.WriteString error: %v", err)
	}
	f.Close()
	defer os.Remove(f.Name())

	var nilObject *metav1.DeleteOptions
	typedObject := interface{}(nilObject)
	c := testRESTClient(t, nil)
	tests := []struct {
		input    interface{}
		expected string
		headers  map[string]string
	}{
		{[]byte(data), data, nil},
		{f.Name(), data, nil},
		{strings.NewReader(data), data, nil},
		{obj, string(bodyExpected), map[string]string{"Content-Type": "application/json"}},
		{typedObject, "", nil},
	}
	for i, tt := range tests {
		r := c.Post().Body(tt.input)
		if r.err != nil {
			t.Errorf("%d: r.Body(%#v) error: %v", i, tt, r.err)
			continue
		}
		if tt.headers != nil {
			for k, v := range tt.headers {
				if r.headers.Get(k) != v {
					t.Errorf("%d: r.headers[%q] = %q; want %q", i, k, v, v)
				}
			}
		}

		if r.body == nil {
			if len(tt.expected) != 0 {
				t.Errorf("%d: r.body = %q; want %q", i, r.body, tt.expected)
			}
			continue
		}
		buf := make([]byte, len(tt.expected))
		if _, err := r.body.Read(buf); err != nil {
			t.Errorf("%d: r.body.Read error: %v", i, err)
			continue
		}
		body := string(buf)
		if body != tt.expected {
			t.Errorf("%d: r.body = %q; want %q", i, body, tt.expected)
		}
	}
}

func TestWatch(t *testing.T) {
	var table = []struct {
		t   watch.EventType
		obj runtime.Object
	}{
		{watch.Added, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "first"}}},
		{watch.Modified, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "second"}}},
		{watch.Deleted, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "last"}}},
	}

	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("need flusher!")
		}

		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()

		encoder := restclientwatch.NewEncoder(streaming.NewEncoder(w, scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)), scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion))
		for _, item := range table {
			if err := encoder.Encode(&watch.Event{Type: item.t, Object: item.obj}); err != nil {
				panic(err)
			}
			flusher.Flush()
		}
	}))
	defer testServer.Close()

	s := testRESTClient(t, testServer)
	watching, err := s.Get().Prefix("path/to/watch/thing").Watch()
	if err != nil {
		t.Fatalf("Unexpected error")
	}

	for _, item := range table {
		got, ok := <-watching.ResultChan()
		if !ok {
			t.Fatalf("Unexpected early close")
		}
		if e, a := item.t, got.Type; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := item.obj, got.Object; !apiequality.Semantic.DeepDerivative(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}

	_, ok := <-watching.ResultChan()
	if ok {
		t.Fatal("Unexpected non-close")
	}
}

func TestStream(t *testing.T) {
	expectedBody := "expected body"

	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("need flusher!")
		}
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(expectedBody))
		flusher.Flush()
	}))
	defer testServer.Close()

	s := testRESTClient(t, testServer)
	readCloser, err := s.Get().Prefix("path/to/stream/thing").Stream()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer readCloser.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(readCloser)
	resultBody := buf.String()

	if expectedBody != resultBody {
		t.Errorf("Expected %s, got %s", expectedBody, resultBody)
	}
}

func testRESTClient(t testing.TB, srv *httptest.Server) *RESTClient {
	baseURL, _ := url.Parse("http://localhost")
	if srv != nil {
		var err error
		baseURL, err = url.Parse(srv.URL)
		if err != nil {
			t.Fatalf("failed to parse test URL: %v", err)
		}
	}
	versionedAPIPath := defaultResourcePathWithPrefix("", "", "", "")
	client, err := NewRESTClient(baseURL, versionedAPIPath, defaultContentConfig(), 0, 0, nil, nil)
	if err != nil {
		t.Fatalf("failed to create a client: %v", err)
	}
	return client
}

func TestDoContext(t *testing.T) {
	receivedCh := make(chan struct{})
	block := make(chan struct{})
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		close(receivedCh)
		<-block
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()
	defer close(block)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-receivedCh
		cancel()
	}()

	c := testRESTClient(t, testServer)
	_, err := c.Verb("GET").
		Context(ctx).
		Prefix("foo").
		DoRaw()
	if err == nil {
		t.Fatal("Expected context cancellation error")
	}
}

func buildString(length int) string {
	s := make([]byte, length)
	for i := range s {
		s[i] = 'a'
	}
	return string(s)
}

func init() {
	klog.InitFlags(nil)
}

func TestTruncateBody(t *testing.T) {
	tests := []struct {
		body  string
		want  string
		level string
	}{
		// Anything below 8 is completely truncated
		{
			body:  "Completely truncated below 8",
			want:  " [truncated 28 chars]",
			level: "0",
		},
		// Small strings are not truncated by high levels
		{
			body:  "Small body never gets truncated",
			want:  "Small body never gets truncated",
			level: "10",
		},
		{
			body:  "Small body never gets truncated",
			want:  "Small body never gets truncated",
			level: "8",
		},
		// Strings are truncated to 1024 if level is less than 9.
		{
			body:  buildString(2000),
			level: "8",
			want:  fmt.Sprintf("%s [truncated 976 chars]", buildString(1024)),
		},
		// Strings are truncated to 10240 if level is 9.
		{
			body:  buildString(20000),
			level: "9",
			want:  fmt.Sprintf("%s [truncated 9760 chars]", buildString(10240)),
		},
		// Strings are not truncated if level is 10 or higher
		{
			body:  buildString(20000),
			level: "10",
			want:  buildString(20000),
		},
		// Strings are not truncated if level is 10 or higher
		{
			body:  buildString(20000),
			level: "11",
			want:  buildString(20000),
		},
	}

	l := flag.Lookup("v").Value.(flag.Getter).Get().(klog.Level)
	for _, test := range tests {
		flag.Set("v", test.level)
		got := truncateBody(test.body)
		if got != test.want {
			t.Errorf("truncateBody(%v) = %v, want %v", test.body, got, test.want)
		}
	}
	flag.Set("v", l.String())
}

func defaultResourcePathWithPrefix(prefix, resource, namespace, name string) string {
	var path string
	path = "/api/" + v1.SchemeGroupVersion.Version

	if prefix != "" {
		path = path + "/" + prefix
	}
	if namespace != "" {
		path = path + "/namespaces/" + namespace
	}
	// Resource names are lower case.
	resource = strings.ToLower(resource)
	if resource != "" {
		path = path + "/" + resource
	}
	if name != "" {
		path = path + "/" + name
	}
	return path
}
