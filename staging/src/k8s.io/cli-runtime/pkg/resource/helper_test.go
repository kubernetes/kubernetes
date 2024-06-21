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
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"

	// TODO we need to remove this linkage and create our own scheme
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
)

func objBody(obj runtime.Object) io.ReadCloser {
	return io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(corev1Codec, obj))))
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

func V1DeepEqualSafePodStatus() corev1.PodStatus {
	return corev1.PodStatus{
		Conditions: []corev1.PodCondition{
			{
				Status: corev1.ConditionTrue,
				Type:   corev1.PodReady,
			},
		},
	}
}

func TestHelperDelete(t *testing.T) {
	tests := []struct {
		name    string
		Err     bool
		Req     func(*http.Request) bool
		Resp    *http.Response
		HttpErr error
	}{
		{
			name:    "test1",
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			name: "test2",
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			name: "test3pkg/kubectl/genericclioptions/resource/helper_test.go",
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
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &fake.RESTClient{
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Resp:                 tt.Resp,
				Err:                  tt.HttpErr,
			}
			modifier := &Helper{
				RESTClient:      client,
				NamespaceScoped: true,
			}
			_, err := modifier.Delete("bar", "foo")
			if (err != nil) != tt.Err {
				t.Errorf("unexpected error: %t %v", tt.Err, err)
			}
			if err != nil {
				return
			}
			if tt.Req != nil && !tt.Req(client.Req) {
				t.Errorf("unexpected request: %#v", client.Req)
			}
		})
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
		name    string
		Resp    *http.Response
		HttpErr error
		Modify  bool
		Object  runtime.Object

		ExpectObject runtime.Object
		Err          bool
		Req          func(*http.Request) bool
	}{
		{
			name:    "test1",
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			name: "test1",
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			name: "test1",
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
			name:         "test1",
			Modify:       false,
			Object:       &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			ExpectObject: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			Resp:         &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})},
			Req:          expectPost,
		},
		{
			name:   "test1",
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
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &fake.RESTClient{
				GroupVersion:         corev1GV,
				NegotiatedSerializer: scheme.Codecs,
				Resp:                 tt.Resp,
				Err:                  tt.HttpErr,
			}
			modifier := &Helper{
				RESTClient:      client,
				NamespaceScoped: true,
			}
			_, err := modifier.Create("bar", tt.Modify, tt.Object)
			if (err != nil) != tt.Err {
				t.Errorf("%d: unexpected error: %t %v", i, tt.Err, err)
			}
			if err != nil {
				return
			}
			if tt.Req != nil && !tt.Req(client.Req) {
				t.Errorf("%d: unexpected request: %#v", i, client.Req)
			}
			body, err := io.ReadAll(client.Req.Body)
			if err != nil {
				t.Fatalf("%d: unexpected error: %#v", i, err)
			}
			t.Logf("got body: %s", string(body))
			expect := []byte{}
			if tt.ExpectObject != nil {
				expect = []byte(runtime.EncodeOrDie(corev1Codec, tt.ExpectObject))
			}
			if !reflect.DeepEqual(expect, body) {
				t.Errorf("%d: unexpected body: %s (expected %s)", i, string(body), string(expect))
			}
		})
	}
}

func TestHelperGet(t *testing.T) {
	tests := []struct {
		name        string
		subresource string
		Err         bool
		Req         func(*http.Request) bool
		Resp        *http.Response
		HttpErr     error
	}{
		{
			name:    "test1",
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			name: "test1",
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			name: "test1",
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
		{
			name:        "test with subresource",
			subresource: "status",
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
				if parts[3] != "status" {
					t.Errorf("url doesn't contain subresource: %#v", req)
					return false
				}
				return true
			},
		},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &fake.RESTClient{
				GroupVersion:         corev1GV,
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Resp:                 tt.Resp,
				Err:                  tt.HttpErr,
			}
			modifier := &Helper{
				RESTClient:      client,
				NamespaceScoped: true,
				Subresource:     tt.subresource,
			}
			obj, err := modifier.Get("bar", "foo")

			if (err != nil) != tt.Err {
				t.Errorf("unexpected error: %d %t %v", i, tt.Err, err)
			}
			if err != nil {
				return
			}
			if obj.(*corev1.Pod).Name != "foo" {
				t.Errorf("unexpected object: %#v", obj)
			}
			if tt.Req != nil && !tt.Req(client.Req) {
				t.Errorf("unexpected request: %#v", client.Req)
			}
		})
	}
}

func TestHelperList(t *testing.T) {
	labelKey := "labelSelector"
	tests := []struct {
		name    string
		Err     bool
		Req     func(*http.Request) bool
		Resp    *http.Response
		HttpErr error
	}{
		{
			name:    "test1",
			HttpErr: errors.New("failure"),
			Err:     true,
		},
		{
			name: "test2",
			Resp: &http.Response{
				StatusCode: http.StatusNotFound,
				Header:     header(),
				Body:       objBody(&metav1.Status{Status: metav1.StatusFailure}),
			},
			Err: true,
		},
		{
			name: "test3",
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
				if req.URL.Query().Get(labelKey) != labels.SelectorFromSet(labels.Set{"foo": "baz"}).String() {
					t.Errorf("url doesn't contain query parameters: %#v", req.URL)
					return false
				}
				return true
			},
		},
		{
			name: "test with",
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
				if req.URL.Query().Get(labelKey) != labels.SelectorFromSet(labels.Set{"foo": "baz"}).String() {
					t.Errorf("url doesn't contain query parameters: %#v", req.URL)
					return false
				}
				return true
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &fake.RESTClient{
				GroupVersion:         corev1GV,
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Resp:                 tt.Resp,
				Err:                  tt.HttpErr,
			}
			modifier := &Helper{
				RESTClient:      client,
				NamespaceScoped: true,
			}
			obj, err := modifier.List("bar", corev1GV.String(), &metav1.ListOptions{LabelSelector: "foo=baz"})
			if (err != nil) != tt.Err {
				t.Errorf("unexpected error: %t %v", tt.Err, err)
			}
			if err != nil {
				return
			}
			if obj.(*corev1.PodList).Items[0].Name != "foo" {
				t.Errorf("unexpected object: %#v", obj)
			}
			if tt.Req != nil && !tt.Req(client.Req) {
				t.Errorf("unexpected request: %#v", client.Req)
			}
		})
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

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			_, err := modifier.List("bar",
				corev1GV.String(),
				&metav1.ListOptions{LabelSelector: tt.LabelSelector, FieldSelector: tt.FieldSelector})
			if tt.Err {
				if err == nil {
					t.Errorf("%q expected error: %q", tt.Name, tt.ErrMsg)
				}
				if err != nil && err.Error() != tt.ErrMsg {
					t.Errorf("%q expected error: %q", tt.Name, tt.ErrMsg)
				}
			}
		})
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
		Name            string
		Resp            *http.Response
		HTTPClient      *http.Client
		HttpErr         error
		Overwrite       bool
		Object          runtime.Object
		Namespace       string
		NamespaceScoped bool
		Subresource     string

		ExpectPath   string
		ExpectObject runtime.Object
		Err          bool
		Req          func(string, *http.Request) bool
	}{
		{
			Name:            "test1",
			Namespace:       "bar",
			NamespaceScoped: true,
			HttpErr:         errors.New("failure"),
			Err:             true,
		},
		{
			Name:            "test2",
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
			Name:            "test3",
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
			Name:            "test4",
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
			Name: "test5",
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
			Name:            "test6",
			Namespace:       "bar",
			NamespaceScoped: true,
			Object:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			ExpectPath:      "/namespaces/bar/foo",
			ExpectObject:    &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"}},
			Resp:            &http.Response{StatusCode: http.StatusOK, Header: header(), Body: objBody(&metav1.Status{Status: metav1.StatusSuccess})},
			Req:             expectPut,
		},
		{
			Name:            "test7 - with status subresource",
			Namespace:       "bar",
			NamespaceScoped: true,
			Subresource:     "status",
			Object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Status:     V1DeepEqualSafePodStatus(),
			},
			ExpectPath: "/namespaces/bar/foo/status",
			ExpectObject: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: "10"},
				Status:     V1DeepEqualSafePodStatus(),
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
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			client := &fake.RESTClient{
				GroupVersion:         corev1GV,
				NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
				Client:               tt.HTTPClient,
				Resp:                 tt.Resp,
				Err:                  tt.HttpErr,
			}
			modifier := &Helper{
				RESTClient:      client,
				NamespaceScoped: tt.NamespaceScoped,
				Subresource:     tt.Subresource,
			}
			_, err := modifier.Replace(tt.Namespace, "foo", tt.Overwrite, tt.Object)
			if (err != nil) != tt.Err {
				t.Fatalf("unexpected error: %t %v", tt.Err, err)
			}
			if err != nil {
				return
			}
			if tt.Req != nil && (client.Req == nil || !tt.Req(tt.ExpectPath, client.Req)) {
				t.Fatalf("unexpected request: %#v", client.Req)
			}
			body, err := io.ReadAll(client.Req.Body)
			if err != nil {
				t.Fatalf("unexpected error: %#v", err)
			}
			expect := []byte{}
			if tt.ExpectObject != nil {
				expect = []byte(runtime.EncodeOrDie(corev1Codec, tt.ExpectObject))
			}
			if !reflect.DeepEqual(expect, body) {
				t.Fatalf("unexpected body: %s", string(body))
			}
		})
	}
}

func TestEnhanceListError(t *testing.T) {
	podGVR := corev1.SchemeGroupVersion.WithResource(corev1.ResourcePods.String())
	podSubject := podGVR.String()
	tests := []struct {
		name string
		err  error
		opts metav1.ListOptions
		subj string

		expectedErr     string
		expectStatusErr bool
	}{
		{
			name:            "leaves resource expired error as is",
			err:             apierrors.NewResourceExpired("resourceversion too old"),
			opts:            metav1.ListOptions{},
			subj:            podSubject,
			expectedErr:     "resourceversion too old",
			expectStatusErr: true,
		}, {
			name:            "leaves unrecognized error as is",
			err:             errors.New("something went wrong"),
			opts:            metav1.ListOptions{},
			subj:            podSubject,
			expectedErr:     "something went wrong",
			expectStatusErr: false,
		}, {
			name:            "bad request StatusError without selectors",
			err:             apierrors.NewBadRequest("request is invalid"),
			opts:            metav1.ListOptions{},
			subj:            podSubject,
			expectedErr:     "Unable to list \"/v1, Resource=pods\": request is invalid",
			expectStatusErr: true,
		}, {
			name: "bad request StatusError with selectors",
			err:  apierrors.NewBadRequest("request is invalid"),
			opts: metav1.ListOptions{
				LabelSelector: "a=b",
				FieldSelector: ".spec.nodeName=foo",
			},
			subj:            podSubject,
			expectedErr:     "Unable to find \"/v1, Resource=pods\" that match label selector \"a=b\", field selector \".spec.nodeName=foo\": request is invalid",
			expectStatusErr: true,
		}, {
			name:            "not found without selectors",
			err:             apierrors.NewNotFound(podGVR.GroupResource(), "foo"),
			opts:            metav1.ListOptions{},
			subj:            podSubject,
			expectedErr:     "Unable to list \"/v1, Resource=pods\": pods \"foo\" not found",
			expectStatusErr: true,
		}, {
			name: "not found StatusError with selectors",
			err:  apierrors.NewNotFound(podGVR.GroupResource(), "foo"),
			opts: metav1.ListOptions{
				LabelSelector: "a=b",
				FieldSelector: ".spec.nodeName=foo",
			},
			subj:            podSubject,
			expectedErr:     "Unable to find \"/v1, Resource=pods\" that match label selector \"a=b\", field selector \".spec.nodeName=foo\": pods \"foo\" not found",
			expectStatusErr: true,
		}, {
			name: "non StatusError without selectors",
			err: fmt.Errorf("extra info: %w", apierrors.NewNotFound(podGVR.GroupResource(),
				"foo")),
			opts:            metav1.ListOptions{},
			subj:            podSubject,
			expectedErr:     "Unable to list \"/v1, Resource=pods\": extra info: pods \"foo\" not found",
			expectStatusErr: false,
		}, {
			name: "non StatusError with selectors",
			err:  fmt.Errorf("extra info: %w", apierrors.NewNotFound(podGVR.GroupResource(), "foo")),
			opts: metav1.ListOptions{
				LabelSelector: "a=b",
				FieldSelector: ".spec.nodeName=foo",
			},
			subj: podSubject,
			expectedErr: "Unable to find \"/v1, " +
				"Resource=pods\" that match label selector \"a=b\", " +
				"field selector \".spec.nodeName=foo\": extra info: pods \"foo\" not found",
			expectStatusErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := EnhanceListError(tt.err, tt.opts, tt.subj)
			if err == nil {
				t.Errorf("EnhanceListError did not return an error")
			}
			if err.Error() != tt.expectedErr {
				t.Errorf("EnhanceListError() error = %q, expectedErr %q", err, tt.expectedErr)
			}
			if tt.expectStatusErr {
				if _, ok := err.(*apierrors.StatusError); !ok {
					t.Errorf("EnhanceListError incorrectly returned a non-StatusError: %v", err)
				}
			}
		})
	}
}

func TestFollowContinue(t *testing.T) {
	var continueTokens []string
	tests := []struct {
		name        string
		initialOpts *metav1.ListOptions
		tokensSeen  []string
		listFunc    func(metav1.ListOptions) (runtime.Object, error)

		expectedTokens []string
		wantErr        string
	}{
		{
			name:        "updates list options with continue token until list finished",
			initialOpts: &metav1.ListOptions{},
			listFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				continueTokens = append(continueTokens, options.Continue)
				obj := corev1.PodList{}
				switch options.Continue {
				case "":
					metadataAccessor.SetContinue(&obj, "abc")
				case "abc":
					metadataAccessor.SetContinue(&obj, "def")
				case "def":
					metadataAccessor.SetKind(&obj, "ListComplete")
				}
				return &obj, nil
			},
			expectedTokens: []string{"", "abc", "def"},
		},
		{
			name:        "stops looping if listFunc returns an error",
			initialOpts: &metav1.ListOptions{},
			listFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				continueTokens = append(continueTokens, options.Continue)
				obj := corev1.PodList{}
				switch options.Continue {
				case "":
					metadataAccessor.SetContinue(&obj, "abc")
				case "abc":
					return nil, fmt.Errorf("err from list func")
				case "def":
					metadataAccessor.SetKind(&obj, "ListComplete")
				}
				return &obj, nil
			},
			expectedTokens: []string{"", "abc"},
			wantErr:        "err from list func",
		},
	}
	for _, tt := range tests {
		continueTokens = []string{}
		t.Run(tt.name, func(t *testing.T) {
			err := FollowContinue(tt.initialOpts, tt.listFunc)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("FollowContinue was expected to return an error and did not")
				} else if err.Error() != tt.wantErr {
					t.Fatalf("wanted error %q, got %q", tt.wantErr, err.Error())
				}
			} else {
				if err != nil {
					t.Errorf("FollowContinue failed: %v", tt.wantErr)
				}
				if !reflect.DeepEqual(continueTokens, tt.expectedTokens) {
					t.Errorf("got token list %q, wanted %q", continueTokens, tt.expectedTokens)
				}
			}
		})
	}
}
