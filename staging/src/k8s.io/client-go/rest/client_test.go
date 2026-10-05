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
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	v1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/kubernetes/scheme"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/klog/v2/ktesting"

	"github.com/google/go-cmp/cmp"
)

type TestParam struct {
	actualError           error
	expectingError        bool
	actualCreated         bool
	expCreated            bool
	expStatus             *metav1.Status
	testBody              bool
	testBodyErrorIsNotNil bool
}

// TestSerializer makes sure that you're always able to decode metav1.Status
func TestSerializer(t *testing.T) {
	gv := v1beta1.SchemeGroupVersion
	contentConfig := ContentConfig{
		ContentType:          "application/json",
		GroupVersion:         &gv,
		NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
	}

	n := runtime.NewClientNegotiator(contentConfig.NegotiatedSerializer, gv)
	d, err := n.Decoder("application/json", nil)
	if err != nil {
		t.Fatal(err)
	}

	// bytes based on actual return from API server when encoding an "unversioned" object
	obj, err := runtime.Decode(d, []byte(`{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Success"}`))
	t.Log(obj)
	if err != nil {
		t.Fatal(err)
	}
}

func TestDoRequestSuccess(t *testing.T) {
	testServer, fakeHandler, status := testServerEnv(t, 200)
	defer testServer.Close()

	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Prefix("test").Do(context.Background()).Raw()

	testParam := TestParam{actualError: err, expectingError: false, expCreated: true,
		expStatus: status, testBody: true, testBodyErrorIsNotNil: false}
	validate(testParam, t, body, fakeHandler)
}

func TestDoRequestFailed(t *testing.T) {
	status := &metav1.Status{
		Code:    http.StatusNotFound,
		Status:  metav1.StatusFailure,
		Reason:  metav1.StatusReasonNotFound,
		Message: " \"\" not found",
		Details: &metav1.StatusDetails{},
	}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   404,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = c.Get().Do(context.Background()).Error()
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	ss, ok := err.(errors.APIStatus)
	if !ok {
		t.Errorf("unexpected error type %v", err)
	}
	actual := ss.Status()
	if !reflect.DeepEqual(status, &actual) {
		t.Errorf("Unexpected mis-match: %s", cmp.Diff(status, &actual))
	}
}

func TestDoRawRequestFailed(t *testing.T) {
	status := &metav1.Status{
		Code:    http.StatusNotFound,
		Status:  metav1.StatusFailure,
		Reason:  metav1.StatusReasonNotFound,
		Message: "the server could not find the requested resource",
		Details: &metav1.StatusDetails{
			Causes: []metav1.StatusCause{
				{Type: metav1.CauseTypeUnexpectedServerResponse, Message: "unknown"},
			},
		},
	}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   404,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := c.Get().Do(context.Background()).Raw()

	if err == nil || body == nil {
		t.Errorf("unexpected non-error: %#v", body)
	}
	ss, ok := err.(errors.APIStatus)
	if !ok {
		t.Errorf("unexpected error type %v", err)
	}
	actual := ss.Status()
	if !reflect.DeepEqual(status, &actual) {
		t.Errorf("Unexpected mis-match: %s", cmp.Diff(status, &actual))
	}
}

func TestDoRequestCreated(t *testing.T) {
	testServer, fakeHandler, status := testServerEnv(t, 201)
	defer testServer.Close()

	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	created := false
	body, err := c.Get().Prefix("test").Do(context.Background()).WasCreated(&created).Raw()

	testParam := TestParam{actualError: err, expectingError: false, expCreated: true,
		expStatus: status, testBody: false}
	validate(testParam, t, body, fakeHandler)
}

func TestDoRequestNotCreated(t *testing.T) {
	testServer, fakeHandler, expectedStatus := testServerEnv(t, 202)
	defer testServer.Close()
	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	created := false
	body, err := c.Get().Prefix("test").Do(context.Background()).WasCreated(&created).Raw()
	testParam := TestParam{actualError: err, expectingError: false, expCreated: false,
		expStatus: expectedStatus, testBody: false}
	validate(testParam, t, body, fakeHandler)
}

func TestDoRequestAcceptedNoContentReturned(t *testing.T) {
	testServer, fakeHandler, _ := testServerEnv(t, 204)
	defer testServer.Close()

	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	created := false
	body, err := c.Get().Prefix("test").Do(context.Background()).WasCreated(&created).Raw()
	testParam := TestParam{actualError: err, expectingError: false, expCreated: false,
		testBody: false}
	validate(testParam, t, body, fakeHandler)
}

func TestBadRequest(t *testing.T) {
	testServer, fakeHandler, _ := testServerEnv(t, 400)
	defer testServer.Close()
	c, err := restClient(testServer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	created := false
	body, err := c.Get().Prefix("test").Do(context.Background()).WasCreated(&created).Raw()
	testParam := TestParam{actualError: err, expectingError: true, expCreated: false,
		testBody: true}
	validate(testParam, t, body, fakeHandler)
}

func validate(testParam TestParam, t *testing.T, body []byte, fakeHandler *utiltesting.FakeHandler) {
	switch {
	case testParam.expectingError && testParam.actualError == nil:
		t.Errorf("Expected error")
	case !testParam.expectingError && testParam.actualError != nil:
		t.Error(testParam.actualError)
	}
	if !testParam.expCreated {
		if testParam.actualCreated {
			t.Errorf("Expected object not to be created")
		}
	}
	statusOut, err := runtime.Decode(scheme.Codecs.UniversalDeserializer(), body)
	if testParam.testBody {
		if testParam.testBodyErrorIsNotNil && err == nil {
			t.Errorf("Expected Error")
		}
		if !testParam.testBodyErrorIsNotNil && err != nil {
			t.Errorf("Unexpected Error: %v", err)
		}
	}

	if testParam.expStatus != nil {
		if !reflect.DeepEqual(testParam.expStatus, statusOut) {
			t.Errorf("Unexpected mis-match. Expected %#v.  Saw %#v", testParam.expStatus, statusOut)
		}
	}
	fakeHandler.ValidateRequest(t, "/"+v1.SchemeGroupVersion.String()+"/test", "GET", nil)

}

func TestHTTPMethods(t *testing.T) {
	testServer, _, _ := testServerEnv(t, 200)
	defer testServer.Close()
	c, _ := restClient(testServer)

	request := c.Post()
	if request == nil {
		t.Errorf("Post : Object returned should not be nil")
	}

	request = c.Get()
	if request == nil {
		t.Errorf("Get: Object returned should not be nil")
	}

	request = c.Put()
	if request == nil {
		t.Errorf("Put : Object returned should not be nil")
	}

	request = c.Delete()
	if request == nil {
		t.Errorf("Delete : Object returned should not be nil")
	}

	request = c.Patch(types.JSONPatchType)
	if request == nil {
		t.Errorf("Patch : Object returned should not be nil")
	}
}

func TestHTTPProxy(t *testing.T) {
	ctx := context.Background()
	testServer, fh, _ := testServerEnv(t, 200)
	fh.ResponseBody = "backend data"
	defer testServer.Close()

	testProxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		to, err := url.Parse(req.RequestURI)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		w.Write([]byte("proxied: "))
		httputil.NewSingleHostReverseProxy(to).ServeHTTP(w, req)
	}))
	defer testProxyServer.Close()

	t.Log(testProxyServer.URL)

	u, err := url.Parse(testProxyServer.URL)
	if err != nil {
		t.Fatalf("Failed to parse test proxy server url: %v", err)
	}

	c, err := RESTClientFor(&Config{
		Host: testServer.URL,
		ContentConfig: ContentConfig{
			GroupVersion:         &v1.SchemeGroupVersion,
			NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
		},
		Proxy:    http.ProxyURL(u),
		Username: "user",
		Password: "pass",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	request := c.Get()
	if request == nil {
		t.Fatalf("Get: Object returned should not be nil")
	}

	b, err := request.DoRaw(ctx)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got, want := string(b), "proxied: backend data"; !cmp.Equal(got, want) {
		t.Errorf("unexpected body: %v", cmp.Diff(want, got))
	}
}

func TestCreateBackoffManager(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	theUrl, _ := url.Parse("http://localhost")

	// 1 second base backoff + duration of 2 seconds -> exponential backoff for requests.
	t.Setenv(envBackoffBase, "1")
	t.Setenv(envBackoffDuration, "2")
	backoff := readExpBackoffConfig()
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	if backoff.CalculateBackoffWithContext(ctx, theUrl)/time.Second != 2 {
		t.Errorf("Backoff env not working.")
	}

	// 0 duration -> no backoff.
	t.Setenv(envBackoffBase, "1")
	t.Setenv(envBackoffDuration, "0")
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	backoff = readExpBackoffConfig()
	if backoff.CalculateBackoffWithContext(ctx, theUrl)/time.Second != 0 {
		t.Errorf("Zero backoff duration, but backoff still occurring.")
	}

	// No env -> No backoff.
	t.Setenv(envBackoffBase, "")
	t.Setenv(envBackoffDuration, "")
	backoff = readExpBackoffConfig()
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	backoff.UpdateBackoffWithContext(ctx, theUrl, nil, 500)
	if backoff.CalculateBackoffWithContext(ctx, theUrl)/time.Second != 0 {
		t.Errorf("Backoff should have been 0.")
	}

}

func testServerEnv(t *testing.T, statusCode int) (*httptest.Server, *utiltesting.FakeHandler, *metav1.Status) {
	status := &metav1.Status{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Status"}, Status: fmt.Sprintf("%s", metav1.StatusSuccess)}
	expectedBody, _ := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), status)
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   statusCode,
		ResponseBody: string(expectedBody),
		T:            t,
	}
	testServer := httptest.NewServer(&fakeHandler)
	return testServer, &fakeHandler, status
}

func restClient(testServer *httptest.Server) (*RESTClient, error) {
	c, err := RESTClientFor(&Config{
		Host: testServer.URL,
		ContentConfig: ContentConfig{
			GroupVersion:         &v1.SchemeGroupVersion,
			NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
		},
		Username: "user",
		Password: "pass",
	})
	return c, err
}

func TestDropManagedFieldsAccept(t *testing.T) {
	for _, tc := range []struct {
		name       string
		enabled    bool
		drop       bool
		request    func(*RESTClient) *Request
		wantAccept string
	}{
		{
			name:       "default",
			enabled:    true,
			drop:       true,
			request:    (*RESTClient).Get,
			wantAccept: "application/json; drop=metadata.managedFields,*/*; drop=metadata.managedFields",
		},
		{
			name:    "replaced by SetHeader",
			enabled: true,
			drop:    true,
			request: func(c *RESTClient) *Request {
				return c.Get().SetHeader("Accept", "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1;q=0.9", "application/json")
			},
			wantAccept: "application/json; drop=metadata.managedFields,application/json; as=PartialObjectMetadata; drop=metadata.managedFields; g=meta.k8s.io; q=0.9; v=v1",
		},
		{
			name:       "gate disabled",
			enabled:    false,
			drop:       true,
			request:    (*RESTClient).Get,
			wantAccept: "application/json, */*",
		},
		{
			name:       "not opted in",
			enabled:    true,
			drop:       false,
			request:    (*RESTClient).Get,
			wantAccept: "application/json, */*",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ManagedFieldsOptOutClient, tc.enabled)
			var accept string
			c, err := RESTClientForConfigAndClient(&Config{
				Host: "localhost",
				ContentConfig: ContentConfig{
					GroupVersion:         &v1.SchemeGroupVersion,
					NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
					DropManagedFields:    tc.drop,
				},
			}, clientForFunc(func(req *http.Request) (*http.Response, error) {
				accept = req.Header.Get("Accept")
				return &http.Response{StatusCode: http.StatusOK, Body: http.NoBody}, nil
			}))
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}
			if err := tc.request(c).Do(context.Background()).Error(); err != nil {
				t.Fatalf("request failed: %v", err)
			}
			if accept != tc.wantAccept {
				t.Errorf("Accept = %q, want %q", accept, tc.wantAccept)
			}
		})
	}
}

func TestDropManagedFieldsDecode(t *testing.T) {
	const pod = `{"kind":"Pod","apiVersion":"v1","metadata":{"name":"pod","managedFields":[{"manager":"test"}]}}`
	const podList = `{"kind":"PodList","apiVersion":"v1","items":[` + pod + `]}`
	list := func(c *RESTClient) (metav1.Object, error) {
		var pods v1.PodList
		if err := c.Get().Do(context.Background()).Into(&pods); err != nil {
			return nil, err
		}
		return &pods.Items[0], nil
	}
	for _, tc := range []struct {
		name              string
		enabled           bool
		drop              bool
		body              string
		get               func(*RESTClient) (metav1.Object, error)
		wantManagedFields bool
	}{
		{
			name:    "list",
			enabled: true,
			drop:    true,
			body:    podList,
			get:     list,
		},
		{
			name:    "watch",
			enabled: true,
			drop:    true,
			body:    `{"type":"ADDED","object":` + pod + `}`,
			get: func(c *RESTClient) (metav1.Object, error) {
				w, err := c.Get().Watch(context.Background())
				if err != nil {
					return nil, err
				}
				defer w.Stop()
				event := <-w.ResultChan()
				pod, ok := event.Object.(*v1.Pod)
				if !ok {
					return nil, fmt.Errorf("unexpected %s event: %#v", event.Type, event.Object)
				}
				return pod, nil
			},
		},
		{
			name:              "gate disabled",
			enabled:           false,
			drop:              true,
			body:              podList,
			get:               list,
			wantManagedFields: true,
		},
		{
			name:              "not opted in",
			enabled:           true,
			drop:              false,
			body:              podList,
			get:               list,
			wantManagedFields: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ManagedFieldsOptOutClient, tc.enabled)
			c, err := RESTClientForConfigAndClient(&Config{
				Host: "localhost",
				ContentConfig: ContentConfig{
					GroupVersion:         &v1.SchemeGroupVersion,
					NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
					DropManagedFields:    tc.drop,
				},
			}, clientForFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(tc.body)),
				}, nil
			}))
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}
			obj, err := tc.get(c)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			if got := len(obj.GetManagedFields()) > 0; got != tc.wantManagedFields {
				t.Errorf("managedFields present = %t, want %t", got, tc.wantManagedFields)
			}
		})
	}
}
