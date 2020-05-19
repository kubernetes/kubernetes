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
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	v1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/scheme"
	utiltesting "k8s.io/client-go/util/testing"

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
		t.Errorf("Unexpected mis-match: %s", diff.ObjectReflectDiff(status, &actual))
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
		t.Errorf("Unexpected mis-match: %s", diff.ObjectReflectDiff(status, &actual))
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

	t.Logf(testProxyServer.URL)

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

	theUrl, _ := url.Parse("http://localhost")

	// 1 second base backoff + duration of 2 seconds -> exponential backoff for requests.
	os.Setenv(envBackoffBase, "1")
	os.Setenv(envBackoffDuration, "2")
	backoff := readExpBackoffConfig()
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	if backoff.CalculateBackoff(theUrl)/time.Second != 2 {
		t.Errorf("Backoff env not working.")
	}

	// 0 duration -> no backoff.
	os.Setenv(envBackoffBase, "1")
	os.Setenv(envBackoffDuration, "0")
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff = readExpBackoffConfig()
	if backoff.CalculateBackoff(theUrl)/time.Second != 0 {
		t.Errorf("Zero backoff duration, but backoff still occurring.")
	}

	// No env -> No backoff.
	os.Setenv(envBackoffBase, "")
	os.Setenv(envBackoffDuration, "")
	backoff = readExpBackoffConfig()
	backoff.UpdateBackoff(theUrl, nil, 500)
	backoff.UpdateBackoff(theUrl, nil, 500)
	if backoff.CalculateBackoff(theUrl)/time.Second != 0 {
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
