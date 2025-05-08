/*
Copyright 2025 The Kubernetes Authors.

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

package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	endpointstesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	utiltesting "k8s.io/client-go/util/testing"
)

// Fake API versions, similar to api/latest.go
const namedGroupPrefix = "apis"
const testAPIGroup = "test.group"

var testGroupV1 = schema.GroupVersion{Group: testAPIGroup, Version: "1"}
var testGroupV2 = schema.GroupVersion{Group: testAPIGroup, Version: "2"}
var testCodecV2 = codecs.LegacyCodec(testGroupV2)

func addTestTypesV2() {
	scheme.AddKnownTypes(testGroupV2,
		&endpointstesting.Simple{},
		&endpointstesting.SimpleList{},
	)
	metav1.AddToGroupVersion(scheme, testGroupV2)
}

func init() {
	addTestTypesV2()
}

func TestWatchHTTPErrors(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	responseDoneCh := make(chan struct{})
	go func() {
		defer watcher.Stop()
		<-responseDoneCh
	}()

	timeoutCtx, timeoutCancel := context.WithCancel(ctx)
	defer timeoutCancel()

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		TimeoutContext: timeoutCtx,
		Scope:          &RequestScope{},
		Watcher:        watcher,
		DoneChannel:    responseDoneCh,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,
	}

	s := httptest.NewServer(serveWatch(watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, err := url.Parse(s.URL)
	require.NoError(t, err)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
	require.NoError(t, err)
	client := http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer apitesting.Close(t, resp.Body)

	// Send error to server from storage
	errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error")).Status()
	watcher.Error(&errStatus)
	watcher.Stop()

	// Decode error from the response
	decoder := json.NewDecoder(resp.Body)
	var got watchJSON
	err = decoder.Decode(&got)
	require.NoError(t, err)
	require.Equal(t, watch.Error, got.Type)
	status := &metav1.Status{}
	err = json.Unmarshal(got.Object, status)
	require.NoError(t, err)
	expectedStatus := &metav1.Status{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Status",
			APIVersion: "v1",
		},
		Code:    500,
		Status:  "Failure",
		Message: "Internal error occurred: we got an error",
		Reason:  errStatus.Reason,
		Details: errStatus.Details,
	}
	require.Equal(t, expectedStatus, status)

	// Close the response body to signal the server to stop serving.
	require.NoError(t, resp.Body.Close())

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")
}

func TestWatchHTTPErrorsBeforeServe(t *testing.T) {
	ctx := t.Context()
	responseDoneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}

	// Setup a new watchserver.
	// Most of the fields are unused for this test, because serveWatch exits before calling HandleHTTP.
	watchServer := &WatchServer{
		Scope: &RequestScope{
			Serializer: runtime.NewSimpleNegotiatedSerializer(info),
			Kind:       testGroupV1.WithKind("test"),
		},
		DoneChannel: responseDoneCh,
	}

	statusErr := apierrors.NewInternalError(fmt.Errorf("we got an error"))
	errStatus := statusErr.Status()

	s := httptest.NewServer(serveWatch(watchServer, statusErr))
	defer s.Close()

	// Setup a client
	dest, err := url.Parse(s.URL)
	require.NoError(t, err)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
	require.NoError(t, err)
	client := http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer apitesting.Close(t, resp.Body)

	// We had already got an error before watch serve started
	decoder := json.NewDecoder(resp.Body)
	var status *metav1.Status
	err = decoder.Decode(&status)
	require.NoError(t, err)
	expectedStatus := &metav1.Status{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Status",
			APIVersion: "v1",
		},
		Code:    500,
		Status:  "Failure",
		Message: "Internal error occurred: we got an error",
		Reason:  errStatus.Reason,
		Details: errStatus.Details,
	}
	require.Equal(t, expectedStatus, status)

	// Close the response body to signal the server to stop serving.
	// This isn't strictly necessary, since the test serveWatch doesn't block,
	// but it would be if this were the real watch server.
	require.NoError(t, resp.Body.Close())

	// Wait for the server to close the DoneChannel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, responseDoneCh)
	require.NoError(t, err)
}

func TestWatchHTTPDynamicClientErrors(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	responseDoneCh := make(chan struct{})
	go func() {
		defer watcher.Stop()
		<-responseDoneCh
	}()

	timeoutCtx, timeoutCancel := context.WithCancel(ctx)
	defer timeoutCancel()

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		TimeoutContext: timeoutCtx,
		Scope:          &RequestScope{},
		Watcher:        watcher,
		DoneChannel:    responseDoneCh,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,
	}

	s := httptest.NewServer(serveWatch(watchServer, nil))
	defer s.Close()
	defer s.CloseClientConnections()

	client := dynamic.NewForConfigOrDie(&restclient.Config{
		Host:    s.URL,
		APIPath: "/" + namedGroupPrefix,
	}).Resource(testGroupV2.WithResource("simple"))

	_, err := client.Watch(ctx, metav1.ListOptions{})
	require.Equal(t, runtime.NegotiateError{Stream: true, ContentType: "testcase/json"}, err)

	// The client should automatically close the connection on error.

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")
}

func TestWatchHTTPTimeout(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	responseDoneCh := make(chan struct{})
	go func() {
		defer watcher.Stop()
		<-responseDoneCh
	}()

	timeoutCtx, timeoutCancel := context.WithCancel(ctx)
	defer timeoutCancel()

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		TimeoutContext: timeoutCtx,
		Scope:          &RequestScope{},
		Watcher:        watcher,
		DoneChannel:    responseDoneCh,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,
	}

	s := httptest.NewServer(serveWatch(watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, err := url.Parse(s.URL)
	require.NoError(t, err)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
	require.NoError(t, err)
	client := http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer apitesting.Close(t, resp.Body)

	// Send object added event to server from storage
	watcher.Add(&endpointstesting.Simple{TypeMeta: metav1.TypeMeta{APIVersion: testGroupV2.String()}})

	// Make sure we can actually watch an endpoint
	decoder := json.NewDecoder(resp.Body)
	var got watchJSON
	err = decoder.Decode(&got)
	require.NoError(t, err)

	// Simulate server-side timeout.
	// Technically, this sends Canceled, not DeadlineExceeded,
	// but they're handled the same by the server.
	timeoutCancel()

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")

	// Ensure the response receives EOF after server-side timeout
	// TODO(karlkfi): Should this be DeadlineExceeded? Seems to be a race condition.
	err = decoder.Decode(&got)
	require.Equal(t, io.EOF, err)

	// Close the response body. The server has already stopped, but the client
	// should always close the response body when done reading the response.
	require.NoError(t, resp.Body.Close())
}

// watchJSON defines the expected JSON wire equivalent of watch.Event.
// Use this for testing instead of metav1.WatchEvent to ensure the wire format
// doesn't change and skip the automatic decoding of WatchEvent.Object.Raw into
// WatchEvent.Object.Object.
type watchJSON struct {
	Type   watch.EventType `json:"type,omitempty"`
	Object json.RawMessage `json:"object,omitempty"`
}

// serveWatch will serve a watch response according to the watcher and watchServer.
// Before watchServer.HandleHTTP, an error may occur like k8s.io/apiserver/pkg/endpoints/handlers/watch.go#serveWatch does.
func serveWatch(watchServer *WatchServer, preServeErr error) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		watchServer.RequestContext = req.Context()
		if preServeErr != nil {
			defer close(watchServer.DoneChannel)
			responsewriters.ErrorNegotiated(preServeErr, watchServer.Scope.Serializer, watchServer.Scope.Kind.GroupVersion(), w, req)
			return
		}
		watchServer.HandleHTTP(w, req)
	}
}
