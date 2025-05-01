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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

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
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		Scope:    &RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh: timeoutCh, done: doneCh},
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

	// Wait for the server to call the CancelFunc returned by
	// TimeoutFactory.TimeoutCh, closing the done channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, doneCh)
	require.NoError(t, err)

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")
}

func TestWatchHTTPErrorsBeforeServe(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		Scope: &RequestScope{
			Serializer: runtime.NewSimpleNegotiatedSerializer(info),
			Kind:       testGroupV1.WithKind("test"),
		},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,

		// TimeoutFactory should not be needed, because the server should error
		// before calling TimeoutFactory.TimeoutCh.
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

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")
}

func TestWatchHTTPDynamicClientErrors(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		Scope:    &RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh: timeoutCh, done: doneCh},
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
	//TODO(karlkfi): Fix watch client to close the response body and stop the storage watcher after a NegotiateError
	require.False(t, watcher.IsStopped())

	// Wait for the server to call the CancelFunc returned by
	// TimeoutFactory.TimeoutCh, closing the done channel.
	// err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, doneCh)
	// require.NoError(t, err)

	// Wait for the server to call watcher.Stop, closing the result channel.
	// err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	// require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	// require.Truef(t, watcher.IsStopped(),
	// 	"Leaked watcher goroutine after request done")
}

func TestWatchHTTPTimeout(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &WatchServer{
		Scope:    &RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh: timeoutCh, done: doneCh},
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

	// Trigger server-side timeout.
	close(timeoutCh)

	// Wait for the server to call the CancelFunc returned by
	// TimeoutFactory.TimeoutCh, closing the done channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, doneCh)
	require.NoError(t, err)

	// Wait for the server to call watcher.Stop, closing the result channel.
	err = utiltesting.WaitForChannelToCloseWithTimeout(ctx, wait.ForeverTestTimeout, watcher.ResultChan())
	require.NoError(t, err)

	// Confirm watcher.Stop was called by the server.
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goroutine after request done")

	// Make sure we can't receive any more events after the watch timeout
	err = decoder.Decode(&got)
	require.Equal(t, io.EOF, err)

	// Close the response body to clean up watch client resources.
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

type fakeTimeoutFactory struct {
	timeoutCh chan time.Time
	done      chan struct{}
}

func (t *fakeTimeoutFactory) TimeoutCh() (<-chan time.Time, func() bool) {
	return t.timeoutCh, func() bool {
		defer close(t.done)
		return true
	}
}

// serveWatch will serve a watch response according to the watcher and watchServer.
// Before watchServer.HandleHTTP, an error may occur like k8s.io/apiserver/pkg/endpoints/handlers/watch.go#serveWatch does.
func serveWatch(watchServer *WatchServer, preServeErr error) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		defer watchServer.Watching.Stop()

		if preServeErr != nil {
			responsewriters.ErrorNegotiated(preServeErr, watchServer.Scope.Serializer, watchServer.Scope.Kind.GroupVersion(), w, req)
			return
		}

		watchServer.HandleHTTP(w, req)
	}
}
