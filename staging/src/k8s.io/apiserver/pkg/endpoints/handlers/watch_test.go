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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
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
	"k8s.io/apiserver/pkg/endpoints/metrics"
	endpointstesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
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

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
	client := http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer apitesting.Close(t, resp.Body)

	// Send error to server from storage
	errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error")).Status()
	watcher.Error(&errStatus)
	watcher.Stop()

	// Make sure we can actually watch an endpoint
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
}

func TestWatchHTTPErrorsBeforeServe(t *testing.T) {
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
		Scope: &RequestScope{
			Serializer: runtime.NewSimpleNegotiatedSerializer(info),
			Kind:       testGroupV1.WithKind("test"),
		},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: testCodecV2,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, doneCh},
	}

	statusErr := apierrors.NewInternalError(fmt.Errorf("we got an error"))
	errStatus := statusErr.Status()

	s := httptest.NewServer(serveWatch(watcher, watchServer, statusErr))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
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

	// check for leaks
	require.Truef(t, watcher.IsStopped(),
		"Leaked watcher goruntine after request done")
}

func TestWatchHTTPDynamicClientErrors(t *testing.T) {
	watcher := watch.NewFake()
	timeoutCh := make(chan time.Time)
	done := make(chan struct{})

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

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, done},
	}

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()
	defer s.CloseClientConnections()

	client := dynamic.NewForConfigOrDie(&restclient.Config{
		Host:    s.URL,
		APIPath: "/" + namedGroupPrefix,
	}).Resource(testGroupV2.WithResource("simple"))

	_, err := client.Watch(context.TODO(), metav1.ListOptions{})
	require.Equal(t, runtime.NegotiateError{Stream: true, ContentType: "testcase/json"}, err)
}

func TestWatchHTTPTimeout(t *testing.T) {
	ctx := t.Context()
	watcher := watch.NewFake()
	timeoutCh := make(chan time.Time)
	done := make(chan struct{})

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

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, done},
	}

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
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

	// Timeout and check for leaks
	close(timeoutCh)
	select {
	case <-done:
		eventCh := watcher.ResultChan()
		select {
		case _, opened := <-eventCh:
			if opened {
				t.Errorf("Watcher received unexpected event")
			}
			if !watcher.IsStopped() {
				t.Errorf("Watcher is not stopped")
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Leaked watch on timeout")
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Failed to stop watcher after %s of timeout signal", wait.ForeverTestTimeout.String())
	}

	// Make sure we can't receive any more events through the timeout watch
	err = decoder.Decode(&got)
	require.Equal(t, io.EOF, err)
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
func serveWatch(watcher watch.Interface, watchServer *WatchServer, preServeErr error) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		defer watcher.Stop()

		if preServeErr != nil {
			responsewriters.ErrorNegotiated(preServeErr, watchServer.Scope.Serializer, watchServer.Scope.Kind.GroupVersion(), w, req)
			return
		}

		watchServer.HandleHTTP(w, req)
	}
}

type fakeCachingObject struct {
	obj runtime.Object

	once sync.Once
	raw  []byte
	err  error
}

func (f *fakeCachingObject) CacheEncode(_ runtime.Identifier, encode func(runtime.Object, io.Writer) error, w io.Writer) error {
	f.once.Do(func() {
		buffer := bytes.NewBuffer(nil)
		f.err = encode(f.obj, buffer)
		f.raw = buffer.Bytes()
	})

	if f.err != nil {
		return f.err
	}

	_, err := w.Write(f.raw)
	return err
}

func (f *fakeCachingObject) GetObject() runtime.Object {
	return f.obj
}

func (f *fakeCachingObject) GetObjectKind() schema.ObjectKind {
	return f.obj.GetObjectKind()
}

func (f *fakeCachingObject) DeepCopyObject() runtime.Object {
	return &fakeCachingObject{obj: f.obj.DeepCopyObject()}
}

var _ runtime.CacheableObject = &fakeCachingObject{}
var _ runtime.Object = &fakeCachingObject{}

func TestWatchEventSizes(t *testing.T) {
	metrics.Register()
	gvr := schema.GroupVersionResource{Group: "group", Version: "version", Resource: "resource"}
	testCases := []struct {
		name     string
		event    watch.Event
		wantSize int64
	}{
		{
			name: "regular object",
			event: watch.Event{
				Type:   watch.Added,
				Object: &endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			},
			wantSize: 2 * 98,
		},
		{
			name: "cached object",
			event: watch.Event{
				Type: watch.Added,
				Object: &fakeCachingObject{obj: &runtime.Unknown{
					Raw:         []byte(`{"kind":"Simple","apiVersion":"v1","metadata":{"name":"foo"}}`),
					ContentType: runtime.ContentTypeJSON,
				}},
			},
			wantSize: 2 * 88,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			metrics.WatchEventsSizes.Reset()
			metrics.WatchEvents.Reset()

			watcher := watch.NewFake()
			timeoutCh := make(chan time.Time)
			doneCh := make(chan struct{})

			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
			require.True(t, ok)
			require.NotNil(t, info.StreamSerializer)
			serializer := info.StreamSerializer

			watchServer := &WatchServer{
				Scope: &RequestScope{
					Resource: gvr,
				},
				Watching:        watcher,
				MediaType:       "application/json",
				Framer:          serializer.Framer,
				Encoder:         testCodecV2,
				EmbeddedEncoder: testCodecV2,
				TimeoutFactory:  &fakeTimeoutFactory{timeoutCh: timeoutCh, done: doneCh},
			}

			s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
			defer s.Close()

			// Setup a client
			dest, _ := url.Parse(s.URL)
			dest.Path = "/" + namedGroupPrefix + "/" + testGroupV2.Group + "/" + testGroupV2.Version + "/simple"
			dest.RawQuery = "watch=true"

			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, dest.String(), nil)
			client := http.Client{}
			resp, err := client.Do(req)
			require.NoError(t, err)
			defer apitesting.Close(t, resp.Body)

			// Send object twice so that in case of caching the cached version is used.
			watcher.Action(tc.event.Type, tc.event.Object)
			watcher.Action(tc.event.Type, tc.event.Object)

			close(timeoutCh)
			<-doneCh

			expected := fmt.Sprintf(`# HELP apiserver_watch_events_sizes [ALPHA] Watch event size distribution in bytes
# TYPE apiserver_watch_events_sizes histogram
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="1024"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="2048"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="4096"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="8192"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="16384"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="32768"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="65536"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="131072"} 2
apiserver_watch_events_sizes_bucket{group="group",resource="resource",version="version",le="+Inf"} 2
apiserver_watch_events_sizes_sum{group="group",resource="resource",version="version"} %d
apiserver_watch_events_sizes_count{group="group",resource="resource",version="version"} 2

# HELP apiserver_watch_events_total [ALPHA] Number of events sent in watch clients
# TYPE apiserver_watch_events_total counter
apiserver_watch_events_total{group="group",resource="resource",version="version"} 2
`, tc.wantSize)

			err = metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_watch_events_sizes", "apiserver_watch_events_total")
			require.NoError(t, err)
		})
	}
}
