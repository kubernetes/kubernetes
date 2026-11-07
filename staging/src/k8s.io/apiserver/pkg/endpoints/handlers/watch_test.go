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
	"errors"
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
	"golang.org/x/net/http2"
	"k8s.io/apimachinery/pkg/api/apitesting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	endpointstesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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

			expected := fmt.Sprintf(`# HELP apiserver_watch_events_sizes [BETA] Watch event size distribution in bytes
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

# HELP apiserver_watch_events_total [BETA] Number of events sent in watch clients
# TYPE apiserver_watch_events_total counter
apiserver_watch_events_total{group="group",resource="resource",version="version"} 2
`, tc.wantSize)

			err = metricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), "apiserver_watch_events_sizes", "apiserver_watch_events_total")
			require.NoError(t, err)
		})
	}
}

// recordingResponseWriter captures individual Write sizes and cumulative body
// lengths at each Flush call.
type recordingResponseWriter struct {
	mu       sync.Mutex
	header   http.Header
	body     bytes.Buffer
	writes   []int
	flushed  []int
	onFlush  func(flushedCount int)
	writeErr error
}

func newRecordingResponseWriter() *recordingResponseWriter {
	return &recordingResponseWriter{header: make(http.Header)}
}

func (w *recordingResponseWriter) Header() http.Header { return w.header }

func (w *recordingResponseWriter) WriteHeader(int) {}

func (w *recordingResponseWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.writeErr != nil {
		return 0, w.writeErr
	}
	w.writes = append(w.writes, len(p))
	return w.body.Write(p)
}

func (w *recordingResponseWriter) Flush() {
	w.mu.Lock()
	w.flushed = append(w.flushed, w.body.Len())
	n := len(w.flushed)
	onFlush := w.onFlush
	w.mu.Unlock()
	if onFlush != nil {
		onFlush(n)
	}
}

type countingEncoder struct {
	runtime.Encoder
	count    int
	onEncode func(count int)
}

func (e *countingEncoder) Encode(obj runtime.Object, w io.Writer) error {
	if err := e.Encoder.Encode(obj, w); err != nil {
		return err
	}
	e.count++
	if e.onEncode != nil {
		e.onEncode(e.count)
	}
	return nil
}

func newBatchTestWatchServer(t testing.TB, watcher watch.Interface, embeddedEncoder runtime.Encoder) *WatchServer {
	t.Helper()
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	require.True(t, ok)
	require.NotNil(t, info.StreamSerializer)

	return &WatchServer{
		Scope:           &RequestScope{},
		Watching:        watcher,
		MediaType:       runtime.ContentTypeJSON,
		Framer:          info.StreamSerializer.Framer,
		Encoder:         testCodecV2,
		EmbeddedEncoder: embeddedEncoder,
		TimeoutFactory:  &fakeTimeoutFactory{timeoutCh: make(chan time.Time), done: make(chan struct{})},
	}
}

func newBatchTestRequest(ctx context.Context) *http.Request {
	return httptest.NewRequestWithContext(ctx, http.MethodGet, "/"+namedGroupPrefix+"/"+testGroupV2.Group+"/"+testGroupV2.Version+"/simple?watch=true", nil)
}

func expectedWatchNames(n int) []string {
	names := make([]string, n)
	for i := range names {
		names[i] = fmt.Sprintf("item-%d", i)
	}
	return names
}

func decodeJSONWatchNames(t *testing.T, body []byte) []string {
	t.Helper()
	var names []string
	dec := json.NewDecoder(bytes.NewReader(body))
	for {
		var got watchJSON
		err := dec.Decode(&got)
		if errors.Is(err, io.EOF) {
			return names
		}
		require.NoError(t, err)
		var obj endpointstesting.Simple
		require.NoError(t, json.Unmarshal(got.Object, &obj))
		names = append(names, obj.Name)
	}
}

func TestWatchHTTPBatchesQueuedEvents(t *testing.T) {
	// 100 small (~100 B) events queue ~10 KB, well under watchBatchSize,
	// so they must leave in a single ResponseWriter.Write.
	const n = 100
	watcher := watch.NewFakeWithChanSize(n, false)
	for _, name := range expectedWatchNames(n) {
		watcher.Add(&endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}
	watcher.Stop()

	ws := newBatchTestWatchServer(t, watcher, testCodecV2)
	w := newRecordingResponseWriter()
	ws.HandleHTTP(w, newBatchTestRequest(t.Context()))

	require.Equal(t, expectedWatchNames(n), decodeJSONWatchNames(t, w.body.Bytes()))
	require.Len(t, w.writes, 1, "expected queued frames under watchBatchSize to combine into 1 write")
	require.Equal(t, []int{0, w.body.Len()}, w.flushed)
}

func TestWatchHTTPWritesThroughWithoutQueue(t *testing.T) {
	// A few events, each sent only after the previous one was flushed, so the
	// queue is always empty. The count only needs to be >1 to show one
	// write and one flush per event.
	const n = 5
	// Buffered like the cacher's result channel, to show write-through depends
	// on queue length, not on the channel being unbuffered.
	watcher := watch.NewFakeWithChanSize(16, false)
	ws := newBatchTestWatchServer(t, watcher, testCodecV2)

	flushedCh := make(chan struct{}, n+1)
	w := newRecordingResponseWriter()
	w.onFlush = func(int) {
		flushedCh <- struct{}{}
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		ws.HandleHTTP(w, newBatchTestRequest(t.Context()))
	}()

	// Wait for BeginStream header flush.
	<-flushedCh
	for _, name := range expectedWatchNames(n) {
		watcher.Add(&endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: name}})
		<-flushedCh
	}
	watcher.Stop()
	<-done

	require.Equal(t, expectedWatchNames(n), decodeJSONWatchNames(t, w.body.Bytes()))
	require.Len(t, w.writes, n)
	require.Len(t, w.flushed, n+1)
	require.Zero(t, w.flushed[0])
	for i := 1; i <= n; i++ {
		require.Greater(t, w.flushed[i], w.flushed[i-1])
	}
	require.Equal(t, w.body.Len(), w.flushed[n])
}

// TestWatchHTTPFlushesBatchOnDrain checks that frames already batched reach the
// client when the server drains watches while a backlog is still queued.
func TestWatchHTTPFlushesBatchOnDrain(t *testing.T) {
	// Queue 50 events and drain the watch after the 5th is encoded, so the
	// handler exits mid-batch with ~45 still queued.
	const (
		queued     = 50
		drainAfter = 5
	)
	watcher := watch.NewFakeWithChanSize(queued, false)
	for _, name := range expectedWatchNames(queued) {
		watcher.Add(&endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}

	drainCh := make(chan struct{})
	enc := &countingEncoder{Encoder: testCodecV2}
	enc.onEncode = func(count int) {
		if count == drainAfter {
			close(drainCh)
		}
	}

	ws := newBatchTestWatchServer(t, watcher, enc)
	ws.DrainWatchCh = drainCh
	w := newRecordingResponseWriter()
	ws.HandleHTTP(w, newBatchTestRequest(t.Context()))

	require.GreaterOrEqual(t, enc.count, drainAfter)
	// Go selects uniformly at random between the closed drain channel and ch,
	// so draining all 45 remaining events before exit has probability 2^-45.
	require.Less(t, enc.count, queued, "watch should exit while backlog remains in channel")
	// Every event already encoded before exit is flushed by defer rw.Close().
	require.Equal(t, expectedWatchNames(enc.count), decodeJSONWatchNames(t, w.body.Bytes()))
	require.Len(t, w.writes, 1)
}

func TestWatchHTTPWatchListFlushesBookmarkMidBacklog(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)

	// Both sides of the initial-events-end bookmark are non-empty and queued
	// together, so the bookmark flush is the only thing splitting them into
	// exactly two writes.
	const (
		initialCount = 10
		liveCount    = 10
	)
	total := initialCount + 1 + liveCount
	watcher := watch.NewFakeWithChanSize(total, false)
	for i := range initialCount {
		watcher.Add(&endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("init-%d", i)}})
	}
	watcher.Action(watch.Bookmark, &endpointstesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "bookmark",
			ResourceVersion: "100",
			Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
		},
	})
	for i := range liveCount {
		watcher.Add(&endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("live-%d", i)}})
	}
	watcher.Stop()

	var hookCalls int
	ws := newBatchTestWatchServer(t, watcher, testCodecV2)
	ws.isWatchListRequest = true
	ws.watchListCompleteHook = func() { hookCalls++ }

	ctx := apirequest.WithReceivedTimestamp(t.Context(), time.Now())
	w := newRecordingResponseWriter()
	ws.HandleHTTP(w, newBatchTestRequest(ctx))

	require.Equal(t, 1, hookCalls)
	// Batch 1 flushes at the initial-events-end bookmark even with live events
	// still queued; Batch 2 flushes when the remaining live backlog empties.
	require.Len(t, w.writes, 2)
	require.Len(t, w.flushed, 3)
	require.Positive(t, w.flushed[1])
	require.Equal(t, w.body.Len(), w.flushed[2])
	require.Len(t, decodeJSONWatchNames(t, w.body.Bytes()), total)
}

func TestPlainResponseWriterFlushAndCloseRelease(t *testing.T) {
	t.Run("failed flush releases buffer", func(t *testing.T) {
		boom := errors.New("broken pipe")
		failing := newRecordingResponseWriter()
		failing.writeErr = boom

		pw := &plainResponseWriter{w: failing, flusher: failing}
		pw.StartBatch()
		_, err := pw.Write([]byte("queued-frame"))
		require.NoError(t, err)

		require.ErrorIs(t, pw.Flush(), boom)
		require.Nil(t, pw.bw, "failed Flush must release pooled buffer")
		require.Empty(t, failing.flushed, "underlying Flusher should not run when batch flush fails")
		require.NoError(t, pw.Close(), "Close after failed Flush should be a clean no-op")
	})

	t.Run("close skips flush when request context is done", func(t *testing.T) {
		done := make(chan struct{})
		close(done)
		failing := newRecordingResponseWriter()
		failing.writeErr = errors.New("client disconnected")

		pw := &plainResponseWriter{w: failing, flusher: failing, done: done}
		pw.StartBatch()
		_, err := pw.Write([]byte("buffered-before-disconnect"))
		require.NoError(t, err)

		require.NoError(t, pw.Close())
		require.Nil(t, pw.bw)
		require.Empty(t, failing.writes)
	})

	t.Run("close skips flush after write failure", func(t *testing.T) {
		boom := errors.New("broken pipe")
		failing := newRecordingResponseWriter()
		failing.writeErr = boom

		pw := &plainResponseWriter{w: failing, flusher: failing}
		pw.StartBatch()
		// Exceed watchBatchSize so Write attempts to flush to failing and records writeFailed.
		_, err := pw.Write(bytes.Repeat([]byte("a"), watchBatchSize+1))
		require.ErrorIs(t, err, boom)

		require.NoError(t, pw.Close(), "Close should not re-return a sticky Write error")
		require.Nil(t, pw.bw)
	})
}

func startHTTP2WatchServer(tb testing.TB, handler http.Handler) (*httptest.Server, *http.Client) {
	tb.Helper()
	srv := httptest.NewUnstartedServer(handler)
	// Match kube-apiserver's SecureServingInfo HTTP/2 framing parameters.
	const resourceBody99Percentile = 256 * 1024
	http2Opts := &http2.Server{
		MaxConcurrentStreams:         100,
		MaxReadFrameSize:             resourceBody99Percentile,
		MaxUploadBufferPerStream:     resourceBody99Percentile,
		MaxUploadBufferPerConnection: resourceBody99Percentile * 100,
	}
	require.NoError(tb, http2.ConfigureServer(srv.Config, http2Opts))
	srv.TLS = srv.Config.TLSConfig
	srv.StartTLS()
	tb.Cleanup(srv.Close)

	baseTransport := srv.Client().Transport.(*http.Transport).Clone()
	h2Transport, err := http2.ConfigureTransports(baseTransport)
	require.NoError(tb, err)
	h2Transport.DisableCompression = true
	return srv, &http.Client{Transport: baseTransport}
}

// BenchmarkWatchHTTP2Backlog measures sustained single-stream watch throughput
// through WatchServer.HandleHTTP over HTTP/2 + TLS when ResultChan stays
// backlogged, comparing protobuf and JSON ~9.5 KB Pod watch events.
func BenchmarkWatchHTTP2Backlog(b *testing.B) {
	metrics.Register()

	encodings := []struct {
		name      string
		mediaType string
	}{
		{name: "protobuf", mediaType: runtime.ContentTypeProtobuf},
		{name: "json", mediaType: runtime.ContentTypeJSON},
	}

	// ~9.5 KB serialized Pod frame (>2x HTTP/2's 4 KiB handlerChunkWriteSize).
	basePod := &examplev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: examplev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "pod-0",
			Namespace:   "default",
			Annotations: map[string]string{"pad": strings.Repeat("x", 9*1024)},
		},
	}
	gvr := examplev1.SchemeGroupVersion.WithResource("pods")

	for _, enc := range encodings {
		b.Run("encoding="+enc.name, func(b *testing.B) {
			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), enc.mediaType)
			require.True(b, ok)
			require.NotNil(b, info.StreamSerializer)

			streamMediaType := enc.mediaType
			if enc.mediaType != runtime.ContentTypeJSON {
				streamMediaType += ";stream=watch"
			}
			eventEncoder := codecs.EncoderForVersion(info.StreamSerializer.Serializer, examplev1.SchemeGroupVersion)
			embeddedEncoder := codecs.EncoderForVersion(info.Serializer, examplev1.SchemeGroupVersion)

			// Pre-warm CacheableObject serialization (mirroring watchCache's
			// cachingObject) and measure exact per-event wire bytes.
			cachedPod := &fakeCachingObject{obj: basePod.DeepCopy()}
			warmEvent := watch.Event{Type: watch.Modified, Object: cachedPod}
			var warmFrame bytes.Buffer
			warmEnc := newWatchEncoder(b.Context(), gvr, embeddedEncoder, eventEncoder, info.StreamSerializer.Framer.NewFrameWriter(&warmFrame))
			require.NoError(b, warmEnc.Encode(warmEvent))
			frameBytes := warmFrame.Len()

			// Deep enough that the producer stays ahead of HandleHTTP, keeping the
			// stream backlogged for the whole run.
			ch := make(chan watch.Event, 2048)
			handlerDone := make(chan struct{})
			srv, client := startHTTP2WatchServer(b, http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(handlerDone)
				ws := &WatchServer{
					Scope:           &RequestScope{Resource: gvr},
					Watching:        watch.NewProxyWatcher(ch),
					MediaType:       streamMediaType,
					Framer:          info.StreamSerializer.Framer,
					Encoder:         eventEncoder,
					EmbeddedEncoder: embeddedEncoder,
					TimeoutFactory:  &fakeTimeoutFactory{timeoutCh: make(chan time.Time), done: make(chan struct{})},
				}
				ws.HandleHTTP(w, req)
			}))

			req, err := http.NewRequestWithContext(b.Context(), http.MethodGet, srv.URL+"/api/v1/pods?watch=true", nil)
			require.NoError(b, err)
			resp, err := client.Do(req)
			require.NoError(b, err)
			require.Equal(b, http.StatusOK, resp.StatusCode)
			require.Equal(b, 2, resp.ProtoMajor, "benchmark requires HTTP/2")

			type readResult struct {
				n   int64
				err error
			}
			readDone := make(chan readResult, 1)
			go func() {
				n, copyErr := io.Copy(io.Discard, resp.Body)
				_ = resp.Body.Close()
				readDone <- readResult{n: n, err: copyErr}
			}()

			b.ReportAllocs()
			b.SetBytes(int64(frameBytes))
			b.ResetTimer()
			start := time.Now()

			for range b.N {
				ch <- warmEvent
			}
			close(ch)

			res := <-readDone
			<-handlerDone

			elapsed := time.Since(start)
			b.StopTimer()

			require.NoError(b, res.err)
			require.Equal(b, int64(b.N)*int64(frameBytes), res.n)
			b.ReportMetric(float64(b.N)/elapsed.Seconds(), "events/s")
			b.ReportMetric(float64(frameBytes), "B/frame")
		})
	}
}
