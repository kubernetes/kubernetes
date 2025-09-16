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

package endpoints

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/watch"
	example "k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/endpoints/request"
	endpointstesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/registry/rest"
)

// watchJSON defines the expected JSON wire equivalent of watch.Event
type watchJSON struct {
	Type   watch.EventType `json:"type,omitempty"`
	Object json.RawMessage `json:"object,omitempty"`
}

// roundTripOrDie round trips an object to get defaults set.
func roundTripOrDie(codec runtime.Codec, object runtime.Object) runtime.Object {
	data, err := runtime.Encode(codec, object)
	if err != nil {
		panic(err)
	}
	obj, err := runtime.Decode(codec, data)
	if err != nil {
		panic(err)
	}
	return obj
}

var watchTestTable = []struct {
	t   watch.EventType
	obj runtime.Object
}{
	{watch.Added, &endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}},
	{watch.Modified, &endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}},
	{watch.Deleted, &endpointstesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}},
}

func podWatchTestTable() []struct {
	t   watch.EventType
	obj runtime.Object
} {
	// creaze lazily here in a func because podWatchTestTable can only be used after all types are registered.
	return []struct {
		t   watch.EventType
		obj runtime.Object
	}{
		{watch.Added, roundTripOrDie(codec, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})},
		{watch.Modified, roundTripOrDie(codec, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}})},
		{watch.Deleted, roundTripOrDie(codec, &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}})},
	}
}

func TestWatchWebsocket(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	_ = rest.Watcher(simpleStorage) // Give compile error if this doesn't work.
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Scheme = "ws" // Required by websocket, though the server never sees it.
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	ws, err := websocket.Dial(dest.String(), "", "http://localhost")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer apitesting.Close(t, ws)

	try := func(action watch.EventType, object runtime.Object) {
		// Send
		simpleStorage.fakeWatch.Action(action, object)
		// Test receive
		var got watchJSON
		err := websocket.JSON.Receive(ws, &got)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if got.Type != action {
			t.Errorf("Unexpected type: %v", got.Type)
		}
		gotObj, err := runtime.Decode(codec, got.Object)
		if err != nil {
			t.Fatalf("Decode error: %v\n%v", err, got)
		}
		if e, a := object, gotObj; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}

	for _, item := range watchTestTable {
		try(item.t, item.obj)
	}
	simpleStorage.fakeWatch.Stop()

	var got watchJSON
	err = websocket.JSON.Receive(ws, &got)
	if err == nil {
		t.Errorf("Unexpected non-error")
	}
}

func TestWatchWebsocketClientClose(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	_ = rest.Watcher(simpleStorage) // Give compile error if this doesn't work.
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Scheme = "ws" // Required by websocket, though the server never sees it.
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	ws, err := websocket.Dial(dest.String(), "", "http://localhost")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Ensure the websocket is closed without error
	closeWebSocket := apitesting.Close
	defer func() {
		closeWebSocket(t, ws)
	}()

	try := func(action watch.EventType, object runtime.Object) {
		// Send
		simpleStorage.fakeWatch.Action(action, object)
		// Test receive
		var got watchJSON
		err := websocket.JSON.Receive(ws, &got)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if got.Type != action {
			t.Errorf("Unexpected type: %v", got.Type)
		}
		gotObj, err := runtime.Decode(codec, got.Object)
		if err != nil {
			t.Fatalf("Decode error: %v\n%v", err, got)
		}
		if e, a := object, gotObj; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}

	// Send/receive should work
	for _, item := range watchTestTable {
		try(item.t, item.obj)
	}

	// Sending normal data should be ignored
	websocket.JSON.Send(ws, map[string]interface{}{"test": "data"})

	// Send/receive should still work
	for _, item := range watchTestTable {
		try(item.t, item.obj)
	}

	// Websocket.Close errors if called twice.
	// So disable the defer close and close immediately instead.
	closeWebSocket = apitesting.CloseNoOp
	require.NoError(t, ws.Close())

	select {
	case data, ok := <-simpleStorage.fakeWatch.ResultChan():
		if ok {
			t.Errorf("expected a closed result channel, but got watch result %#v", data)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("watcher did not close when client closed")
	}

	// Reading from a closed websocket should error
	var got watchJSON
	err = websocket.JSON.Receive(ws, &got)
	require.ErrorContains(t, err, "use of closed network connection")
}

func TestWatchClientClose(t *testing.T) {
	ctx := t.Context()
	simpleStorage := &SimpleRESTStorage{}
	_ = rest.Watcher(simpleStorage) // Give compile error if this doesn't work.
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/simples"
	dest.RawQuery = "watch=1"

	request, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	request.Header.Add("Accept", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Ensure the response body is closed without error
	defer apitesting.Close(t, response.Body)

	if response.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(response.Body)
		t.Fatalf("Unexpected response: %#v\n%s", response, string(b))
	}

	// Close response to cause a cancel on the server
	if err := response.Body.Close(); err != nil {
		t.Fatalf("Unexpected close client err: %v", err)
	}

	select {
	case data, ok := <-simpleStorage.fakeWatch.ResultChan():
		if ok {
			t.Errorf("expected a closed result channel, but got watch result %#v", data)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("watcher did not close when client closed")
	}

	// Reading from a closed response body should error
	_, err = io.Copy(io.Discard, response.Body)
	require.ErrorContains(t, err, "http: read on closed response body")
}

func TestWatchRead(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	_ = rest.Watcher(simpleStorage) // Give compile error if this doesn't work.
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/simples"
	dest.RawQuery = "watch=1"

	connectHTTP := func(ctx context.Context, accept string) (io.ReadCloser, string) {
		client := http.Client{}
		request, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		request.Header.Add("Accept", accept)

		response, err := client.Do(request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if response.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(response.Body)
			t.Fatalf("Unexpected response for accept: %q: %#v\n%s", accept, response, string(b))
		}
		return response.Body, response.Header.Get("Content-Type")
	}

	connectWebSocket := func(ctx context.Context, accept string) (io.ReadCloser, string) {
		dest := *dest
		dest.Scheme = "ws" // Required by websocket, though the server never sees it.
		config, err := websocket.NewConfig(dest.String(), "http://localhost")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		config.Header.Add("Accept", accept)
		ws, err := config.DialContext(ctx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return ws, "__default__"
	}

	cases := []struct {
		Accept              string
		ExpectedContentType string
		MediaType           string
	}{
		{
			Accept:              "application/json",
			ExpectedContentType: "application/json",
			MediaType:           "application/json",
		},
		{
			Accept:              "application/json;stream=watch",
			ExpectedContentType: "application/json", // legacy behavior
			MediaType:           "application/json",
		},
		// TODO: yaml stream serialization requires that RawExtension.MarshalJSON
		// be able to understand nested encoding (since yaml calls json.Marshal
		// rather than yaml.Marshal, which results in the raw bytes being in yaml).
		/*{
			Accept:              "application/yaml",
			ExpectedContentType: "application/yaml;stream=watch",
			MediaType:           "application/yaml",
		},*/
		{
			Accept:              "application/vnd.kubernetes.protobuf",
			ExpectedContentType: "application/vnd.kubernetes.protobuf;stream=watch",
			MediaType:           "application/vnd.kubernetes.protobuf",
		},
		{
			Accept:              "application/vnd.kubernetes.protobuf;stream=watch",
			ExpectedContentType: "application/vnd.kubernetes.protobuf;stream=watch",
			MediaType:           "application/vnd.kubernetes.protobuf",
		},
	}
	protocols := []struct {
		name        string
		selfFraming bool
		fn          func(context.Context, string) (io.ReadCloser, string)
	}{
		{name: "http", fn: connectHTTP},
		{name: "websocket", selfFraming: true, fn: connectWebSocket},
	}

	for _, protocol := range protocols {
		for textIndex, test := range cases {
			testName := fmt.Sprintf("%s-%d", protocol.name, textIndex)
			t.Run(testName, func(t *testing.T) {
				ctx := t.Context()
				info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), test.MediaType)
				if !ok || info.StreamSerializer == nil {
					t.Fatal(info)
				}
				streamSerializer := info.StreamSerializer

				r, contentType := protocol.fn(ctx, test.Accept)
				closeBody := apitesting.Close
				defer func() {
					closeBody(t, r)
				}()

				if contentType != "__default__" && contentType != test.ExpectedContentType {
					t.Errorf("Unexpected content type: %#v", contentType)
				}
				objectCodec := codecs.DecoderToVersion(info.Serializer, testInternalGroupVersion)

				var fr io.ReadCloser = r
				if !protocol.selfFraming {
					fr = streamSerializer.Framer.NewFrameReader(r)
				}
				d := streaming.NewDecoder(fr, streamSerializer.Serializer)
				// Websockets error if closed twice.
				// So disable the Body.Close and use Decoder.Close instead.
				closeBody = apitesting.CloseNoOp
				defer apitesting.Close(t, d)

				var w *watch.FakeWatcher
				for w == nil {
					w = simpleStorage.Watcher()
					time.Sleep(time.Millisecond)
				}

				for i, item := range podWatchTestTable() {
					action, object := item.t, item.obj
					name := fmt.Sprintf("%s-%s-%d", protocol.name, test.MediaType, i)

					// Send
					w.Action(action, object)
					// Test receive
					var got metav1.WatchEvent
					_, _, err := d.Decode(nil, &got)
					if err != nil {
						t.Fatalf("%s: Unexpected error: %v", name, err)
					}
					if got.Type != string(action) {
						t.Errorf("%s: Unexpected type: %v", name, got.Type)
					}

					gotObj, err := runtime.Decode(objectCodec, got.Object.Raw)
					if err != nil {
						t.Fatalf("%s: Decode error: %v", name, err)
					}
					if e, a := object, gotObj; !apiequality.Semantic.DeepEqual(e, a) {
						t.Errorf("%s: different: %s", name, cmp.Diff(e, a))
					}
				}
				w.Stop()

				var got metav1.WatchEvent
				_, _, err := d.Decode(nil, &got)
				if err == nil {
					t.Errorf("Unexpected non-error")
				}
			})
		}
	}
}

func TestWatchHTTPAccept(t *testing.T) {
	ctx := t.Context()
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	request, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	request.Header.Set("Accept", "application/XYZ")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// TODO: once this is fixed, this test will change
	if response.StatusCode != http.StatusNotAcceptable {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestWatchParamParsing(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{
		"simples":     simpleStorage,
		"simpleroots": simpleStorage,
	})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)

	rootPath := "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	namespacedPath := "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/namespaces/other/simpleroots"

	cases := []struct {
		path            string
		rawQuery        string
		resourceVersion string
		labelSelector   string
		fieldSelector   string
		namespace       string
	}{
		{
			path:            rootPath,
			rawQuery:        "resourceVersion=1234",
			resourceVersion: "1234",
			labelSelector:   "",
			fieldSelector:   "",
			namespace:       metav1.NamespaceAll,
		}, {
			path:            rootPath,
			rawQuery:        "resourceVersion=314159&fieldSelector=Host%3D&labelSelector=name%3Dfoo",
			resourceVersion: "314159",
			labelSelector:   "name=foo",
			fieldSelector:   "Host=",
			namespace:       metav1.NamespaceAll,
		}, {
			path:            rootPath,
			rawQuery:        "fieldSelector=id%3dfoo&resourceVersion=1492",
			resourceVersion: "1492",
			labelSelector:   "",
			fieldSelector:   "id=foo",
			namespace:       metav1.NamespaceAll,
		}, {
			path:            rootPath,
			rawQuery:        "",
			resourceVersion: "",
			labelSelector:   "",
			fieldSelector:   "",
			namespace:       metav1.NamespaceAll,
		},
		{
			path:            namespacedPath,
			rawQuery:        "resourceVersion=1234",
			resourceVersion: "1234",
			labelSelector:   "",
			fieldSelector:   "",
			namespace:       "other",
		}, {
			path:            namespacedPath,
			rawQuery:        "resourceVersion=314159&fieldSelector=Host%3D&labelSelector=name%3Dfoo",
			resourceVersion: "314159",
			labelSelector:   "name=foo",
			fieldSelector:   "Host=",
			namespace:       "other",
		}, {
			path:            namespacedPath,
			rawQuery:        "fieldSelector=id%3dfoo&resourceVersion=1492",
			resourceVersion: "1492",
			labelSelector:   "",
			fieldSelector:   "id=foo",
			namespace:       "other",
		}, {
			path:            namespacedPath,
			rawQuery:        "",
			resourceVersion: "",
			labelSelector:   "",
			fieldSelector:   "",
			namespace:       "other",
		},
	}

	for testIndex, test := range cases {
		testName := fmt.Sprintf("%d", testIndex)
		t.Run(testName, func(t *testing.T) {
			ctx := t.Context()
			simpleStorage.requestedLabelSelector = labels.Everything()
			simpleStorage.requestedFieldSelector = fields.Everything()
			simpleStorage.requestedResourceVersion = "5" // Prove this is set in all cases
			simpleStorage.requestedResourceNamespace = ""
			dest.Path = test.path
			dest.RawQuery = test.rawQuery

			req, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
			require.NoError(t, err)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("%v: unexpected error: %v", test.rawQuery, err)
			}
			defer apitesting.Close(t, resp.Body)
			if e, a := test.namespace, simpleStorage.requestedResourceNamespace; e != a {
				t.Errorf("%v: expected %v, got %v", test.rawQuery, e, a)
			}
			if e, a := test.resourceVersion, simpleStorage.requestedResourceVersion; e != a {
				t.Errorf("%v: expected %v, got %v", test.rawQuery, e, a)
			}
			if e, a := test.labelSelector, simpleStorage.requestedLabelSelector.String(); e != a {
				t.Errorf("%v: expected %v, got %v", test.rawQuery, e, a)
			}
			if e, a := test.fieldSelector, simpleStorage.requestedFieldSelector.String(); e != a {
				t.Errorf("%v: expected %v, got %v", test.rawQuery, e, a)
			}
		})
	}
}

func TestWatchProtocolSelection(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()
	defer server.CloseClientConnections()
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	cases := []struct {
		isWebsocket bool
		connHeader  string
	}{
		{true, "Upgrade"},
		{true, "keep-alive, Upgrade"},
		{true, "upgrade"},
		{false, "keep-alive"},
	}

	for _, test := range cases {
		testName := fmt.Sprintf("websocket:%v header:%s", test.isWebsocket, test.connHeader)
		t.Run(testName, func(t *testing.T) {
			ctx := t.Context()
			request, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			request.Header.Set("Connection", test.connHeader)
			request.Header.Set("Upgrade", "websocket")

			response, err := client.Do(request)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// The requests recognized as websocket requests based on connection
			// and upgrade headers will not also have the necessary Sec-Websocket-*
			// headers so it is expected to throw a 400
			if test.isWebsocket && response.StatusCode != http.StatusBadRequest {
				t.Errorf("Unexpected response %#v", response)
			}

			if !test.isWebsocket && response.StatusCode != http.StatusOK {
				t.Errorf("Unexpected response %#v", response)
			}
		})
	}

}

// BenchmarkWatchHTTP measures the cost of serving a watch.
func BenchmarkWatchHTTP(b *testing.B) {
	items := benchmarkItems(b)

	// use ASCII names to capture the cost of handling ASCII only self-links
	for i := range items {
		item := &items[i]
		item.Namespace = fmt.Sprintf("namespace-%d", i)
		item.Name = fmt.Sprintf("reasonable-name-%d", i)
	}

	runWatchHTTPBenchmark(b, toObjectSlice(items), "")
}

func BenchmarkWatchHTTP_UTF8(b *testing.B) {
	items := benchmarkItems(b)

	// use UTF names to capture the cost of handling UTF-8 escaping in self-links
	for i := range items {
		item := &items[i]
		item.Namespace = fmt.Sprintf("躀痢疈蜧í柢-%d", i)
		item.Name = fmt.Sprintf("翏Ŏ熡韐-%d", i)
	}

	runWatchHTTPBenchmark(b, toObjectSlice(items), "")
}

func toObjectSlice(in []example.Pod) []runtime.Object {
	var res []runtime.Object
	for _, pod := range in {
		res = append(res, &pod)
	}
	return res
}

func runWatchHTTPBenchmark(b *testing.B, items []runtime.Object, contentType string) {
	ctx := b.Context()
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	req, err := http.NewRequestWithContext(ctx, request.MethodGet, dest.String(), nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}
	req.Header.Add("Accept", contentType)

	response, err := client.Do(req)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		b.Fatalf("Unexpected response %#v", response)
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer apitesting.Close(b, response.Body)
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			b.Error(err)
		}
		wg.Done()
	}()

	actions := []watch.EventType{watch.Added, watch.Modified, watch.Deleted}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simpleStorage.fakeWatch.Action(actions[i%len(actions)], items[i%len(items)])
	}
	simpleStorage.fakeWatch.Stop()
	wg.Wait()
	b.StopTimer()
}

// BenchmarkWatchWebsocket measures the cost of serving a watch.
func BenchmarkWatchWebsocket(b *testing.B) {
	items := benchmarkItems(b)

	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Scheme = "ws" // Required by websocket, though the server never sees it.
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	ws, err := websocket.Dial(dest.String(), "", "http://localhost")
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer apitesting.Close(b, ws)
		if _, err := io.Copy(io.Discard, ws); err != nil {
			b.Error(err)
		}
		wg.Done()
	}()

	actions := []watch.EventType{watch.Added, watch.Modified, watch.Deleted}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simpleStorage.fakeWatch.Action(actions[i%len(actions)], &items[i%len(items)])
	}
	simpleStorage.fakeWatch.Stop()
	wg.Wait()
	b.StopTimer()
}

// BenchmarkWatchProtobuf measures the cost of serving a watch.
func BenchmarkWatchProtobuf(b *testing.B) {
	items := benchmarkItems(b)

	runWatchHTTPBenchmark(b, toObjectSlice(items), "application/vnd.kubernetes.protobuf")
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

func wrapCachingObject(in []example.Pod) []runtime.Object {
	var res []runtime.Object
	for _, pod := range in {
		res = append(res, &fakeCachingObject{obj: &pod})
	}
	return res
}

func BenchmarkWatchCachingObjectJSON(b *testing.B) {
	items := benchmarkItems(b)

	runWatchHTTPBenchmark(b, wrapCachingObject(items), "")
}

func BenchmarkWatchCachingObjectProtobuf(b *testing.B) {
	items := benchmarkItems(b)

	runWatchHTTPBenchmark(b, wrapCachingObject(items), "application/vnd.kubernetes.protobuf")
}
