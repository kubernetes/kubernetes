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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/websocket"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	example "k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	apitesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
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
	{watch.Added, &apitesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}},
	{watch.Modified, &apitesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}},
	{watch.Deleted, &apitesting.Simple{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}},
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

func TestWatchWebsocketServerClose(t *testing.T) {
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
	defer func(t *testing.T) {
		require.NoError(t, ws.Close())
	}(t)

	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
	if err != nil {
		t.Fatalf("waiting for watcher: %v", err)
	}

	try := func(action watch.EventType, object runtime.Object) {
		// Send
		storageWatcher.Action(action, object)
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

	// Storage closes the watcher before the client is done reading.
	storageWatcher.Close()

	var got watchJSON
	err = websocket.JSON.Receive(ws, &got)
	if !errors.Is(err, io.EOF) {
		t.Errorf("Expected EOF error, but got: %v", err)
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
	defer func(t *testing.T) {
		require.NoError(t, ws.Close())
	}(t)

	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
	if err != nil {
		t.Fatalf("waiting for watcher: %v", err)
	}

	try := func(action watch.EventType, object runtime.Object) {
		// Send
		storageWatcher.Action(action, object)
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

	// Storage waits for the server to stop reading, then closes the watcher.
	go func() {
		<-storageWatcher.StopChan()
		storageWatcher.Close()
	}()

	// Client requests a close
	ws.Close()

	select {
	case data, ok := <-storageWatcher.ResultChan():
		if ok {
			t.Errorf("expected a closed result channel, but got watch result %#v", data)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("watcher did not close when client closed")
	}

	var got watchJSON
	err = websocket.JSON.Receive(ws, &got)
	if err == nil || !strings.Contains(err.Error(), "use of closed network connection") {
		t.Errorf("Expected closed connection error, but got: %v", err)
	}
}

func TestWatchClientClose(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	_ = rest.Watcher(simpleStorage) // Give compile error if this doesn't work.
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/simples"
	dest.RawQuery = "watch=1"

	request, err := http.NewRequest("GET", dest.String(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	request.Header.Add("Accept", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func(t *testing.T) {
		require.NoError(t, response.Body.Close())
	}(t)

	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
	if err != nil {
		t.Fatalf("waiting for watcher: %v", err)
	}

	if response.StatusCode != http.StatusOK {
		b, _ := ioutil.ReadAll(response.Body)
		t.Fatalf("Unexpected response: %#v\n%s", response, string(b))
	}

	// Storage waits for the server to stop reading, then closes the watcher.
	go func() {
		<-storageWatcher.StopChan()
		storageWatcher.Close()
	}()

	// Close response to cause a cancel on the server
	if err := response.Body.Close(); err != nil {
		t.Fatalf("Unexpected close client err: %v", err)
	}

	select {
	case data, ok := <-storageWatcher.ResultChan():
		if ok {
			t.Errorf("expected a closed result channel, but got watch result %#v", data)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("watcher did not close when client closed")
	}
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

	connectHTTP := func(accept string) (io.ReadCloser, string) {
		client := http.Client{}
		request, err := http.NewRequest("GET", dest.String(), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		request.Header.Add("Accept", accept)

		response, err := client.Do(request)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if response.StatusCode != http.StatusOK {
			b, _ := ioutil.ReadAll(response.Body)
			t.Fatalf("Unexpected response for accept: %q: %#v\n%s", accept, response, string(b))
		}
		return response.Body, response.Header.Get("Content-Type")
	}

	connectWebSocket := func(accept string) (io.ReadCloser, string) {
		dest := *dest
		dest.Scheme = "ws" // Required by websocket, though the server never sees it.
		config, err := websocket.NewConfig(dest.String(), "http://localhost")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		config.Header.Add("Accept", accept)
		ws, err := websocket.DialConfig(config)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return ws, "__default__"
	}

	testCases := []struct {
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
		fn          func(string) (io.ReadCloser, string)
	}{
		{name: "http", fn: connectHTTP},
		{name: "websocket", selfFraming: true, fn: connectWebSocket},
	}

	for _, protocol := range protocols {
		for _, test := range testCases {
			func() {
				info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), test.MediaType)
				if !ok || info.StreamSerializer == nil {
					t.Fatal(info)
				}
				streamSerializer := info.StreamSerializer

				r, contentType := protocol.fn(test.Accept)
				defer func(t *testing.T) {
					require.NoError(t, r.Close())
				}(t)

				if contentType != "__default__" && contentType != test.ExpectedContentType {
					t.Errorf("Unexpected content type: %#v", contentType)
				}
				objectCodec := codecs.DecoderToVersion(info.Serializer, testInternalGroupVersion)

				var fr io.ReadCloser = r
				if !protocol.selfFraming {
					fr = streamSerializer.Framer.NewFrameReader(r)
				}
				d := streaming.NewDecoder(fr, streamSerializer.Serializer)

				ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
				defer cancel()
				storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
				if err != nil {
					t.Fatalf("waiting for watcher: %v", err)
				}

				for i, item := range podWatchTestTable() {
					action, object := item.t, item.obj
					name := fmt.Sprintf("%s-%s-%d", protocol.name, test.MediaType, i)

					// Send
					storageWatcher.Action(action, object)
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

				// Storage waits for the server to stop reading, then closes the watcher.
				go func() {
					<-storageWatcher.StopChan()
					storageWatcher.Close()
				}()

				// Client closes the decoder
				assert.NoError(t, d.Close())

				var got metav1.WatchEvent
				_, _, err = d.Decode(nil, &got)
				// TODO: Why is there a race condition that makes the error non-deterministic?
				switch {
				case err == nil:
					t.Errorf("Expected closed connection error, but got: %v", err)
				case strings.Contains(err.Error(), "use of closed network connection"):
				case strings.Contains(err.Error(), "http: read on closed response body"):
				default:
					t.Errorf("Expected closed connection error, but got: %v", err)

				}
			}()
		}
	}
}

func TestWatchHTTPAccept(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	request, err := http.NewRequest("GET", dest.String(), nil)
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

	table := []struct {
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

	for _, item := range table {
		simpleStorage.requestedLabelSelector = labels.Everything()
		simpleStorage.requestedFieldSelector = fields.Everything()
		simpleStorage.requestedResourceVersion = "5" // Prove this is set in all cases
		simpleStorage.requestedResourceNamespace = ""
		dest.Path = item.path
		dest.RawQuery = item.rawQuery
		resp, err := http.Get(dest.String())
		if err != nil {
			t.Errorf("%v: unexpected error: %v", item.rawQuery, err)
			continue
		}
		resp.Body.Close()
		if e, a := item.namespace, simpleStorage.requestedResourceNamespace; e != a {
			t.Errorf("%v: expected %v, got %v", item.rawQuery, e, a)
		}
		if e, a := item.resourceVersion, simpleStorage.requestedResourceVersion; e != a {
			t.Errorf("%v: expected %v, got %v", item.rawQuery, e, a)
		}
		if e, a := item.labelSelector, simpleStorage.requestedLabelSelector.String(); e != a {
			t.Errorf("%v: expected %v, got %v", item.rawQuery, e, a)
		}
		if e, a := item.fieldSelector, simpleStorage.requestedFieldSelector.String(); e != a {
			t.Errorf("%v: expected %v, got %v", item.rawQuery, e, a)
		}
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

	table := []struct {
		isWebsocket bool
		connHeader  string
	}{
		{true, "Upgrade"},
		{true, "keep-alive, Upgrade"},
		{true, "upgrade"},
		{false, "keep-alive"},
	}

	for _, item := range table {
		request, err := http.NewRequest("GET", dest.String(), nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		request.Header.Set("Connection", item.connHeader)
		request.Header.Set("Upgrade", "websocket")

		response, err := client.Do(request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		// The requests recognized as websocket requests based on connection
		// and upgrade headers will not also have the necessary Sec-Websocket-*
		// headers so it is expected to throw a 400
		if item.isWebsocket && response.StatusCode != http.StatusBadRequest {
			t.Errorf("Unexpected response %#v", response)
		}

		if !item.isWebsocket && response.StatusCode != http.StatusOK {
			t.Errorf("Unexpected response %#v", response)
		}
	}

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
func serveWatch(watcher watch.Interface, watchServer *handlers.WatchServer, preServeErr error) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		if preServeErr != nil {
			// Cleanup the Watcher
			defer watcher.Stop()
			// Cleanup the TimeoutFactory
			_, cleanup := watchServer.TimeoutFactory.TimeoutCh()
			defer cleanup()
			responsewriters.ErrorNegotiated(preServeErr, watchServer.Scope.Serializer, watchServer.Scope.Kind.GroupVersion(), w, req)
			return
		}

		watchServer.HandleHTTP(w, req)
	}
}

func TestWatchHTTPErrors(t *testing.T) {
	watcher := watch.NewFake()
	defer watcher.Close()
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &handlers.WatchServer{
		Scope:    &handlers.RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         newCodec,
		EmbeddedEncoder: newCodec,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, doneCh},
	}

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, _ := http.NewRequest("GET", dest.String(), nil)
	client := http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error")).Status()
	watcher.Error(&errStatus)
	// Storage closes watcher immediately after error event
	watcher.Close()

	// Make sure we can actually watch an endpoint
	decoder := json.NewDecoder(resp.Body)
	var got watchJSON
	err = decoder.Decode(&got)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if got.Type != watch.Error {
		t.Fatalf("unexpected watch type: %#v", got)
	}
	status := &metav1.Status{}
	if err := json.Unmarshal(got.Object, status); err != nil {
		t.Fatal(err)
	}
	if status.Kind != "Status" || status.APIVersion != "v1" || status.Code != 500 || status.Status != "Failure" || !strings.Contains(status.Message, "we got an error") {
		t.Fatalf("error: %#v", status)
	}

	// Validate the handler stopped the watcher
	select {
	case _, ok := <-watcher.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected watcher to be stopped")
	}

	// Validate the handler cleaned up the TimeoutFactory
	select {
	case _, ok := <-doneCh:
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected done channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected timeout factory to be done")
	}
}

func TestWatchHTTPErrorsBeforeServe(t *testing.T) {
	watcher := watch.NewFake()
	defer watcher.Close()
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &handlers.WatchServer{
		Scope: &handlers.RequestScope{
			Serializer: runtime.NewSimpleNegotiatedSerializer(info),
			Kind:       testGroupVersion.WithKind("test"),
		},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         newCodec,
		EmbeddedEncoder: newCodec,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, doneCh},
	}

	errStatus := apierrors.NewInternalError(fmt.Errorf("we got an error"))

	s := httptest.NewServer(serveWatch(watcher, watchServer, errStatus))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, err := http.NewRequest("GET", dest.String(), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	client := http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// We had already got an error before watch serve started
	decoder := json.NewDecoder(resp.Body)
	var status *metav1.Status
	err = decoder.Decode(&status)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if status.Kind != "Status" || status.APIVersion != "v1" || status.Code != 500 || status.Status != "Failure" || !strings.Contains(status.Message, "we got an error") {
		t.Fatalf("error: %#v", status)
	}

	// Storage waits for the server to stop reading, then closes the watcher.
	go func() {
		<-watcher.StopChan()
		watcher.Close()
	}()

	// Validate the handler stopped the watcher
	select {
	case _, ok := <-watcher.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected watcher to be stopped")
	}

	// Validate the handler cleaned up the TimeoutFactory
	select {
	case _, ok := <-doneCh:
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected done channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected timeout factory to be done")
	}
}

func TestWatchHTTPDynamicClientErrors(t *testing.T) {
	watcher := watch.NewFake()
	defer watcher.Close()
	timeoutCh := make(chan time.Time)
	defer close(timeoutCh)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &handlers.WatchServer{
		Scope:    &handlers.RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         newCodec,
		EmbeddedEncoder: newCodec,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, doneCh},
	}

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()
	defer s.CloseClientConnections()

	client := dynamic.NewForConfigOrDie(&restclient.Config{
		Host:    s.URL,
		APIPath: "/" + prefix,
	}).Resource(newGroupVersion.WithResource("simple"))

	_, err := client.Watch(context.TODO(), metav1.ListOptions{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var nErr runtime.NegotiateError
	if !errors.As(err, &nErr) {
		t.Fatalf("unexpected error: %v", err)
	}
	if !nErr.Stream {
		t.Fatalf("unexpected error: %v", err)
	}
	if nErr.ContentType != "testcase/json" {
		t.Fatalf("unexpected error: %v", err)
	}

	// Storage waits for the server to stop reading, then closes the watcher.
	go func() {
		<-watcher.StopChan()
		watcher.Close()
	}()

	// client.Watch errored client-side while trying to create a stream decoder,
	// and the storage watcher never received any events. So the server remains
	// blocked until either the connection closes, the server times out, or the
	// request times out. So shut down the connection to terminate the request
	// and wait until the handler stops the watcher.
	s.CloseClientConnections()

	// Validate the handler stopped the watcher
	select {
	case _, ok := <-watcher.StopChan():
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected stop channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected watcher to be stopped")
	}

	// Validate the handler cleaned up the TimeoutFactory
	select {
	case _, ok := <-doneCh:
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected done channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected timeout factory to be done")
	}
}

func TestWatchHTTPTimeout(t *testing.T) {
	watcher := watch.NewFake()
	defer watcher.Close()
	timeoutCh := make(chan time.Time)
	doneCh := make(chan struct{})

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok || info.StreamSerializer == nil {
		t.Fatal(info)
	}
	serializer := info.StreamSerializer

	// Setup a new watchserver
	watchServer := &handlers.WatchServer{
		Scope:    &handlers.RequestScope{},
		Watching: watcher,

		MediaType:       "testcase/json",
		Framer:          serializer.Framer,
		Encoder:         newCodec,
		EmbeddedEncoder: newCodec,

		TimeoutFactory: &fakeTimeoutFactory{timeoutCh, doneCh},
	}

	s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
	defer s.Close()

	// Setup a client
	dest, _ := url.Parse(s.URL)
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/simple"
	dest.RawQuery = "watch=true"

	req, _ := http.NewRequest("GET", dest.String(), nil)
	client := http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	watcher.Add(&apitesting.Simple{TypeMeta: metav1.TypeMeta{APIVersion: newGroupVersion.String()}})

	// Storage waits for the server to stop reading, then closes the watcher.
	go func() {
		<-watcher.StopChan()
		watcher.Close()
	}()

	// Make sure we can actually watch an endpoint
	decoder := json.NewDecoder(resp.Body)
	var got watchJSON
	err = decoder.Decode(&got)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer func(t *testing.T) {
		require.NoError(t, resp.Body.Close())
	}(t)

	// Timeout should cause the server to close the watch connection,
	// which should stop the storage watcher, which should close the result channel.
	close(timeoutCh)
	// Check for leaks
	select {
	case <-doneCh:
		eventCh := watcher.ResultChan()
	loop:
		for {
			select {
			case e, opened := <-eventCh:
				if !opened {
					// result channel closed, as expected
					break loop
				}
				t.Errorf("Watcher received unexpected event: %v", e)
			case <-time.After(wait.ForeverTestTimeout):
				t.Errorf("Leaked watch on timeout")
				break loop
			}
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("Failed to stop watcher after %s of timeout signal", wait.ForeverTestTimeout.String())
	}

	// Make sure we can't receive any more events through the timeout watch
	err = decoder.Decode(&got)
	if err != io.EOF {
		t.Errorf("Unexpected non-error")
	}

	// Validate the handler cleaned up the TimeoutFactory
	select {
	case _, ok := <-doneCh:
		if !ok {
			// closed as expected
			break
		}
		t.Error("Unexpected done channel event")
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Expected timeout factory to be done")
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
	simpleStorage := &SimpleRESTStorage{}
	handler := handle(map[string]rest.Storage{"simples": simpleStorage})
	server := httptest.NewServer(handler)
	defer server.Close()
	client := http.Client{}

	dest, _ := url.Parse(server.URL)
	dest.Path = "/" + prefix + "/" + newGroupVersion.Group + "/" + newGroupVersion.Version + "/watch/simples"
	dest.RawQuery = ""

	request, err := http.NewRequest("GET", dest.String(), nil)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}
	request.Header.Add("Accept", contentType)

	response, err := client.Do(request)
	if err != nil {
		b.Fatalf("unexpected error: %v", err)
	}
	defer func(b *testing.B) {
		require.NoError(b, response.Body.Close())
	}(b)
	if response.StatusCode != http.StatusOK {
		b.Fatalf("Unexpected response %#v", response)
	}

	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
	if err != nil {
		b.Fatalf("waiting for watcher: %v", err)
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer func(b *testing.B) {
			require.NoError(b, response.Body.Close())
		}(b)
		if _, err := io.Copy(ioutil.Discard, response.Body); err != nil {
			b.Error(err)
		}
		wg.Done()
	}()

	actions := []watch.EventType{watch.Added, watch.Modified, watch.Deleted}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storageWatcher.Action(actions[i%len(actions)], items[i%len(items)])
	}
	// Storage closes the watcher when done sending events
	storageWatcher.Close()
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

	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	storageWatcher, err := simpleStorage.WaitForWatcher(ctx)
	if err != nil {
		b.Fatalf("waiting for watcher: %v", err)
	}

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer ws.Close()
		if _, err := io.Copy(ioutil.Discard, ws); err != nil {
			b.Error(err)
		}
		wg.Done()
	}()

	actions := []watch.EventType{watch.Added, watch.Modified, watch.Deleted}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		storageWatcher.Action(actions[i%len(actions)], &items[i%len(items)])
	}
	// Storage closes the watcher when done sending events
	storageWatcher.Close()
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
