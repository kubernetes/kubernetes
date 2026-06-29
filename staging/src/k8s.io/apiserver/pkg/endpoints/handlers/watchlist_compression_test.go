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
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	endpointstesting "k8s.io/apiserver/pkg/endpoints/testing"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const gzipContentOrDieEncodingLevel = 1

func TestWatchResponseWriter(t *testing.T) {
	scenarios := []struct {
		name                    string
		contentEncoding         string
		actualData              []byte
		expectedContentEncoding string
		expectedVary            string
		expectedBody            []byte
	}{
		{
			name:                    "without compression",
			contentEncoding:         "",
			actualData:              []byte("hello world"),
			expectedContentEncoding: "",
			expectedVary:            "",
			expectedBody:            []byte("hello world"),
		},
		{
			name:                    "with gzip compression",
			contentEncoding:         "gzip",
			actualData:              []byte("hello world"),
			expectedContentEncoding: "gzip",
			expectedVary:            "Accept-Encoding",
			expectedBody:            gzipContentOrDie([]byte("hello world"), gzipContentOrDieEncodingLevel, true),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			rw := newWatchResponseWriter(recorder, recorder, scenario.contentEncoding, true)
			rw.BeginStream("application/json")

			// verify headers
			if got := recorder.Header().Get("Content-Type"); got != "application/json" {
				t.Errorf("Content-Type = %q, want %q", got, "application/json")
			}
			if got := recorder.Header().Get("Transfer-Encoding"); got != "chunked" {
				t.Errorf("Transfer-Encoding = %q, want %q", got, "chunked")
			}
			if got := recorder.Header().Get("Content-Encoding"); got != scenario.expectedContentEncoding {
				t.Errorf("Content-Encoding = %q, want %q", got, scenario.expectedContentEncoding)
			}
			if got := recorder.Header().Get("Vary"); got != scenario.expectedVary {
				t.Errorf("Vary = %q, want %q", got, scenario.expectedVary)
			}
			if got := recorder.Code; got != http.StatusOK {
				t.Errorf("status code = %d, want %d", got, http.StatusOK)
			}

			// write and verify body
			n, err := rw.Write(scenario.actualData)
			if err != nil {
				t.Fatalf("Write failed: %v", err)
			}
			if n != len(scenario.actualData) {
				t.Errorf("Write returned %d, want %d", n, len(scenario.actualData))
			}

			// flush and verify
			if err := rw.Flush(); err != nil {
				t.Fatalf("Flush failed: %v", err)
			}
			if !recorder.Flushed {
				t.Error("expected recorder to be flushed")
			}

			// close and verify body
			if err := rw.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}
			if !bytes.Equal(recorder.Body.Bytes(), scenario.expectedBody) {
				t.Errorf("body = %v, want %v", recorder.Body.Bytes(), scenario.expectedBody)
			}
		})
	}
}

func TestWatchResponseWriterDoubleClose(t *testing.T) {
	scenarios := []struct {
		name      string
		newWriter func(*testing.T, *httptest.ResponseRecorder) watchStreamWriter
	}{
		{
			name: "perFlushGzipWriter",
			newWriter: func(t *testing.T, r *httptest.ResponseRecorder) watchStreamWriter {
				return &perFlushGzipWriter{delegateRW: r, flusher: r}
			},
		},
		{
			name: "watchResponseWriter",
			newWriter: func(t *testing.T, r *httptest.ResponseRecorder) watchStreamWriter {
				rw := newWatchResponseWriter(r, r, "gzip", true)
				rw.BeginStream("application/json")
				return rw
			},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			w := scenario.newWriter(t, httptest.NewRecorder())
			_, err := w.Write([]byte("data"))
			require.NoError(t, err)
			require.NoError(t, w.Close())
			require.NoError(t, w.Close())
		})
	}
}

func gzipContentOrDie(data []byte, level int, flush bool) []byte {
	buf := &bytes.Buffer{}
	gw, err := gzip.NewWriterLevel(buf, level)
	if err != nil {
		panic(err)
	}
	if _, err := gw.Write(data); err != nil {
		panic(err)
	}
	if flush {
		if err := gw.Flush(); err != nil {
			panic(err)
		}
	}
	if err := gw.Close(); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func TestGzipNewReaderFailsOnUncompressedContent(t *testing.T) {
	_, err := gzip.NewReader(bytes.NewReader([]byte(`{"type":"ADDED","object":{}}`)))
	require.Error(t, err)
}

func TestPerFlushGzipWriter(t *testing.T) {
	scenarios := []struct {
		name          string
		writes        []string
		flushPerWrite bool
		expected      string
	}{
		{
			name:          "single event with flush",
			writes:        []string{"hello"},
			flushPerWrite: true,
			expected:      "hello",
		},
		{
			name:          "flush after every write produces concatenated gzip members",
			writes:        []string{"first", "second", "third"},
			flushPerWrite: true,
			expected:      "firstsecondthird",
		},
		{
			name:          "flush only at the end produces a single gzip member",
			writes:        []string{"first", "second", "third"},
			flushPerWrite: false,
			expected:      "firstsecondthird",
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			w := &perFlushGzipWriter{delegateRW: recorder, flusher: recorder}
			for _, msg := range scenario.writes {
				_, err := w.Write([]byte(msg))
				require.NoError(t, err)
				if scenario.flushPerWrite {
					require.NoError(t, w.Flush())
				}
			}
			if !scenario.flushPerWrite {
				require.NoError(t, w.Flush())
			}
			require.NoError(t, w.Close())
			gr, err := gzip.NewReader(recorder.Body)
			require.NoError(t, err)
			got, err := io.ReadAll(gr)
			require.NoError(t, err)
			require.Equal(t, scenario.expected, string(got))
		})
	}
}

func TestWatchServerCompression(t *testing.T) {
	simpleEventFn := func(name string) watch.Event {
		return watch.Event{
			Type: watch.Added,
			Object: &endpointstesting.Simple{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				TypeMeta:   metav1.TypeMeta{APIVersion: testGroupV2.String()},
			},
		}
	}
	bookmarkEventFn := func() watch.Event {
		return watch.Event{
			Type: watch.Bookmark,
			Object: &endpointstesting.Simple{
				ObjectMeta: metav1.ObjectMeta{
					ResourceVersion: "100",
					Annotations:     map[string]string{metav1.InitialEventsAnnotationKey: "true"},
				},
				TypeMeta: metav1.TypeMeta{APIVersion: testGroupV2.String()},
			},
		}
	}

	scenarios := []struct {
		name               string
		featureGateEnabled bool
		isWatchListRequest bool
		acceptEncoding     string
		expectGzip         bool
		events             []watch.Event
	}{
		{
			name:               "watchlist request is compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "gzip",
			expectGzip:         true,
			events:             []watch.Event{simpleEventFn("obj-1")},
		},
		{
			name:               "events after initial-events-end bookmark are readable",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "gzip",
			expectGzip:         true,
			events: []watch.Event{
				simpleEventFn("pre-bookmark"),
				bookmarkEventFn(),
				simpleEventFn("post-bookmark"),
			},
		},
		{
			name:               "regular watch request is not compressed",
			featureGateEnabled: true,
			isWatchListRequest: false,
			acceptEncoding:     "gzip",
			expectGzip:         false,
			events:             []watch.Event{simpleEventFn("obj-1")},
		},
		{
			name:               "watchlist request without accept-encoding is not compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "",
			expectGzip:         false,
			events:             []watch.Event{simpleEventFn("obj-1")},
		},
		{
			name:               "watchlist request with feature gate disabled is not compressed",
			featureGateEnabled: false,
			isWatchListRequest: true,
			acceptEncoding:     "gzip",
			expectGzip:         false,
			events:             []watch.Event{simpleEventFn("obj-1")},
		},
		{
			name:               "watchlist request with multi value accept-encoding is compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "deflate, gzip",
			expectGzip:         true,
			events:             []watch.Event{simpleEventFn("obj-1")},
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchListCompression, scenario.featureGateEnabled)

			watcher := watch.NewFakeWithOptions(watch.FakeOptions{ChannelSize: len(scenario.events)})

			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
			require.True(t, ok)
			require.NotNil(t, info.StreamSerializer)

			watchServer := &WatchServer{
				Scope:              &RequestScope{},
				Watching:           watcher,
				MediaType:          "application/json",
				Framer:             info.StreamSerializer.Framer,
				Encoder:            testCodecV2,
				EmbeddedEncoder:    testCodecV2,
				TimeoutFactory:     &fakeTimeoutFactory{done: make(chan struct{})},
				isWatchListRequest: scenario.isWatchListRequest,
			}

			for _, event := range scenario.events {
				watcher.Action(event.Type, event.Object)
			}

			s := httptest.NewServer(serveWatch(watcher, watchServer, nil))
			defer s.Close()

			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, s.URL, nil)
			require.NoError(t, err)
			if scenario.acceptEncoding != "" {
				req.Header.Set("Accept-Encoding", scenario.acceptEncoding)
			}
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer func() {
				require.NoError(t, resp.Body.Close())
			}()

			var body io.Reader = resp.Body
			if scenario.expectGzip {
				require.Equal(t, "gzip", resp.Header.Get("Content-Encoding"))
				gr, err := gzip.NewReader(resp.Body)
				require.NoError(t, err)
				defer func() {
					require.NoError(t, gr.Close())
				}()
				body = gr
			} else {
				require.Empty(t, resp.Header.Get("Content-Encoding"))
			}

			decoder := json.NewDecoder(body)
			for i, event := range scenario.events {
				var got watchJSON
				require.NoError(t, decoder.Decode(&got), "decoding event %d", i)
				require.Equal(t, event.Type, got.Type, "event %d type", i)
				var obj endpointstesting.Simple
				require.NoError(t, json.Unmarshal(got.Object, &obj), "unmarshalling event %d", i)
				require.Equal(t, testGroupV2.String(), obj.APIVersion, "event %d", i)
				require.Equal(t, event.Object.(*endpointstesting.Simple).Name, obj.Name, "event %d", i)
			}
		})
	}
}
