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
	recorder := httptest.NewRecorder()
	rw := newWatchResponseWriter(recorder, recorder, "gzip", true)
	rw.BeginStream("application/json")

	if _, err := rw.Write([]byte("data")); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	if err := rw.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}
	if err := rw.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
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

func TestWatchServerCompression(t *testing.T) {
	scenarios := []struct {
		name               string
		featureGateEnabled bool
		isWatchListRequest bool
		acceptEncoding     string
		expectGzip         bool
	}{
		{
			name:               "watchlist request is compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "gzip",
			expectGzip:         true,
		},
		{
			name:               "regular watch request is not compressed",
			featureGateEnabled: true,
			isWatchListRequest: false,
			acceptEncoding:     "gzip",
			expectGzip:         false,
		},
		{
			name:               "watchlist request without accept-encoding is not compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "",
			expectGzip:         false,
		},
		{
			name:               "watchlist request with feature gate disabled is not compressed",
			featureGateEnabled: false,
			isWatchListRequest: true,
			acceptEncoding:     "gzip",
			expectGzip:         false,
		},
		{
			name:               "watchlist request with multi value accept-encoding is compressed",
			featureGateEnabled: true,
			isWatchListRequest: true,
			acceptEncoding:     "deflate, gzip",
			expectGzip:         true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchListCompression, scenario.featureGateEnabled)

			watcher := watch.NewFakeWithOptions(watch.FakeOptions{ChannelSize: 1})

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

			watcher.Add(&endpointstesting.Simple{TypeMeta: metav1.TypeMeta{APIVersion: testGroupV2.String()}})

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

			var got watchJSON
			require.NoError(t, json.NewDecoder(body).Decode(&got))
			require.Equal(t, watch.Added, got.Type)
			var obj endpointstesting.Simple
			require.NoError(t, json.Unmarshal(got.Object, &obj))
			require.Equal(t, testGroupV2.String(), obj.APIVersion)
		})
	}
}
