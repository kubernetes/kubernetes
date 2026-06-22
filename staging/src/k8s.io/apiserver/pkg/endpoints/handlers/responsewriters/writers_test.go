/*
Copyright 2016 The Kubernetes Authors.

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

package responsewriters

import (
	"bytes"
	"compress/gzip"
	"context"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	jsonserializer "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

const benchmarkSeed = 100

//go:embed testdata/exemplar_pod.yaml
var benchmarkExemplarPodYAML []byte

func TestSerializeObjectParallel(t *testing.T) {
	largePayload := bytes.Repeat([]byte("0123456789abcdef"), defaultGzipThresholdBytes/16+1)
	type test struct {
		name string

		mediaType  string
		out        []byte
		outErrs    []error
		req        *http.Request
		statusCode int
		object     runtime.Object

		wantCode    int
		wantHeaders http.Header
	}
	newTest := func() test {
		return test{
			name:      "compress on gzip",
			out:       largePayload,
			mediaType: "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"gzip"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type":     []string{"application/json"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
		}
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, true)
	for i := 0; i < 100; i++ {
		ctt := newTest()
		t.Run(ctt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("recovered from err %v", r)
				}
			}()
			t.Parallel()

			encoder := &fakeEncoder{
				buf:  ctt.out,
				errs: ctt.outErrs,
			}
			if ctt.statusCode == 0 {
				ctt.statusCode = http.StatusOK
			}
			recorder := &fakeResponseRecorder{
				ResponseRecorder:   httptest.NewRecorder(),
				fe:                 encoder,
				errorAfterEncoding: true,
			}
			SerializeObject(ctt.mediaType, encoder, recorder, ctt.req, ctt.statusCode, ctt.object)
			result := recorder.Result()
			if result.StatusCode != ctt.wantCode {
				t.Fatalf("unexpected code: %v", result.StatusCode)
			}
			if !reflect.DeepEqual(result.Header, ctt.wantHeaders) {
				t.Fatal(cmp.Diff(ctt.wantHeaders, result.Header))
			}
		})
	}
}

func TestSerializeObject(t *testing.T) {
	smallPayload := []byte("{test-object,test-object}")
	largePayload := bytes.Repeat([]byte("0123456789abcdef"), defaultGzipThresholdBytes/16+1)
	tests := []struct {
		name string

		compressionEnabled bool

		mediaType  string
		out        []byte
		outErrs    []error
		req        *http.Request
		statusCode int
		object     runtime.Object

		wantCode    int
		wantHeaders http.Header
		wantBody    []byte
	}{
		{
			name:        "serialize object",
			out:         smallPayload,
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{""}},
			wantBody:    smallPayload,
		},

		{
			name:        "return content type",
			out:         smallPayload,
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "return status code",
			statusCode:  http.StatusBadRequest,
			out:         smallPayload,
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			wantCode:    http.StatusBadRequest,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "fail to encode object",
			out:         smallPayload,
			outErrs:     []error{fmt.Errorf("bad")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			wantCode:    http.StatusInternalServerError,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "fail to encode object or status",
			out:         smallPayload,
			outErrs:     []error{fmt.Errorf("bad"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			wantCode:    http.StatusInternalServerError,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte(": bad"),
		},

		{
			name:        "fail to encode object or status with status code",
			out:         smallPayload,
			outErrs:     []error{kerrors.NewNotFound(schema.GroupResource{}, "test"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			statusCode:  http.StatusOK,
			wantCode:    http.StatusNotFound,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte("NotFound:  \"test\" not found"),
		},

		{
			name:        "fail to encode object or status with status code and keeps previous error",
			out:         smallPayload,
			outErrs:     []error{kerrors.NewNotFound(schema.GroupResource{}, "test"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}, URL: &url.URL{Path: "/path"}},
			statusCode:  http.StatusNotAcceptable,
			wantCode:    http.StatusNotAcceptable,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte("NotFound:  \"test\" not found"),
		},

		{
			name:      "compression requires feature gate",
			out:       largePayload,
			mediaType: "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"gzip"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    largePayload,
		},

		{
			name:               "compress on gzip",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"gzip"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type":     []string{"application/json"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent(largePayload, gzipContentEncodingLevel),
		},

		{
			name:               "compression is not performed on small objects",
			compressionEnabled: true,
			out:                smallPayload,
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"gzip"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type": []string{"application/json"},
			},
			wantBody: smallPayload,
		},

		{
			name:               "compress when multiple encodings are requested",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"deflate, , gzip,"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type":     []string{"application/json"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent(largePayload, gzipContentEncodingLevel),
		},

		{
			name:               "ignore compression on deflate",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"deflate"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type": []string{"application/json"},
			},
			wantBody: largePayload,
		},

		{
			name:               "ignore compression on unrecognized types",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{", ,  other, nothing, what, "},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type": []string{"application/json"},
			},
			wantBody: largePayload,
		},

		{
			name:               "errors are compressed",
			compressionEnabled: true,
			statusCode:         http.StatusInternalServerError,
			out:                smallPayload,
			outErrs:            []error{errors.New(string(largePayload)), errors.New("bad2")},
			mediaType:          "application/json",
			req: &http.Request{
				Header: http.Header{
					"Accept-Encoding": []string{"gzip"},
				},
				URL: &url.URL{Path: "/path"},
			},
			wantCode: http.StatusInternalServerError,
			wantHeaders: http.Header{
				"Content-Type":     []string{"text/plain"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent([]byte(": "+string(largePayload)), gzipContentEncodingLevel),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, tt.compressionEnabled)

			encoder := &fakeEncoder{
				buf:  tt.out,
				errs: tt.outErrs,
			}
			if tt.statusCode == 0 {
				tt.statusCode = http.StatusOK
			}
			recorder := httptest.NewRecorder()
			SerializeObject(tt.mediaType, encoder, recorder, tt.req, tt.statusCode, tt.object)
			result := recorder.Result()
			if result.StatusCode != tt.wantCode {
				t.Fatalf("unexpected code: %v", result.StatusCode)
			}
			if !reflect.DeepEqual(result.Header, tt.wantHeaders) {
				t.Fatal(cmp.Diff(tt.wantHeaders, result.Header))
			}
			body, _ := io.ReadAll(result.Body)
			if !bytes.Equal(tt.wantBody, body) {
				t.Fatalf("wanted:\n%s\ngot:\n%s", hex.Dump(tt.wantBody), hex.Dump(body))
			}
		})
	}
}

func TestDeferredResponseWriter_Write(t *testing.T) {
	smallChunk := bytes.Repeat([]byte("b"), defaultGzipThresholdBytes-1)
	largeChunk := bytes.Repeat([]byte("b"), defaultGzipThresholdBytes+1)

	tests := []struct {
		name          string
		chunks        [][]byte
		expectGzip    bool
		expectHeaders http.Header
	}{
		{
			name:          "no writes",
			chunks:        nil,
			expectGzip:    false,
			expectHeaders: http.Header{},
		},
		{
			name:       "one empty write",
			chunks:     [][]byte{{}},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
		{
			name:       "one single byte write",
			chunks:     [][]byte{{'{'}},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
		{
			name:       "one small chunk write",
			chunks:     [][]byte{smallChunk},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
		{
			name:       "two small chunk writes",
			chunks:     [][]byte{smallChunk, smallChunk},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
		{
			name:       "one single byte and one small chunk write",
			chunks:     [][]byte{{'{'}, smallChunk},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
		{
			name:       "two single bytes and one small chunk write",
			chunks:     [][]byte{{'{'}, {'{'}, smallChunk},
			expectGzip: true,
			expectHeaders: http.Header{
				"Content-Type":     []string{"text/plain"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
		},
		{
			name:       "one large chunk writes",
			chunks:     [][]byte{largeChunk},
			expectGzip: true,
			expectHeaders: http.Header{
				"Content-Type":     []string{"text/plain"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
		},
		{
			name:       "two large chunk writes",
			chunks:     [][]byte{largeChunk, largeChunk},
			expectGzip: true,
			expectHeaders: http.Header{
				"Content-Type":     []string{"text/plain"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
		},
		{
			name:       "one small chunk and one large chunk write",
			chunks:     [][]byte{smallChunk, largeChunk},
			expectGzip: false,
			expectHeaders: http.Header{
				"Content-Type": []string{"text/plain"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockResponseWriter := httptest.NewRecorder()

			drw := &deferredResponseWriter{
				mediaType:       "text/plain",
				statusCode:      200,
				contentEncoding: "gzip",
				hw:              mockResponseWriter,
				ctx:             context.Background(),
			}

			fullPayload := []byte{}

			for _, chunk := range tt.chunks {
				n, err := drw.Write(chunk)

				if err != nil {
					t.Fatalf("unexpected error while writing chunk: %v", err)
				}
				if n != len(chunk) {
					t.Errorf("write is not complete, expected: %d bytes, written: %d bytes", len(chunk), n)
				}

				fullPayload = append(fullPayload, chunk...)
			}

			err := drw.Close()
			if err != nil {
				t.Fatalf("unexpected error when closing deferredResponseWriter: %v", err)
			}

			res := mockResponseWriter.Result()

			if res.StatusCode != http.StatusOK {
				t.Fatalf("status code is not writtend properly, expected: 200, got: %d", res.StatusCode)
			}
			if !reflect.DeepEqual(res.Header, tt.expectHeaders) {
				t.Fatal(cmp.Diff(tt.expectHeaders, res.Header))
			}

			resBytes, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("unexpected error occurred while reading response body: %v", err)
			}

			if tt.expectGzip {
				gr, err := gzip.NewReader(bytes.NewReader(resBytes))
				if err != nil {
					t.Fatalf("failed to create gzip reader: %v", err)
				}

				decompressed, err := io.ReadAll(gr)
				if err != nil {
					t.Fatalf("failed to decompress: %v", err)
				}

				if !bytes.Equal(fullPayload, decompressed) {
					t.Errorf("payload mismatch, expected: %s, got: %s", fullPayload, decompressed)
				}
			} else {
				if !bytes.Equal(fullPayload, resBytes) {
					t.Errorf("payload mismatch, expected: %s, got: %s", fullPayload, resBytes)
				}
			}
		})
	}
}

func benchmarkChunkingGzip(b *testing.B, count int, chunk []byte) {
	mockResponseWriter := httptest.NewRecorder()
	mockResponseWriter.Body = nil

	drw := &deferredResponseWriter{
		mediaType:       "text/plain",
		statusCode:      200,
		contentEncoding: "gzip",
		hw:              mockResponseWriter,
		ctx:             context.Background(),
	}
	b.ResetTimer()
	for i := 0; i < count; i++ {
		n, err := drw.Write(chunk)
		if err != nil {
			b.Fatalf("unexpected error while writing chunk: %v", err)
		}
		if n != len(chunk) {
			b.Errorf("write is not complete, expected: %d bytes, written: %d bytes", len(chunk), n)
		}
	}
	err := drw.Close()
	if err != nil {
		b.Fatalf("unexpected error when closing deferredResponseWriter: %v", err)
	}
	res := mockResponseWriter.Result()
	if res.StatusCode != http.StatusOK {
		b.Fatalf("status code is not writtend properly, expected: 200, got: %d", res.StatusCode)
	}
}

func BenchmarkChunkingGzip(b *testing.B) {
	tests := []struct {
		count int
		size  int
	}{
		{
			count: 100,
			size:  1_000,
		},
		{
			count: 100,
			size:  100_000,
		},
		{
			count: 1_000,
			size:  100_000,
		},
		{
			count: 1_000,
			size:  1_000_000,
		},
		{
			count: 10_000,
			size:  100_000,
		},
		{
			count: 100_000,
			size:  10_000,
		},
		{
			count: 1,
			size:  100_000,
		},
		{
			count: 1,
			size:  1_000_000,
		},
		{
			count: 1,
			size:  10_000_000,
		},
		{
			count: 1,
			size:  100_000_000,
		},
		{
			count: 1,
			size:  1_000_000_000,
		},
	}

	for _, t := range tests {
		b.Run(fmt.Sprintf("Count=%d/Size=%d", t.count, t.size), func(b *testing.B) {
			chunk := []byte(rand.String(t.size))
			benchmarkChunkingGzip(b, t.count, chunk)
		})
	}
}

func toProtoBuf(b *testing.B, list *v1.PodList) []byte {
	out, err := list.Marshal()
	if err != nil {
		b.Fatalf("Failed to marshal list to protobuf: %v", err)
	}
	return out
}

func toJSON(b *testing.B, list *v1.PodList) []byte {
	out, err := json.Marshal(list)
	if err != nil {
		b.Fatalf("Failed to marshal list to json: %v", err)
	}
	return out
}

func benchmarkSerializeObject(b *testing.B, payload []byte, gzip bool) {
	req := &http.Request{
		URL: &url.URL{Path: "/path"},
	}
	if gzip {
		req.Header = http.Header{
			"Accept-Encoding": []string{"gzip"},
		}
	}

	featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.APIResponseCompression, true)

	encoder := &fakeEncoder{
		buf: payload,
	}

	b.ResetTimer()
	responseBytesTotal := 0
	for b.Loop() {
		recorder := httptest.NewRecorder()
		SerializeObject("application/json", encoder, recorder, req, http.StatusOK, nil /* object */)
		result := recorder.Result()
		if result.StatusCode != http.StatusOK {
			b.Fatalf("incorrect status code: got %v;  want: %v", result.StatusCode, http.StatusOK)
		}
		responseBytesTotal += recorder.Body.Len()
	}
	b.ReportMetric(float64(responseBytesTotal/b.N), "writtenBytes/op")
}

func BenchmarkSerializeObject(b *testing.B) {
	for _, count := range []int{1_000, 10_000, 100_000} {
		b.Run(fmt.Sprintf("Count=%d", count), func(b *testing.B) {
			medias := []struct {
				name    string
				convert func(*testing.B, *v1.PodList) []byte
			}{
				{"Json", toJSON},
				{"Protobuf", toProtoBuf},
			}
			podList := benchmarkItems(b, count)
			for _, media := range medias {
				b.Run(fmt.Sprintf("MediaType=%s", media.name), func(b *testing.B) {
					payload := media.convert(b, podList)
					for _, gzip := range []bool{true, false} {
						b.Run(fmt.Sprintf("Compression=%v", gzip), func(b *testing.B) {
							benchmarkSerializeObject(b, payload, gzip)
						})
					}
				})
			}
		})
	}
}

func benchmarkProtobufEncodeWithAllocator(b *testing.B, encoder runtime.EncoderWithAllocator, object runtime.Object, pooled bool) {
	simpleAllocator := &runtime.SimpleAllocator{}
	discard := io.Discard

	b.StopTimer()
	if pooled {
		allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
		if err := encoder.EncodeWithAllocator(object, discard, allocator); err != nil {
			runtime.AllocatorPool.Put(allocator)
			b.Fatalf("warm pooled allocator: %v", err)
		}
		runtime.AllocatorPool.Put(allocator)
	} else {
		if err := encoder.EncodeWithAllocator(object, discard, simpleAllocator); err != nil {
			b.Fatalf("warm simple allocator: %v", err)
		}
	}
	b.StartTimer()

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		if pooled {
			allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
			err = encoder.EncodeWithAllocator(object, discard, allocator)
			runtime.AllocatorPool.Put(allocator)
		} else {
			err = encoder.EncodeWithAllocator(object, discard, simpleAllocator)
		}
		if err != nil {
			b.Fatalf("unexpected encode error: %v", err)
		}
	}
}

func benchmarkProtobufEncodeWithAllocatorParallel(b *testing.B, encoder runtime.EncoderWithAllocator, object runtime.Object, pooled bool) {
	discard := io.Discard

	b.StopTimer()
	if pooled {
		allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
		if err := encoder.EncodeWithAllocator(object, discard, allocator); err != nil {
			runtime.AllocatorPool.Put(allocator)
			b.Fatalf("warm pooled allocator: %v", err)
		}
		runtime.AllocatorPool.Put(allocator)
	} else {
		simpleAllocator := &runtime.SimpleAllocator{}
		if err := encoder.EncodeWithAllocator(object, discard, simpleAllocator); err != nil {
			b.Fatalf("warm simple allocator: %v", err)
		}
	}
	b.StartTimer()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		simpleAllocator := &runtime.SimpleAllocator{}
		for pb.Next() {
			var err error
			if pooled {
				allocator := runtime.AllocatorPool.Get().(*runtime.Allocator)
				err = encoder.EncodeWithAllocator(object, discard, allocator)
				runtime.AllocatorPool.Put(allocator)
			} else {
				err = encoder.EncodeWithAllocator(object, discard, simpleAllocator)
			}
			if err != nil {
				b.Fatalf("unexpected encode error: %v", err)
			}
		}
	})
}

func benchmarkItems(b *testing.B, n int) *v1.PodList {
	b.Helper()
	pod := v1.Pod{}
	if len(benchmarkExemplarPodYAML) == 0 {
		b.Fatal("benchmark exemplar pod is empty")
	}
	if err := yaml.Unmarshal(benchmarkExemplarPodYAML, &pod); err != nil {
		b.Fatalf("decode benchmark exemplar pod: %v", err)
	}

	rand.Seed(benchmarkSeed)
	nodeNames := make([]string, 25)
	for i := range nodeNames {
		nodeNames[i] = rand.String(10)
	}
	list := &v1.PodList{
		Items: make([]v1.Pod, n),
	}
	for i := range n {
		list.Items[i] = *pod.DeepCopy()
		list.Items[i].Namespace = rand.String(10)
		list.Items[i].Name = list.Items[i].GenerateName + rand.String(10)
		list.Items[i].UID = types.UID(rand.String(36))
		list.Items[i].ResourceVersion = ""
		list.Items[i].Spec.NodeName = nodeNames[rand.Intn(len(nodeNames))]
	}
	return list
}

func BenchmarkStreamingProtobufPodListAllocatorReuse(b *testing.B) {
	counts := []int{50, 500, 5_000}
	modes := []struct {
		name     string
		parallel bool
	}{
		{name: "Mode=Single", parallel: false},
		{name: "Mode=Parallel", parallel: true},
	}
	allocators := []struct {
		name   string
		pooled bool
	}{
		{name: "Allocator=Simple", pooled: false},
		{name: "Allocator=Pooled", pooled: true},
	}

	for _, mode := range modes {
		b.Run(mode.name, func(b *testing.B) {
			for _, count := range counts {
				b.Run(fmt.Sprintf("Pods=%d", count), func(b *testing.B) {
					podList := benchmarkItems(b, count)
					serializer := protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{
						StreamingCollectionsEncoding: true,
					})
					for _, allocator := range allocators {
						b.Run(allocator.name, func(b *testing.B) {
							if mode.parallel {
								benchmarkProtobufEncodeWithAllocatorParallel(b, serializer, podList, allocator.pooled)
								return
							}
							benchmarkProtobufEncodeWithAllocator(b, serializer, podList, allocator.pooled)
						})
					}
				})
			}
		})
	}
}

type fakeResponseRecorder struct {
	*httptest.ResponseRecorder
	fe                 *fakeEncoder
	errorAfterEncoding bool
}

func (frw *fakeResponseRecorder) Write(buf []byte) (int, error) {
	if frw.errorAfterEncoding && frw.fe.encodeCalled {
		return 0, errors.New("returning a requested error")
	}
	return frw.ResponseRecorder.Write(buf)
}

type fakeEncoder struct {
	obj  runtime.Object
	buf  []byte
	errs []error

	encodeCalled bool
}

func (e *fakeEncoder) Encode(obj runtime.Object, w io.Writer) error {
	e.obj = obj
	if len(e.errs) > 0 {
		err := e.errs[0]
		e.errs = e.errs[1:]
		return err
	}
	_, err := w.Write(e.buf)
	e.encodeCalled = true
	return err
}

func (e *fakeEncoder) Identifier() runtime.Identifier {
	return runtime.Identifier("fake")
}

func gzipContent(data []byte, level int) []byte {
	buf := &bytes.Buffer{}
	gw, err := gzip.NewWriterLevel(buf, level)
	if err != nil {
		panic(err)
	}
	if _, err := gw.Write(data); err != nil {
		panic(err)
	}
	if err := gw.Close(); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func TestStreamingGzipIntegration(t *testing.T) {
	largeChunk := bytes.Repeat([]byte("b"), defaultGzipThresholdBytes+1)
	tcs := []struct {
		name            string
		serializer      runtime.Encoder
		object          runtime.Object
		expectGzip      bool
		expectStreaming bool
	}{
		{
			name:            "JSON, small object, default -> no gzip",
			serializer:      jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, nil, nil, jsonserializer.SerializerOptions{}),
			object:          &testapigroupv1.CarpList{},
			expectGzip:      false,
			expectStreaming: false,
		},
		{
			name:            "JSON, small object, streaming -> no gzip",
			serializer:      jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, nil, nil, jsonserializer.SerializerOptions{StreamingCollectionsEncoding: true}),
			object:          &testapigroupv1.CarpList{},
			expectGzip:      false,
			expectStreaming: true,
		},
		{
			name:            "JSON, large object, default -> gzip",
			serializer:      jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, nil, nil, jsonserializer.SerializerOptions{}),
			object:          &testapigroupv1.CarpList{TypeMeta: metav1.TypeMeta{Kind: string(largeChunk)}},
			expectGzip:      true,
			expectStreaming: false,
		},
		{
			name:            "JSON, large object, streaming -> gzip",
			serializer:      jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, nil, nil, jsonserializer.SerializerOptions{StreamingCollectionsEncoding: true}),
			object:          &testapigroupv1.CarpList{TypeMeta: metav1.TypeMeta{Kind: string(largeChunk)}},
			expectGzip:      true,
			expectStreaming: true,
		},
		{
			name:            "Protobuf, small object, default -> no gzip",
			serializer:      protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{}),
			object:          &testapigroupv1.CarpList{},
			expectGzip:      false,
			expectStreaming: false,
		},
		{
			name:            "Protobuf, small object, streaming -> no gzip",
			serializer:      protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{StreamingCollectionsEncoding: true}),
			object:          &testapigroupv1.CarpList{},
			expectGzip:      false,
			expectStreaming: true,
		},
		{
			name:            "Protobuf, large object, default -> gzip",
			serializer:      protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{}),
			object:          &testapigroupv1.CarpList{TypeMeta: metav1.TypeMeta{Kind: string(largeChunk)}},
			expectGzip:      true,
			expectStreaming: false,
		},
		{
			name:            "Protobuf, large object, streaming -> gzip",
			serializer:      protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{StreamingCollectionsEncoding: true}),
			object:          &testapigroupv1.CarpList{TypeMeta: metav1.TypeMeta{Kind: string(largeChunk)}},
			expectGzip:      true,
			expectStreaming: true,
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			mockResponseWriter := httptest.NewRecorder()
			drw := &deferredResponseWriter{
				mediaType:       "text/plain",
				statusCode:      200,
				contentEncoding: "gzip",
				hw:              mockResponseWriter,
				ctx:             context.Background(),
			}
			counter := &writeCounter{Writer: drw}
			err := tc.serializer.Encode(tc.object, counter)
			if err != nil {
				t.Fatal(err)
			}
			encoding := mockResponseWriter.Header().Get("Content-Encoding")
			if (encoding == "gzip") != tc.expectGzip {
				t.Errorf("Expect gzip: %v, got: %q", tc.expectGzip, encoding)
			}
			if counter.writeCount < 1 {
				t.Fatalf("Expect at least 1 write")
			}
			if (counter.writeCount > 1) != tc.expectStreaming {
				t.Errorf("Expect streaming: %v, got write count: %d", tc.expectStreaming, counter.writeCount)
			}
		})
	}
}

type writeCounter struct {
	writeCount int
	io.Writer
}

func (b *writeCounter) Write(data []byte) (int, error) {
	b.writeCount++
	return b.Writer.Write(data)
}

func BenchmarkJSONStreaming(b *testing.B) {
	scheme := runtime.NewScheme()
	if err := v1.AddToScheme(scheme); err != nil {
		b.Fatalf("failed to add v1 to scheme: %v", err)
	}
	typer := scheme
	creater := scheme
	serializerStreaming := jsonserializer.NewSerializerWithOptions(
		jsonserializer.DefaultMetaFactory,
		creater,
		typer,
		jsonserializer.SerializerOptions{StreamingCollectionsEncoding: true},
	)
	serializerNonStreaming := jsonserializer.NewSerializerWithOptions(
		jsonserializer.DefaultMetaFactory,
		creater,
		typer,
		jsonserializer.SerializerOptions{StreamingCollectionsEncoding: false},
	)
	podList := benchmarkItems(b, 1000)
	b.Run("Streaming", func(b *testing.B) {
		var buf bytes.Buffer
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf.Reset()
			err := serializerStreaming.Encode(podList, &buf)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("NonStreaming", func(b *testing.B) {
		var buf bytes.Buffer
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf.Reset()
			err := serializerNonStreaming.Encode(podList, &buf)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
