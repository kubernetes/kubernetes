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
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

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
			req:         &http.Request{Header: http.Header{}},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{""}},
			wantBody:    smallPayload,
		},

		{
			name:        "return content type",
			out:         smallPayload,
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "return status code",
			statusCode:  http.StatusBadRequest,
			out:         smallPayload,
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			wantCode:    http.StatusBadRequest,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "fail to encode object",
			out:         smallPayload,
			outErrs:     []error{fmt.Errorf("bad")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			wantCode:    http.StatusInternalServerError,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    smallPayload,
		},

		{
			name:        "fail to encode object or status",
			out:         smallPayload,
			outErrs:     []error{fmt.Errorf("bad"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			wantCode:    http.StatusInternalServerError,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte(": bad"),
		},

		{
			name:        "fail to encode object or status with status code",
			out:         smallPayload,
			outErrs:     []error{errors.NewNotFound(schema.GroupResource{}, "test"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			statusCode:  http.StatusOK,
			wantCode:    http.StatusNotFound,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte("NotFound:  \"test\" not found"),
		},

		{
			name:        "fail to encode object or status with status code and keeps previous error",
			out:         smallPayload,
			outErrs:     []error{errors.NewNotFound(schema.GroupResource{}, "test"), fmt.Errorf("bad2")},
			mediaType:   "application/json",
			req:         &http.Request{Header: http.Header{}},
			statusCode:  http.StatusNotAcceptable,
			wantCode:    http.StatusNotAcceptable,
			wantHeaders: http.Header{"Content-Type": []string{"text/plain"}},
			wantBody:    []byte("NotFound:  \"test\" not found"),
		},

		{
			name:      "compression requires feature gate",
			out:       largePayload,
			mediaType: "application/json",
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"gzip"},
			}},
			wantCode:    http.StatusOK,
			wantHeaders: http.Header{"Content-Type": []string{"application/json"}},
			wantBody:    largePayload,
		},

		{
			name:               "compress on gzip",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"gzip"},
			}},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type":     []string{"application/json"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent(largePayload, defaultGzipContentEncodingLevel),
		},

		{
			name:               "compression is not performed on small objects",
			compressionEnabled: true,
			out:                smallPayload,
			mediaType:          "application/json",
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"gzip"},
			}},
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
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"deflate, , gzip,"},
			}},
			wantCode: http.StatusOK,
			wantHeaders: http.Header{
				"Content-Type":     []string{"application/json"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent(largePayload, defaultGzipContentEncodingLevel),
		},

		{
			name:               "ignore compression on deflate",
			compressionEnabled: true,
			out:                largePayload,
			mediaType:          "application/json",
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"deflate"},
			}},
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
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{", ,  other, nothing, what, "},
			}},
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
			outErrs:            []error{fmt.Errorf(string(largePayload)), fmt.Errorf("bad2")},
			mediaType:          "application/json",
			req: &http.Request{Header: http.Header{
				"Accept-Encoding": []string{"gzip"},
			}},
			wantCode: http.StatusInternalServerError,
			wantHeaders: http.Header{
				"Content-Type":     []string{"text/plain"},
				"Content-Encoding": []string{"gzip"},
				"Vary":             []string{"Accept-Encoding"},
			},
			wantBody: gzipContent([]byte(": "+string(largePayload)), defaultGzipContentEncodingLevel),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIResponseCompression, tt.compressionEnabled)()

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
				t.Fatal(diff.ObjectReflectDiff(tt.wantHeaders, result.Header))
			}
			body, _ := ioutil.ReadAll(result.Body)
			if !bytes.Equal(tt.wantBody, body) {
				t.Fatalf("wanted:\n%s\ngot:\n%s", hex.Dump(tt.wantBody), hex.Dump(body))
			}
		})
	}
}

type fakeEncoder struct {
	obj  runtime.Object
	buf  []byte
	errs []error
}

func (e *fakeEncoder) Encode(obj runtime.Object, w io.Writer) error {
	e.obj = obj
	if len(e.errs) > 0 {
		err := e.errs[0]
		e.errs = e.errs[1:]
		return err
	}
	_, err := w.Write(e.buf)
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
