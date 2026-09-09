/*
Copyright 2024 The Kubernetes Authors.

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

package framework

import (
	"bytes"
	"io"
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/transport"
)

// AssertRequestResponseAsCBOR returns a transport.WrapperFunc that will report a test error if a
// non-empty request or response body contains data that does not appear to be CBOR-encoded.
func AssertRequestResponseAsCBOR(t testing.TB) transport.WrapperFunc {
	unsupportedPatchContentTypes := sets.New(
		"application/json-patch+json",
		"application/merge-patch+json",
		"application/strategic-merge-patch+json",
	)

	return func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if request.Body != nil && !unsupportedPatchContentTypes.Has(request.Header.Get("Content-Type")) {
				requestbody, err := io.ReadAll(request.Body)
				if err != nil {
					t.Error(err)
				}
				if len(requestbody) > 0 {
					err = direct.Unmarshal(requestbody, new(interface{}))
					if err != nil {
						t.Errorf("non-cbor request: 0x%x", requestbody)
					}
				}
				request.Body = io.NopCloser(bytes.NewReader(requestbody))
			}

			response, rterr := rt.RoundTrip(request)
			if rterr != nil {
				return response, rterr
			}

			// We can't synchronously inspect streaming responses, so tee to a buffer
			// and inspect it at the end of the test.
			var buf bytes.Buffer
			response.Body = struct {
				io.Reader
				io.Closer
			}{
				Reader: io.TeeReader(response.Body, &buf),
				Closer: response.Body,
			}
			t.Cleanup(func() {
				if buf.Len() == 0 {
					return
				}
				if err := direct.Unmarshal(buf.Bytes(), new(interface{})); err != nil {
					t.Errorf("non-cbor response: 0x%x", buf.Bytes())
				}
			})

			return response, rterr
		})
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
