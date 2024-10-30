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

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	metainternalscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/client-go/transport"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// SetTestOnlyCBORClientFeatureGatesForTest overrides the CBOR client feature gates in the test-only
// client feature gate instance for the duration of a test. The CBOR client feature gates are
// temporarily registered in their own feature gate instance that does not include runtime wiring to
// command-line flags or environment variables in order to mitigate the risk of enabling a new
// encoding before all integration tests have been demonstrated to pass.
//
// This will be removed as an alpha requirement. The client feature gates will be registered with
// the existing feature gate instance and tests will use
// k8s.io/client-go/features/testing.SetFeatureDuringTest (which unlike
// k8s.io/component-base/featuregate/testing.SetFeatureGateDuringTest does not accept a feature gate
// instance as a parameter).
func SetTestOnlyCBORClientFeatureGatesForTest(tb testing.TB, allowed, preferred bool) {
	originalAllowed := clientfeatures.TestOnlyFeatureGates.Enabled(clientfeatures.TestOnlyClientAllowsCBOR)
	tb.Cleanup(func() {
		if err := clientfeatures.TestOnlyFeatureGates.Set(clientfeatures.TestOnlyClientAllowsCBOR, originalAllowed); err != nil {
			tb.Fatal(err)
		}
	})
	if err := clientfeatures.TestOnlyFeatureGates.Set(clientfeatures.TestOnlyClientAllowsCBOR, allowed); err != nil {
		tb.Fatal(err)
	}

	originalPreferred := clientfeatures.TestOnlyFeatureGates.Enabled(clientfeatures.TestOnlyClientPrefersCBOR)
	tb.Cleanup(func() {
		if err := clientfeatures.TestOnlyFeatureGates.Set(clientfeatures.TestOnlyClientPrefersCBOR, originalPreferred); err != nil {
			tb.Fatal(err)
		}
	})
	if err := clientfeatures.TestOnlyFeatureGates.Set(clientfeatures.TestOnlyClientPrefersCBOR, preferred); err != nil {
		tb.Fatal(err)
	}
}

// EnableCBORForTest patches global state to enable the CBOR serializer and reverses those changes
// at the end of the test. As a risk mitigation, integration tests are initially written this way so
// that integration tests can be implemented fully and incrementally before exposing options
// (including feature gates) that can enable CBOR at runtime. After integration test coverage is
// complete, feature gates will be introduced to completely supersede this mechanism.
func EnableCBORServingAndStorageForTest(tb testing.TB) {
	featuregatetesting.SetFeatureGateDuringTest(tb, utilfeature.TestOnlyFeatureGate, features.TestOnlyCBORServingAndStorage, true)

	newCBORSerializerInfo := func(creater runtime.ObjectCreater, typer runtime.ObjectTyper) runtime.SerializerInfo {
		return runtime.SerializerInfo{
			MediaType:        "application/cbor",
			MediaTypeType:    "application",
			MediaTypeSubType: "cbor",
			Serializer:       cbor.NewSerializer(creater, typer),
			StrictSerializer: cbor.NewSerializer(creater, typer, cbor.Strict(true)),
			StreamSerializer: &runtime.StreamSerializerInfo{
				Framer:     cbor.NewFramer(),
				Serializer: cbor.NewSerializer(creater, typer, cbor.Transcode(false)),
			},
		}
	}

	// Codecs for built-in types are constructed at package initialization time and read by
	// value from REST storage providers.
	codecs := map[*runtime.Scheme]*serializer.CodecFactory{
		legacyscheme.Scheme:           &legacyscheme.Codecs,
		metainternalscheme.Scheme:     &metainternalscheme.Codecs,
		aggregatorscheme.Scheme:       &aggregatorscheme.Codecs,
		apiextensionsapiserver.Scheme: &apiextensionsapiserver.Codecs,
	}

	for scheme, factory := range codecs {
		original := *factory // shallow copy of original value
		tb.Cleanup(func() { *codecs[scheme] = original })
		*codecs[scheme] = serializer.NewCodecFactory(scheme, serializer.WithSerializer(newCBORSerializerInfo))
	}
}

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
