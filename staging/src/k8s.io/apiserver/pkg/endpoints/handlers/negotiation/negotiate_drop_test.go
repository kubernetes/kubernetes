/*
Copyright 2026 The Kubernetes Authors.

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

package negotiation

import (
	"io"
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// taggedSerializer is a do-nothing Serializer with a distinguishing tag, so
// tests can identify which serializer the negotiation layer selected. Plain
// runtime.NewCodec(NoopEncoder{}, NoopDecoder{}) returns value-equal structs
// for every call, which would mask whether a swap actually happened.
type taggedSerializer struct{ tag string }

func (taggedSerializer) Encode(runtime.Object, io.Writer) error { return nil }
func (taggedSerializer) Decode(_ []byte, _ *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	return into, nil, nil
}
func (s taggedSerializer) Identifier() runtime.Identifier { return runtime.Identifier(s.tag) }

// dropNegotiater returns a single application/json SerializerInfo whose
// ExcludeManagedFieldsSerializer is distinct from Serializer, so tests can
// observe the negotiation swap by tag.
type dropNegotiater struct {
	full, stripped, stream runtime.Serializer
}

func (n *dropNegotiater) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{{
		MediaType:                      "application/json",
		MediaTypeType:                  "application",
		MediaTypeSubType:               "json",
		EncodesAsText:                  true,
		Serializer:                     n.full,
		ExcludeManagedFieldsSerializer: n.stripped,
		StreamSerializer: &runtime.StreamSerializerInfo{
			EncodesAsText: true,
			Serializer:    n.stream,
		},
	}}
}

func (n *dropNegotiater) EncoderForVersion(s runtime.Encoder, _ runtime.GroupVersioner) runtime.Encoder {
	return s
}

func (n *dropNegotiater) DecoderToVersion(s runtime.Decoder, _ runtime.GroupVersioner) runtime.Decoder {
	return s
}

// TestNegotiateDropManagedFields covers the Accept: ...;drop=metadata.managedFields
// negotiation path used by every response endpoint (GET, LIST, WATCH, PUT, POST).
// Because all of those route through NegotiateOutputMediaType (or its Stream
// twin for WATCH), exercising the two negotiation entry points is sufficient
// to demonstrate opt-out for the full HTTP verb set the KEP claims.
func TestNegotiateDropManagedFields(t *testing.T) {
	full := taggedSerializer{tag: "full"}
	stripped := taggedSerializer{tag: "stripped"}
	stream := taggedSerializer{tag: "stream"}
	ns := &dropNegotiater{full: full, stripped: stripped, stream: stream}

	cases := []struct {
		name        string
		accept      string
		gateOn      bool
		wantOptOut  bool   // ExcludeManagedFields flag set on MediaTypeOptions
		wantPicked  string // "full" | "stripped"
		wantStream  string // "full" | "stripped" — the embedded-object encoder for WATCH
	}{
		{
			name:       "no drop param",
			accept:     "application/json",
			gateOn:     true,
			wantOptOut: false,
			wantPicked: "full",
			wantStream: "full",
		},
		{
			name:       "drop=managedFields, gate enabled",
			accept:     "application/json;drop=metadata.managedFields",
			gateOn:     true,
			wantOptOut: true,
			wantPicked: "stripped",
			wantStream: "stripped",
		},
		{
			name:       "drop=managedFields, gate disabled — request silently ignored",
			accept:     "application/json;drop=metadata.managedFields",
			gateOn:     false,
			wantOptOut: false,
			wantPicked: "full",
			wantStream: "full",
		},
		{
			name:       "drop with unknown target — ignored, no error",
			accept:     "application/json;drop=spec.someFutureField",
			gateOn:     true,
			wantOptOut: false,
			wantPicked: "full",
			wantStream: "full",
		},
		{
			// goautoneg.ParseAccept uses ',' as the media-type-clause
			// separator, so a comma inside a parameter value terminates the
			// value. If multiple drop targets need to coexist in one Accept
			// header, the KEP must pick a non-',' separator (e.g. '+') or
			// use a different conveyance. See the KEP "Accept Parameter"
			// section — the current "comma-separated lists" wording is
			// incompatible with the standard Accept parser.
			name:       "drop=set with '+' separator — extensible parser",
			accept:     "application/json;drop=spec.someFutureField+metadata.managedFields",
			gateOn:     true,
			wantOptOut: true,
			wantPicked: "stripped",
			wantStream: "stripped",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManagedFieldsOptOut, tc.gateOn)

			req := &http.Request{Header: http.Header{"Accept": []string{tc.accept}}}

			// GET/LIST/PUT/POST response path.
			opts, info, err := NegotiateOutputMediaType(req, ns, DefaultEndpointRestrictions)
			if err != nil {
				t.Fatalf("NegotiateOutputMediaType: %v", err)
			}
			if opts.ExcludeManagedFields != tc.wantOptOut {
				t.Errorf("ExcludeManagedFields = %v, want %v", opts.ExcludeManagedFields, tc.wantOptOut)
			}
			if got := pickName(info.Serializer, full, stripped); got != tc.wantPicked {
				t.Errorf("response Serializer = %s, want %s", got, tc.wantPicked)
			}

			// WATCH path: object encoder swaps but the stream framer doesn't.
			streamInfo, err := NegotiateOutputMediaTypeStream(req, ns, DefaultEndpointRestrictions)
			if err != nil {
				t.Fatalf("NegotiateOutputMediaTypeStream: %v", err)
			}
			if got := pickName(streamInfo.Serializer, full, stripped); got != tc.wantStream {
				t.Errorf("watch object Serializer = %s, want %s", got, tc.wantStream)
			}
			if streamInfo.StreamSerializer == nil || streamInfo.StreamSerializer.Serializer != stream {
				t.Errorf("watch StreamSerializer was unexpectedly swapped — must remain the framer-bound encoder")
			}
		})
	}
}

func pickName(got, full, stripped runtime.Serializer) string {
	switch got.Identifier() {
	case full.Identifier():
		return "full"
	case stripped.Identifier():
		return "stripped"
	default:
		return string(got.Identifier())
	}
}
