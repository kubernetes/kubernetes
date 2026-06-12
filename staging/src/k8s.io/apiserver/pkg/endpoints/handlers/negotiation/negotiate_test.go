/*
Copyright 2015 The Kubernetes Authors.

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
	"mime"
	"net/http"
	"net/url"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// statusError is an object that can be converted into an metav1.Status
type statusError interface {
	Status() metav1.Status
}

type fakeNegotiater struct {
	serializer, streamSerializer runtime.Serializer
	framer                       runtime.Framer
	types, streamTypes           []string
}

func (n *fakeNegotiater) SupportedMediaTypes() []runtime.SerializerInfo {
	var out []runtime.SerializerInfo
	for _, s := range n.types {
		mediaType, _, err := mime.ParseMediaType(s)
		if err != nil {
			panic(err)
		}
		parts := strings.SplitN(mediaType, "/", 2)
		if len(parts) == 1 {
			// this is an error on the server side
			parts = append(parts, "")
		}

		info := runtime.SerializerInfo{
			Serializer:       n.serializer,
			MediaType:        s,
			MediaTypeType:    parts[0],
			MediaTypeSubType: parts[1],
			EncodesAsText:    true,
		}
		for _, t := range n.streamTypes {
			if t == s {
				info.StreamSerializer = &runtime.StreamSerializerInfo{
					EncodesAsText: true,
					Framer:        n.framer,
					Serializer:    n.streamSerializer,
				}
			}
		}
		out = append(out, info)
	}
	return out
}

func (n *fakeNegotiater) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return n.serializer
}

func (n *fakeNegotiater) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return n.serializer
}

var fakeCodec = runtime.NewCodec(runtime.NoopEncoder{}, runtime.NoopDecoder{})

func TestNegotiate(t *testing.T) {
	testCases := []struct {
		accept      string
		req         *http.Request
		ns          *fakeNegotiater
		serializer  runtime.Serializer
		contentType string
		params      map[string]string
		errFn       func(error) bool
	}{
		// pick a default
		{
			req:         &http.Request{},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "*/*",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "application/*",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "application/json",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "application/json",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json", "application/protobuf"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "application/protobuf",
			contentType: "application/protobuf",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json", "application/protobuf"}},
			serializer:  fakeCodec,
		},
		{
			accept:      "application/json; pretty=1",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},
		{
			accept:      "unrecognized/stuff,application/json; pretty=1",
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},

		// q=0 means "not acceptable": fall through to a type the client still accepts
		{
			accept:      "application/json;q=0,application/protobuf",
			contentType: "application/protobuf",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json", "application/protobuf"}},
			serializer:  fakeCodec,
		},
		// q=0 on the only supported type leaves nothing acceptable
		{
			accept: "application/json;q=0",
			ns:     &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: application/json"
			},
		},
		// a wildcard q=0 rejects everything
		{
			accept: "*/*;q=0",
			ns:     &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: application/json"
			},
		},
		// a specific q=0 overrides a wildcard, so the only supported type is rejected
		{
			accept: "*/*,application/json;q=0",
			ns:     &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: application/json"
			},
		},

		// query param triggers pretty
		{
			req: &http.Request{
				Header: http.Header{"Accept": []string{"application/json"}},
				URL:    &url.URL{RawQuery: "pretty=1"},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},

		// certain user agents trigger pretty
		{
			req: &http.Request{
				Header: http.Header{
					"Accept":     []string{"application/json"},
					"User-Agent": []string{"curl"},
				},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},
		{
			req: &http.Request{
				Header: http.Header{
					"Accept":     []string{"application/json"},
					"User-Agent": []string{"Wget"},
				},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},
		{
			req: &http.Request{
				Header: http.Header{
					"Accept":     []string{"application/json"},
					"User-Agent": []string{"Mozilla/5.0"},
				},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
			params:      map[string]string{"pretty": "1"},
		},
		{
			req: &http.Request{
				Header: http.Header{
					"Accept": []string{"application/json;as=BOGUS;v=v1beta1;g=meta.k8s.io, application/json"},
				},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		{
			req: &http.Request{
				Header: http.Header{
					"Accept": []string{"application/BOGUS, application/json"},
				},
			},
			contentType: "application/json",
			ns:          &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}},
			serializer:  fakeCodec,
		},
		// "application" is not a valid media type, so the server will reject the response during
		// negotiation (the server, in error, has specified an invalid media type)
		{
			accept: "application",
			ns:     &fakeNegotiater{serializer: fakeCodec, types: []string{"application"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: application"
			},
		},
		{
			ns: &fakeNegotiater{},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: "
			},
		},
		{
			accept: "*/*",
			ns:     &fakeNegotiater{},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: "
			},
		},
	}

	for i, test := range testCases {
		req := test.req
		if req == nil {
			req = &http.Request{Header: http.Header{}}
			req.Header.Set("Accept", test.accept)
		}
		_, s, err := NegotiateOutputMediaType(req, test.ns, DefaultEndpointRestrictions)
		switch {
		case err == nil && test.errFn != nil:
			t.Errorf("%d: failed: expected error", i)
			continue
		case err != nil && test.errFn == nil:
			t.Errorf("%d: failed: %v", i, err)
			continue
		case err != nil:
			if !test.errFn(err) {
				t.Errorf("%d: failed: %v", i, err)
			}
			status, ok := err.(statusError)
			if !ok {
				t.Errorf("%d: failed, error should be statusError: %v", i, err)
				continue
			}
			if status.Status().Status != metav1.StatusFailure || status.Status().Code != http.StatusNotAcceptable {
				t.Errorf("%d: failed: %v", i, err)
				continue
			}
			continue
		}
		if test.contentType != s.MediaType {
			t.Errorf("%d: unexpected %s %s", i, test.contentType, s.MediaType)
		}
		if s.Serializer != test.serializer {
			t.Errorf("%d: unexpected %s %s", i, test.serializer, s.Serializer)
		}
	}
}

func fakeSerializerInfoSlice() []runtime.SerializerInfo {
	result := make([]runtime.SerializerInfo, 2)
	result[0] = runtime.SerializerInfo{
		MediaType:        "application/json",
		MediaTypeType:    "application",
		MediaTypeSubType: "json",
	}
	result[1] = runtime.SerializerInfo{
		MediaType:        "application/vnd.kubernetes.protobuf",
		MediaTypeType:    "application",
		MediaTypeSubType: "vnd.kubernetes.protobuf",
	}
	return result
}

func BenchmarkNegotiateMediaTypeOptions(b *testing.B) {
	accepted := fakeSerializerInfoSlice()
	header := "application/vnd.kubernetes.protobuf,*/*"

	for i := 0; i < b.N; i++ {
		options, _ := NegotiateMediaTypeOptions(header, accepted, DefaultEndpointRestrictions)
		if options.Accepted != accepted[1] {
			b.Errorf("Unexpected result")
		}
	}
}

// TestNegotiateAllMediaTypeOptions exercises the plural matcher used by the Accept
// fallback path.
func TestNegotiateAllMediaTypeOptions(t *testing.T) {
	accepted := fakeSerializerInfoSlice() // [json, protobuf]

	testCases := []struct {
		name   string
		header string
		want   []string // MediaType strings in expected order
	}{
		{
			name:   "empty header defaults to first supported",
			header: "",
			want:   []string{"application/json"},
		},
		{
			name:   "single preferred type",
			header: "application/json",
			want:   []string{"application/json"},
		},
		{
			name:   "proto then json preserves order",
			header: "application/vnd.kubernetes.protobuf,application/json",
			want:   []string{"application/vnd.kubernetes.protobuf", "application/json"},
		},
		{
			name:   "json then proto preserves order",
			header: "application/json,application/vnd.kubernetes.protobuf",
			want:   []string{"application/json", "application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "wildcard expands to all supported without duplicates",
			header: "application/vnd.kubernetes.protobuf,*/*",
			want:   []string{"application/vnd.kubernetes.protobuf", "application/json"},
		},
		{
			name:   "unsupported returns empty",
			header: "unrecognized/stuff",
			want:   nil,
		},
		{
			name:   "quality-weighted order",
			header: "application/json;q=0.5,application/vnd.kubernetes.protobuf;q=1.0",
			want:   []string{"application/vnd.kubernetes.protobuf", "application/json"},
		},
		{
			name:   "q=0 excludes the rejected type",
			header: "application/vnd.kubernetes.protobuf,application/json;q=0",
			want:   []string{"application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "q=0 wildcard excludes everything",
			header: "*/*;q=0",
			want:   nil,
		},
		{
			name:   "specific q=0 overrides wildcard",
			header: "application/vnd.kubernetes.protobuf,*/*,application/json;q=0",
			want:   []string{"application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "wildcard accepts the rest while a specific q=0 is rejected",
			header: "*/*,application/json;q=0",
			want:   []string{"application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "wildcard is a fallback when a specific clause fails on params",
			header: "application/json;as=Table,*/*",
			want:   []string{"application/json", "application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "type wildcard expands to all matching subtypes",
			header: "application/*",
			want:   []string{"application/json", "application/vnd.kubernetes.protobuf"},
		},
		{
			name:   "type wildcard q=0 rejects all matching subtypes",
			header: "application/*;q=0",
			want:   nil,
		},
		{
			name:   "exact clause overrides a type wildcard q=0",
			header: "application/json,application/*;q=0",
			want:   []string{"application/json"},
		},
		{
			name:   "lone q=0 rejection matches nothing",
			header: "application/json;q=0",
			want:   nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := negotiateAllMediaTypeOptions(tc.header, accepted, DefaultEndpointRestrictions)
			if len(got) != len(tc.want) {
				t.Fatalf("got %d results, want %d: %+v", len(got), len(tc.want), got)
			}
			for i := range got {
				if got[i].Accepted.MediaType != tc.want[i] {
					t.Errorf("position %d: got %q, want %q", i, got[i].Accepted.MediaType, tc.want[i])
				}
			}
		})
	}
}

// TestNegotiateOutputMediaTypes_NotAcceptable verifies the plural API returns a
// NotAcceptableError when no clause matches any supported serializer.
func TestNegotiateOutputMediaTypes_NotAcceptable(t *testing.T) {
	ns := &fakeNegotiater{serializer: fakeCodec, types: []string{"application/json"}}
	req := &http.Request{Header: http.Header{"Accept": []string{"text/html"}}}
	mts, err := NegotiateOutputMediaTypes(req, ns, DefaultEndpointRestrictions)
	if err == nil {
		t.Fatalf("expected NotAcceptableError, got nil; mts=%v", mts)
	}
	s, ok := err.(statusError)
	if !ok {
		t.Fatalf("expected statusError, got %T", err)
	}
	if s.Status().Code != http.StatusNotAcceptable {
		t.Errorf("expected 406, got %d", s.Status().Code)
	}
}
