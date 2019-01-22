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
	"net/http"
	"net/url"
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
		info := runtime.SerializerInfo{Serializer: n.serializer, MediaType: s, EncodesAsText: true}
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
		s, err := NegotiateOutputSerializer(req, test.ns)
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
