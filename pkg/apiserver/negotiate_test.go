/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type fakeNegotiater struct {
	serializer, streamSerializer runtime.Serializer
	framer                       runtime.Framer
	types, streamTypes           []string
	mediaType, streamMediaType   string
	options, streamOptions       map[string]string
}

func (n *fakeNegotiater) SupportedMediaTypes() []string {
	return n.types
}
func (n *fakeNegotiater) SupportedStreamingMediaTypes() []string {
	return n.streamTypes
}

func (n *fakeNegotiater) SerializerForMediaType(mediaType string, options map[string]string) (runtime.SerializerInfo, bool) {
	n.mediaType = mediaType
	if len(options) > 0 {
		n.options = options
	}
	return runtime.SerializerInfo{Serializer: n.serializer, MediaType: n.mediaType, EncodesAsText: true}, n.serializer != nil
}

func (n *fakeNegotiater) StreamingSerializerForMediaType(mediaType string, options map[string]string) (runtime.StreamSerializerInfo, bool) {
	n.streamMediaType = mediaType
	if len(options) > 0 {
		n.streamOptions = options
	}
	return runtime.StreamSerializerInfo{
		SerializerInfo: runtime.SerializerInfo{
			Serializer:    n.serializer,
			MediaType:     mediaType,
			EncodesAsText: true,
		},
		Framer: n.framer,
	}, n.streamSerializer != nil
}

func (n *fakeNegotiater) EncoderForVersion(serializer runtime.Encoder, gv unversioned.GroupVersion) runtime.Encoder {
	return n.serializer
}

func (n *fakeNegotiater) DecoderToVersion(serializer runtime.Decoder, gv unversioned.GroupVersion) runtime.Decoder {
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
			ns: &fakeNegotiater{types: []string{"a/b/c"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: a/b/c"
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
		{
			accept: "application/json",
			ns:     &fakeNegotiater{types: []string{"application/json"}},
			errFn: func(err error) bool {
				return err.Error() == "only the following media types are accepted: application/json"
			},
		},
	}

	for i, test := range testCases {
		req := test.req
		if req == nil {
			req = &http.Request{Header: http.Header{}}
			req.Header.Set("Accept", test.accept)
		}
		s, err := negotiateOutputSerializer(req, test.ns)
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
			if status.Status().Status != unversioned.StatusFailure || status.Status().Code != http.StatusNotAcceptable {
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
		if !reflect.DeepEqual(test.params, test.ns.options) {
			t.Errorf("%d: unexpected %#v %#v", i, test.params, test.ns.options)
		}
	}
}
