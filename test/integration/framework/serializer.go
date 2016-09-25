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

package framework

import (
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
)

// NewSingleContentTypeSerializer wraps a serializer in a NegotiatedSerializer that handles one content type
func NewSingleContentTypeSerializer(scheme *runtime.Scheme, serializer runtime.Serializer, contentType string) runtime.StorageSerializer {
	return &wrappedSerializer{
		scheme:      scheme,
		serializer:  serializer,
		contentType: contentType,
	}
}

type wrappedSerializer struct {
	scheme      *runtime.Scheme
	serializer  runtime.Serializer
	contentType string
}

var _ runtime.StorageSerializer = &wrappedSerializer{}

func (s *wrappedSerializer) SupportedMediaTypes() []string {
	return []string{s.contentType}
}
func (s *wrappedSerializer) SerializerForMediaType(mediaType string, options map[string]string) (runtime.SerializerInfo, bool) {
	if mediaType != s.contentType {
		return runtime.SerializerInfo{}, false
	}

	return runtime.SerializerInfo{
		Serializer:    s.serializer,
		MediaType:     mediaType,
		EncodesAsText: true, // TODO: this should be parameterized
	}, true
}

func (s *wrappedSerializer) UniversalDeserializer() runtime.Decoder {
	return s.serializer
}

func (s *wrappedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewCodec(encoder, nil, s.scheme, s.scheme, s.scheme, s.scheme, gv, nil)
}

func (s *wrappedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return versioning.NewCodec(nil, decoder, s.scheme, s.scheme, s.scheme, s.scheme, nil, gv)
}
