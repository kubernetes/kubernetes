/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
)

// NewSingleContentTypeSerializer wraps a serializer in a NegotiatedSerializer that handles one content type
func NewSingleContentTypeSerializer(scheme *runtime.Scheme, serializer runtime.Serializer, contentType string) runtime.NegotiatedSerializer {
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

var _ runtime.NegotiatedSerializer = &wrappedSerializer{}

func (s *wrappedSerializer) SupportedMediaTypes() []string {
	return []string{s.contentType}
}
func (s *wrappedSerializer) SerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, bool) {
	if mediaType != s.contentType {
		return nil, false
	}

	return s.serializer, true
}

func (s *wrappedSerializer) EncoderForVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Encoder {
	return versioning.NewCodec(s.serializer, s.serializer, s.scheme, s.scheme, s.scheme, runtime.ObjectTyperToTyper(s.scheme), []unversioned.GroupVersion{gv}, nil)
}

func (s *wrappedSerializer) DecoderToVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Decoder {
	return versioning.NewCodec(s.serializer, s.serializer, s.scheme, s.scheme, s.scheme, runtime.ObjectTyperToTyper(s.scheme), nil, []unversioned.GroupVersion{gv})
}
