/*
Copyright 2019 The Kubernetes Authors.

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

package serializer

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
)

// strictSerializerExtensions are for serializers that are conditionally compiled in.
// Note that this is a separate variable from serializer.CodecFactory.
var strictSerializerExtensions = []func(*runtime.Scheme) (serializerType, bool){}

// NewStrictUniversalDecoder returns a runtime.Decoder which operates in a strict manner.
// This Decoder is capable of decoding all known API objects in all known formats. It is used by clients that do not need
// to encode objects but want to deserialize API objects stored on disk. It only decodes objects in groups registered
// with the scheme. GroupVersions may be passed in `versions` to specify alternate versions of objects to return.
// Populating `versions` is optional; a zero-length `versions` will result in runtime.InternalGroupVersioner -- this matches
// behavior of serializer.CodecFactory.UniversalDecoder().
// If any versions are specified, unrecognized groups will be returned in the version they are encoded as (no conversion).
// This decoder performs defaulting.
//
// This constructor is not included in serializer.CodecFactory because all runtime Decode operations are intended to be strict in the future.
// The current strict checking implementation is not fit for the hot-path of the apiserver due to performance reasons.
// This constructor is a temporary solution until we can rely on serializer.CodecFactory.UniversalDecoder() to return Strict errors.
//
// TODO: the decoder will eventually be removed in favor of dealing with objects in their versioned form
// TODO: only accept a group versioner
func NewStrictUniversalDecoder(scheme *runtime.Scheme, versions []schema.GroupVersion) runtime.Decoder {
	var versioner runtime.GroupVersioner
	if len(versions) == 0 {
		versioner = runtime.InternalGroupVersioner
	} else {
		versioner = schema.GroupVersions(versions)
	}

	codecs := serializer.NewCodecFactory(scheme)

	serializers := newStrictSerializersForScheme(scheme, json.DefaultMetaFactory)
	yamlStrictSerializer := newStrictUniversalDeserializer(scheme, serializers)

	return codecs.CodecForVersions(nil, yamlStrictSerializer, runtime.DisabledGroupVersioner, versioner)
}

// serializerType is an exact copy of serializer.CodecFactory.serializerType
type serializerType struct {
	AcceptContentTypes []string
	ContentType        string
	FileExtensions     []string
	// EncodesAsText should be true if this content type can be represented safely in UTF-8
	EncodesAsText bool

	Serializer       runtime.Serializer
	PrettySerializer runtime.Serializer

	AcceptStreamContentTypes []string
	StreamContentType        string

	Framer           runtime.Framer
	StreamSerializer runtime.Serializer
}

// newStrictUniversalDeserializer is a paired-down version of serializer.CodecFactory.newCodecFactory() that only returns f.universal.
// It is functionally similar to serializer.CodecFactory.UniversalDeserializer().
// This Decoder does not perform defaulting
func newStrictUniversalDeserializer(scheme *runtime.Scheme, serializers []serializerType) runtime.Decoder {
	decoders := make([]runtime.Decoder, 0, len(serializers))
	for _, d := range serializers {
		decoders = append(decoders, d.Serializer)
	}
	return recognizer.NewDecoder(decoders...)
}

// newStrictSerializersForScheme is a strict variant of serializer.CodecFactory.newSerializersForScheme()
func newStrictSerializersForScheme(scheme *runtime.Scheme, mf json.MetaFactory) []serializerType {
	jsonSerializer := json.NewSerializerWithOptions(
		mf, scheme, scheme,
		json.SerializerOptions{Yaml: false, Pretty: false, Strict: true},
	)
	jsonPrettySerializer := json.NewSerializerWithOptions(
		mf, scheme, scheme,
		json.SerializerOptions{Yaml: false, Pretty: true, Strict: true},
	)
	yamlSerializer := json.NewSerializerWithOptions(
		mf, scheme, scheme,
		json.SerializerOptions{Yaml: true, Pretty: false, Strict: true},
	)

	serializers := []serializerType{
		{
			AcceptContentTypes: []string{"application/json"},
			ContentType:        "application/json",
			FileExtensions:     []string{"json"},
			EncodesAsText:      true,
			Serializer:         jsonSerializer,
			PrettySerializer:   jsonPrettySerializer,

			Framer:           json.Framer,
			StreamSerializer: jsonSerializer,
		},
		{
			AcceptContentTypes: []string{"application/yaml"},
			ContentType:        "application/yaml",
			FileExtensions:     []string{"yaml"},
			EncodesAsText:      true,
			Serializer:         yamlSerializer,
		},
	}

	for _, fn := range strictSerializerExtensions {
		if serializer, ok := fn(scheme); ok {
			serializers = append(serializers, serializer)
		}
	}
	return serializers
}
