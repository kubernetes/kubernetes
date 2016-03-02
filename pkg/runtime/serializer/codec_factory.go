/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/runtime/serializer/recognizer"
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
)

type serializerType struct {
	AcceptContentTypes []string
	ContentType        string
	FileExtensions     []string
	Serializer         runtime.Serializer
	PrettySerializer   runtime.Serializer
}

// NewCodecFactory provides methods for retrieving serializers for the supported wire formats
// and conversion wrappers to define preferred internal and external versions. In the future,
// as the internal version is used less, callers may instead use a defaulting serializer and
// only convert objects which are shared internally (Status, common API machinery).
// TODO: allow other codecs to be compiled in?
// TODO: accept a scheme interface
func NewCodecFactory(scheme *runtime.Scheme) CodecFactory {
	return newCodecFactory(scheme, json.DefaultMetaFactory)
}

// newCodecFactory is a helper for testing that allows a different metafactory to be specified.
func newCodecFactory(scheme *runtime.Scheme, mf json.MetaFactory) CodecFactory {
	jsonSerializer := json.NewSerializer(mf, scheme, runtime.ObjectTyperToTyper(scheme), false)
	jsonPrettySerializer := json.NewSerializer(mf, scheme, runtime.ObjectTyperToTyper(scheme), true)
	yamlSerializer := json.NewYAMLSerializer(mf, scheme, runtime.ObjectTyperToTyper(scheme))
	serializers := []serializerType{
		{
			AcceptContentTypes: []string{"application/json"},
			ContentType:        "application/json",
			FileExtensions:     []string{"json"},
			Serializer:         jsonSerializer,
			PrettySerializer:   jsonPrettySerializer,
		},
		{
			AcceptContentTypes: []string{"application/yaml"},
			ContentType:        "application/yaml",
			FileExtensions:     []string{"yaml"},
			Serializer:         yamlSerializer,
		},
	}
	decoders := make([]runtime.Decoder, 0, len(serializers))
	accepts := []string{}
	alreadyAccepted := make(map[string]struct{})
	for _, d := range serializers {
		decoders = append(decoders, d.Serializer)
		for _, mediaType := range d.AcceptContentTypes {
			if _, ok := alreadyAccepted[mediaType]; ok {
				continue
			}
			alreadyAccepted[mediaType] = struct{}{}
			accepts = append(accepts, mediaType)
		}
	}
	return CodecFactory{
		scheme:      scheme,
		serializers: serializers,
		universal:   recognizer.NewDecoder(decoders...),
		accepts:     accepts,

		legacySerializer: jsonSerializer,
	}
}

// CodecFactory provides methods for retrieving codecs and serializers for specific
// versions and content types.
type CodecFactory struct {
	scheme      *runtime.Scheme
	serializers []serializerType
	universal   runtime.Decoder
	accepts     []string

	legacySerializer runtime.Serializer
}

var _ runtime.NegotiatedSerializer = &CodecFactory{}

// SupportedMediaTypes returns the RFC2046 media types that this factory has serializers for.
func (f CodecFactory) SupportedMediaTypes() []string {
	return f.accepts
}

// LegacyCodec encodes output to a given API version, and decodes output into the internal form from
// any recognized source. The returned codec will always encode output to JSON.
//
// This method is deprecated - clients and servers should negotiate a serializer by mime-type and
// invoke CodecForVersions. Callers that need only to read data should use UniversalDecoder().
func (f CodecFactory) LegacyCodec(version ...unversioned.GroupVersion) runtime.Codec {
	return f.CodecForVersions(runtime.NewCodec(f.legacySerializer, f.universal), version, nil)
}

// UniversalDeserializer can convert any stored data recognized by this factory into a Go object that satisfies
// runtime.Object. It does not perform conversion. It does not perform defaulting.
func (f CodecFactory) UniversalDeserializer() runtime.Decoder {
	return f.universal
}

// UniversalDecoder returns a runtime.Decoder capable of decoding all known API objects in all known formats. Used
// by clients that do not need to encode objects but want to deserialize API objects stored on disk. Only decodes
// objects in groups registered with the scheme. The GroupVersions passed may be used to select alternate
// versions of objects to return - by default, runtime.APIVersionInternal is used. If any versions are specified,
// unrecognized groups will be returned in the version they are encoded as (no conversion). This decoder performs
// defaulting.
//
// TODO: the decoder will eventually be removed in favor of dealing with objects in their versioned form
func (f CodecFactory) UniversalDecoder(versions ...unversioned.GroupVersion) runtime.Decoder {
	return f.CodecForVersions(runtime.NoopEncoder{f.universal}, nil, versions)
}

// CodecFor creates a codec with the provided serializer. If an object is decoded and its group is not in the list,
// it will default to runtime.APIVersionInternal. If encode is not specified for an object's group, the object is not
// converted. If encode or decode are nil, no conversion is performed.
func (f CodecFactory) CodecForVersions(serializer runtime.Serializer, encode []unversioned.GroupVersion, decode []unversioned.GroupVersion) runtime.Codec {
	return versioning.NewCodecForScheme(f.scheme, serializer, encode, decode)
}

// DecoderToVersion returns a decoder that targets the provided group version.
func (f CodecFactory) DecoderToVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Decoder {
	return f.CodecForVersions(serializer, nil, []unversioned.GroupVersion{gv})
}

// EncoderForVersion returns an encoder that targets the provided group version.
func (f CodecFactory) EncoderForVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Encoder {
	return f.CodecForVersions(serializer, []unversioned.GroupVersion{gv}, nil)
}

// SerializerForMediaType returns a serializer that matches the provided RFC2046 mediaType, or false if no such
// serializer exists
func (f CodecFactory) SerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, bool) {
	for _, s := range f.serializers {
		for _, accepted := range s.AcceptContentTypes {
			if accepted == mediaType {
				if v, ok := options["pretty"]; ok && v == "1" && s.PrettySerializer != nil {
					return s.PrettySerializer, true
				}
				return s.Serializer, true
			}
		}
	}
	return nil, false
}

// SerializerForFileExtension returns a serializer for the provided extension, or false if no serializer matches.
func (f CodecFactory) SerializerForFileExtension(extension string) (runtime.Serializer, bool) {
	for _, s := range f.serializers {
		for _, ext := range s.FileExtensions {
			if extension == ext {
				return s.Serializer, true
			}
		}
	}
	return nil, false
}
