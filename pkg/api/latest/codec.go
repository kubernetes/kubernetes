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

package latest

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/runtime/serializer/recognizer"
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
)

var (
	Codecs                CodecFactory
	UniversalDeserializer runtime.Decoder

	jsonSerializer       = json.NewSerializer(json.DefaultMetaFactory, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme), false)
	jsonPrettySerializer = json.NewSerializer(json.DefaultMetaFactory, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme), true)
	yamlSerializer       = json.NewYAMLSerializer(json.DefaultMetaFactory, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme))
)

type serializerType struct {
	AcceptContentTypes []string
	ContentType        string
	FileExtensions     []string
	Serializer         runtime.Serializer
	PrettySerializer   runtime.Serializer
}

// allows other codecs to be enabled with compile time flags in their own files
var serializers = []serializerType{
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

func init() {
	Codecs = CodecFactory{
		scheme:      api.Scheme,
		serializers: serializers,
	}
	decoders := make([]runtime.Decoder, 0, len(serializers))
	for _, d := range serializers {
		decoders = append(decoders, d.Serializer)
	}
	UniversalDeserializer = recognizer.NewDecoder(decoders...)
}

type CodecFactory struct {
	scheme      *runtime.Scheme
	serializers []serializerType
}

// UniversalDecoder returns a runtime.Decoder capable of decoding all known API objects in all known formats. Used
// by clients that do not need to encode objects but want to deserialize API objects stored on disk. Only decodes
// objects in groups registered with the scheme. The GroupVersions passed may be used to select alternate
// versions of objects to return - by default, runtime.APIVersionInternal is used. If any versions are specified,
// unrecognized groups will be returned in the version they are encoded as (no conversion).
func (f CodecFactory) UniversalDecoder(versions ...unversioned.GroupVersion) runtime.Decoder {
	return f.CodecForVersions(runtime.NoopEncoder{UniversalDeserializer}, nil, versions)
}

// CodecFor creates a codec with the provided serializer. If an object is decoded and its group is not in the list,
// it will default to runtiem.APIVersionInternal. If encode is not specified for an object's group, the object is not
// converted. If encode or decode are nil, no conversion is performed.
func (CodecFactory) CodecForVersions(serializer runtime.Serializer, encode []unversioned.GroupVersion, decode []unversioned.GroupVersion) runtime.Codec {
	return versioning.NewCodec(api.Scheme, serializer, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme), encode, decode)
}

// SerializerForMediaType returns a serializer that matches the provided RFC2046 mediaType, or false if no such
// serializer exists
func (f CodecFactory) SerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, bool) {
	for _, s := range f.serializers {
		for _, accepted := range s.AcceptContentTypes {
			if accepted == mediaType {
				if _, ok := options["pretty"]; ok && s.PrettySerializer != nil {
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
