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
	"fmt"
	"io"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
	"k8s.io/kubernetes/pkg/runtime/serializer/recognizer"
)

var (
	Codecs                CodecFactory
	UniversalDeserializer runtime.Decoder

	jsonSerializer = json.NewSerializer(conversion.DefaultMetaFactory, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme))
	yamlSerializer = json.NewYAMLSerializer(conversion.DefaultMetaFactory, api.Scheme, runtime.ObjectTyperToTyper(api.Scheme))
)

type serializerType struct {
	AcceptContentTypes []string
	ContentType        string
	FileExtensions     []string
	Serializer         runtime.Serializer
}

// allows other codecs to be enabled with compile time flags in their own files
var serializers = []serializerType{
	{
		AcceptContentTypes: []string{"application/json"},
		ContentType:        "application/json",
		FileExtensions:     []string{"json"},
		Serializer:         jsonSerializer,
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

type noopEncoder struct {
	runtime.Decoder
}

func (e noopEncoder) EncodeToStream(obj runtime.Object, w io.Writer) error {
	return fmt.Errorf("encoding is not allowed for this codec: %v", reflect.TypeOf(e.Decoder))
}

// UniversalDecoder returns a runtime.Decoder capable of decoding all known API objects in all known formats. Used
// by clients that do not need to encode objects but want to deserialize API objects stored on disk. Only decodes
// objects in groups registered with the scheme. The GroupVersions passed may be used to select alternate
// versions of objects to return - by default, runtime.APIVersionInternal is used. If any versions are specified,
// unrecognized groups will be returned in the version they are encoded as (no conversion).
func (f CodecFactory) UniversalDecoder(versions ...unversioned.GroupVersion) runtime.Decoder {
	return f.CodecForVersions(noopEncoder{UniversalDeserializer}, versions...)
}

// CodecFor creates a codec with the provided serializer. If versions are specified, objects are converted to
// those versions in each group, or left unchanged if not recognized. If no versions are specified,
// runtime.APIVersionInternal is used.
func (CodecFactory) CodecForVersions(serializer runtime.Serializer, versions ...unversioned.GroupVersion) runtime.Codec {
	return nil
}

// SerializerForContentTypes returns the first serializer that matches thet provided RFC2046 media types (in order), or
// false if no serializer matched.
func (f CodecFactory) SerializerForContentTypes(mediaTypes ...string) (runtime.Serializer, bool) {
	for _, s := range f.serializers {
		for _, accept := range s.AcceptContentTypes {
			for _, expect := range mediaTypes {
				if expect == accept {
					return s.Serializer, true
				}
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
