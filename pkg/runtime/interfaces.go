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

package runtime

import (
	"io"
	"net/url"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

const (
	// APIVersionInternal may be used if you are registering a type that should not
	// be considered stable or serialized - it is a convention only and has no
	// special behavior in this package.
	APIVersionInternal = "__internal"
)

type Encoder interface {
	// Encode writes an object to a stream. Implementations may return errors if the versions are
	// incompatible, or if no conversion is defined.
	Encode(obj Object, w io.Writer) error
}

type Decoder interface {
	// Decode attempts to deserialize the provided data using either the innate typing of the scheme or the
	// default kind, group, and version provided. It returns a decoded object as well as the kind, group, and
	// version from the serialized data, or an error. If into is non-nil, it will be used as the target type
	// and implementations may choose to use it rather than reallocating an object. However, the object is not
	// guaranteed to be populated. The returned object is not guaranteed to match into. If defaults are
	// provided, they are applied to the data by default. If no defaults or partial defaults are provided, the
	// type of the into may be used to guide conversion decisions.
	Decode(data []byte, defaults *unversioned.GroupVersionKind, into Object) (Object, *unversioned.GroupVersionKind, error)
}

// Serializer is the core interface for transforming objects into a serialized format and back.
// Implementations may choose to perform conversion of the object, but no assumptions should be made.
type Serializer interface {
	Encoder
	Decoder
}

// Codec is a Serializer that deals with the details of versioning objects. It offers the same
// interface as Serializer, so this is a marker to consumers that care about the version of the objects
// they receive.
type Codec Serializer

// ParameterCodec defines methods for serializing and deserializing API objects to url.Values and
// performing any necessary conversion. Unlike the normal Codec, query parameters are not self describing
// and the desired version must be specified.
type ParameterCodec interface {
	// DecodeParameters takes the given url.Values in the specified group version and decodes them
	// into the provided object, or returns an error.
	DecodeParameters(parameters url.Values, from unversioned.GroupVersion, into Object) error
	// EncodeParameters encodes the provided object as query parameters or returns an error.
	EncodeParameters(obj Object, to unversioned.GroupVersion) (url.Values, error)
}

// Framer is a factory for creating readers and writers that obey a particular framing pattern.
type Framer interface {
	NewFrameReader(r io.ReadCloser) io.ReadCloser
	NewFrameWriter(w io.Writer) io.Writer
}

// SerializerInfo contains information about a specific serialization format
type SerializerInfo struct {
	Serializer
	// EncodesAsText indicates this serializer can be encoded to UTF-8 safely.
	EncodesAsText bool
	// MediaType is the value that represents this serializer over the wire.
	MediaType string
}

// StreamSerializerInfo contains information about a specific stream serialization format
type StreamSerializerInfo struct {
	SerializerInfo
	// Framer is the factory for retrieving streams that separate objects on the wire
	Framer
	// Embedded is the type of the nested serialization that should be used.
	Embedded SerializerInfo
}

// NegotiatedSerializer is an interface used for obtaining encoders, decoders, and serializers
// for multiple supported media types. This would commonly be accepted by a server component
// that performs HTTP content negotiation to accept multiple formats.
type NegotiatedSerializer interface {
	// SupportedMediaTypes is the media types supported for reading and writing single objects.
	SupportedMediaTypes() []string
	// SerializerForMediaType returns a serializer for the provided media type. params is the set of
	// parameters applied to the media type that may modify the resulting output. ok will be false
	// if no serializer matched the media type.
	SerializerForMediaType(mediaType string, params map[string]string) (s SerializerInfo, ok bool)

	// SupportedStreamingMediaTypes returns the media types of the supported streaming serializers.
	// Streaming serializers control how multiple objects are written to a stream output.
	SupportedStreamingMediaTypes() []string
	// StreamingSerializerForMediaType returns a serializer for the provided media type that supports
	// reading and writing multiple objects to a stream. It returns a framer and serializer, or an
	// error if no such serializer can be created. Params is the set of parameters applied to the
	// media type that may modify the resulting output. ok will be false if no serializer matched
	// the media type.
	StreamingSerializerForMediaType(mediaType string, params map[string]string) (s StreamSerializerInfo, ok bool)

	// EncoderForVersion returns an encoder that ensures objects being written to the provided
	// serializer are in the provided group version.
	// TODO: take multiple group versions
	EncoderForVersion(serializer Encoder, gv unversioned.GroupVersion) Encoder
	// DecoderForVersion returns a decoder that ensures objects being read by the provided
	// serializer are in the provided group version by default.
	// TODO: take multiple group versions
	DecoderToVersion(serializer Decoder, gv unversioned.GroupVersion) Decoder
}

// StorageSerializer is an interface used for obtaining encoders, decoders, and serializers
// that can read and write data at rest. This would commonly be used by client tools that must
// read files, or server side storage interfaces that persist restful objects.
type StorageSerializer interface {
	// SerializerForMediaType returns a serializer for the provided media type.  Options is a set of
	// parameters applied to the media type that may modify the resulting output.
	SerializerForMediaType(mediaType string, options map[string]string) (SerializerInfo, bool)

	// UniversalDeserializer returns a Serializer that can read objects in multiple supported formats
	// by introspecting the data at rest.
	UniversalDeserializer() Decoder

	// EncoderForVersion returns an encoder that ensures objects being written to the provided
	// serializer are in the provided group version.
	// TODO: take multiple group versions
	EncoderForVersion(serializer Encoder, gv unversioned.GroupVersion) Encoder
	// DecoderForVersion returns a decoder that ensures objects being read by the provided
	// serializer are in the provided group version by default.
	// TODO: take multiple group versions
	DecoderToVersion(serializer Decoder, gv unversioned.GroupVersion) Decoder
}

///////////////////////////////////////////////////////////////////////////////
// Non-codec interfaces

type ObjectVersioner interface {
	ConvertToVersion(in Object, outVersion unversioned.GroupVersion) (out Object, err error)
}

// ObjectConvertor converts an object to a different version.
type ObjectConvertor interface {
	// Convert attempts to convert one object into another, or returns an error. This method does
	// not guarantee the in object is not mutated.
	Convert(in, out interface{}) error
	// ConvertToVersion takes the provided object and converts it the provided version. This
	// method does not guarantee that the in object is not mutated.
	ConvertToVersion(in Object, outVersion unversioned.GroupVersion) (out Object, err error)
	ConvertFieldLabel(version, kind, label, value string) (string, string, error)
}

// ObjectTyper contains methods for extracting the APIVersion and Kind
// of objects.
type ObjectTyper interface {
	// ObjectKinds returns the all possible group,version,kind of the provided object, true if
	// the object is unversioned, or an error if the object is not recognized
	// (IsNotRegisteredError will return true).
	ObjectKinds(Object) ([]unversioned.GroupVersionKind, bool, error)
	// Recognizes returns true if the scheme is able to handle the provided version and kind,
	// or more precisely that the provided version is a possible conversion or decoding
	// target.
	Recognizes(gvk unversioned.GroupVersionKind) bool
}

// ObjectCreater contains methods for instantiating an object by kind and version.
type ObjectCreater interface {
	New(kind unversioned.GroupVersionKind) (out Object, err error)
}

// ObjectCopier duplicates an object.
type ObjectCopier interface {
	// Copy returns an exact copy of the provided Object, or an error if the
	// copy could not be completed.
	Copy(Object) (Object, error)
}

// ResourceVersioner provides methods for setting and retrieving
// the resource version from an API object.
type ResourceVersioner interface {
	SetResourceVersion(obj Object, version string) error
	ResourceVersion(obj Object) (string, error)
}

// SelfLinker provides methods for setting and retrieving the SelfLink field of an API object.
type SelfLinker interface {
	SetSelfLink(obj Object, selfLink string) error
	SelfLink(obj Object) (string, error)

	// Knowing Name is sometimes necessary to use a SelfLinker.
	Name(obj Object) (string, error)
	// Knowing Namespace is sometimes necessary to use a SelfLinker
	Namespace(obj Object) (string, error)
}

// All API types registered with Scheme must support the Object interface. Since objects in a scheme are
// expected to be serialized to the wire, the interface an Object must provide to the Scheme allows
// serializers to set the kind, version, and group the object is represented as. An Object may choose
// to return a no-op ObjectKindAccessor in cases where it is not expected to be serialized.
type Object interface {
	GetObjectKind() unversioned.ObjectKind
}
