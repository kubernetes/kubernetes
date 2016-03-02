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

// Typer retrieves information about an object's group, version, and kind.
type Typer interface {
	// ObjectKind returns the version and kind of the provided object, or an
	// error if the object is not recognized (IsNotRegisteredError will return true).
	// It returns whether the object is considered unversioned at the same time.
	// TODO: align the signature of ObjectTyper with this interface
	ObjectKind(Object) (*unversioned.GroupVersionKind, bool, error)
}

type Encoder interface {
	// EncodeToStream writes an object to a stream. Override versions may be provided for each group
	// that enforce a certain versioning. Implementations may return errors if the versions are incompatible,
	// or if no conversion is defined.
	EncodeToStream(obj Object, stream io.Writer, overrides ...unversioned.GroupVersion) error
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

// NegotiatedSerializer is an interface used for obtaining encoders, decoders, and serializers
// for multiple supported media types.
type NegotiatedSerializer interface {
	SupportedMediaTypes() []string
	SerializerForMediaType(mediaType string, options map[string]string) (Serializer, bool)
	EncoderForVersion(serializer Serializer, gv unversioned.GroupVersion) Encoder
	DecoderToVersion(serializer Serializer, gv unversioned.GroupVersion) Decoder
}

///////////////////////////////////////////////////////////////////////////////
// Non-codec interfaces

type ObjectVersioner interface {
	ConvertToVersion(in Object, outVersion string) (out Object, err error)
}

// ObjectConvertor converts an object to a different version.
type ObjectConvertor interface {
	// Convert attempts to convert one object into another, or returns an error. This method does
	// not guarantee the in object is not mutated.
	Convert(in, out interface{}) error
	// ConvertToVersion takes the provided object and converts it the provided version. This
	// method does not guarantee that the in object is not mutated.
	ConvertToVersion(in Object, outVersion string) (out Object, err error)
	ConvertFieldLabel(version, kind, label, value string) (string, string, error)
}

// ObjectTyper contains methods for extracting the APIVersion and Kind
// of objects.
type ObjectTyper interface {
	// ObjectKind returns the default group,version,kind of the provided object, or an
	// error if the object is not recognized (IsNotRegisteredError will return true).
	ObjectKind(Object) (unversioned.GroupVersionKind, error)
	// ObjectKinds returns the all possible group,version,kind of the provided object, or an
	// error if the object is not recognized (IsNotRegisteredError will return true).
	ObjectKinds(Object) ([]unversioned.GroupVersionKind, error)
	// Recognizes returns true if the scheme is able to handle the provided version and kind,
	// or more precisely that the provided version is a possible conversion or decoding
	// target.
	Recognizes(gvk unversioned.GroupVersionKind) bool
	// IsUnversioned returns true if the provided object is considered unversioned and thus
	// should have Version and Group suppressed in the output. If the object is not recognized
	// in the scheme, ok is false.
	IsUnversioned(Object) (unversioned bool, ok bool)
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
