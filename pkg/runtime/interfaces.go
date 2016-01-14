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

// Codec defines methods for serializing and deserializing API objects.
type Codec interface {
	Decoder
	Encoder
}

// Decoder defines methods for deserializing API objects into a given type
type Decoder interface {
	// TODO: change the signature of this method
	Decode(data []byte) (Object, error)
	// DEPRECATED: This method is being removed
	DecodeToVersion(data []byte, groupVersion unversioned.GroupVersion) (Object, error)
	// DEPRECATED: This method is being removed
	DecodeInto(data []byte, obj Object) error
	// DEPRECATED: This method is being removed
	DecodeIntoWithSpecifiedVersionKind(data []byte, obj Object, groupVersionKind unversioned.GroupVersionKind) error

	DecodeParametersInto(parameters url.Values, obj Object) error
}

// Encoder defines methods for serializing API objects into bytes
type Encoder interface {
	// DEPRECATED: This method is being removed
	Encode(obj Object) (data []byte, err error)
	EncodeToStream(obj Object, stream io.Writer) error

	// TODO: Add method for processing url parameters.
	// EncodeParameters(obj Object) (url.Values, error)
}

// ObjectCodec represents the common mechanisms for converting to and from a particular
// binary representation of an object.
// TODO: Remove this interface - it is used only in CodecFor() method.
type ObjectCodec interface {
	Decoder

	// EncodeToVersion convert and serializes an object in the internal format
	// to a specified output version. An error is returned if the object
	// cannot be converted for any reason.
	EncodeToVersion(obj Object, outVersion string) ([]byte, error)
	EncodeToVersionStream(obj Object, outVersion string, stream io.Writer) error
}

// ObjectDecoder is a convenience interface for identifying serialized versions of objects
// and transforming them into Objects. It intentionally overlaps with ObjectTyper and
// Decoder for use in decode only paths.
// TODO: Consider removing this interface?
type ObjectDecoder interface {
	Decoder
	// DataVersionAndKind returns the group,version,kind of the provided data, or an error
	// if another problem is detected. In many cases this method can be as expensive to
	// invoke as the Decode method.
	DataKind([]byte) (unversioned.GroupVersionKind, error)
	// Recognizes returns true if the scheme is able to handle the provided group,version,kind
	// of an object.
	Recognizes(unversioned.GroupVersionKind) bool
}

///////////////////////////////////////////////////////////////////////////////
// Non-codec interfaces

// ObjectConvertor converts an object to a different version.
type ObjectConvertor interface {
	Convert(in, out interface{}) error
	ConvertToVersion(in Object, outVersion string) (out Object, err error)
	ConvertFieldLabel(version, kind, label, value string) (string, string, error)
}

// ObjectTyper contains methods for extracting the APIVersion and Kind
// of objects.
type ObjectTyper interface {
	// DataKind returns the group,version,kind of the provided data, or an error
	// if another problem is detected. In many cases this method can be as expensive to
	// invoke as the Decode method.
	DataKind([]byte) (unversioned.GroupVersionKind, error)
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
