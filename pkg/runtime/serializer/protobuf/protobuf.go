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

package protobuf

import (
	"bytes"
	"fmt"
	"io"
	"reflect"

	"github.com/gogo/protobuf/proto"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/framer"
)

var (
	// protoEncodingPrefix serves as a magic number for an encoded protobuf message on this serializer. All
	// proto messages serialized by this schema will be preceded by the bytes 0x6b 0x38 0x73, with the fourth
	// byte being reserved for the encoding style. The only encoding style defined is 0x00, which means that
	// the rest of the byte stream is a message of type k8s.io.kubernetes.pkg.runtime.Unknown (proto2).
	//
	// See k8s.io/kubernetes/pkg/runtime/generated.proto for details of the runtime.Unknown message.
	//
	// This encoding scheme is experimental, and is subject to change at any time.
	protoEncodingPrefix = []byte{0x6b, 0x38, 0x73, 0x00}
)

type errNotMarshalable struct {
	t reflect.Type
}

func (e errNotMarshalable) Error() string {
	return fmt.Sprintf("object %v does not implement the protobuf marshalling interface and cannot be encoded to a protobuf message", e.t)
}

func IsNotMarshalable(err error) bool {
	_, ok := err.(errNotMarshalable)
	return err != nil && ok
}

// NewSerializer creates a Protobuf serializer that handles encoding versioned objects into the proper wire form. If a typer
// is passed, the encoded object will have group, version, and kind fields set. If typer is nil, the objects will be written
// as-is (any type info passed with the object will be used).
//
// This encoding scheme is experimental, and is subject to change at any time.
func NewSerializer(creater runtime.ObjectCreater, typer runtime.ObjectTyper, defaultContentType string) *Serializer {
	return &Serializer{
		prefix:      protoEncodingPrefix,
		creater:     creater,
		typer:       typer,
		contentType: defaultContentType,
	}
}

type Serializer struct {
	prefix      []byte
	creater     runtime.ObjectCreater
	typer       runtime.ObjectTyper
	contentType string
}

var _ runtime.Serializer = &Serializer{}

// Decode attempts to convert the provided data into a protobuf message, extract the stored schema kind, apply the provided default
// gvk, and then load that data into an object matching the desired schema kind or the provided into. If into is *runtime.Unknown,
// the raw data will be extracted and no decoding will be performed. If into is not registered with the typer, then the object will
// be straight decoded using normal protobuf unmarshalling (the MarshalTo interface). If into is provided and the original data is
// not fully qualified with kind/version/group, the type of the into will be used to alter the returned gvk. On success or most
// errors, the method will return the calculated schema kind.
func (s *Serializer) Decode(originalData []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	if versioned, ok := into.(*runtime.VersionedObjects); ok {
		into = versioned.Last()
		obj, actual, err := s.Decode(originalData, gvk, into)
		if err != nil {
			return nil, actual, err
		}
		// the last item in versioned becomes into, so if versioned was not originally empty we reset the object
		// array so the first position is the decoded object and the second position is the outermost object.
		// if there were no objects in the versioned list passed to us, only add ourselves.
		if into != nil && into != obj {
			versioned.Objects = []runtime.Object{obj, into}
		} else {
			versioned.Objects = []runtime.Object{obj}
		}
		return versioned, actual, err
	}

	prefixLen := len(s.prefix)
	switch {
	case len(originalData) == 0:
		// TODO: treat like decoding {} from JSON with defaulting
		return nil, nil, fmt.Errorf("empty data")
	case len(originalData) < prefixLen || !bytes.Equal(s.prefix, originalData[:prefixLen]):
		return nil, nil, fmt.Errorf("provided data does not appear to be a protobuf message, expected prefix %v", s.prefix)
	case len(originalData) == prefixLen:
		// TODO: treat like decoding {} from JSON with defaulting
		return nil, nil, fmt.Errorf("empty body")
	}

	data := originalData[prefixLen:]
	unk := runtime.Unknown{}
	if err := unk.Unmarshal(data); err != nil {
		return nil, nil, err
	}

	actual := unk.GroupVersionKind()
	copyKindDefaults(&actual, gvk)

	if intoUnknown, ok := into.(*runtime.Unknown); ok && intoUnknown != nil {
		*intoUnknown = unk
		if len(intoUnknown.ContentType) == 0 {
			intoUnknown.ContentType = s.contentType
		}
		return intoUnknown, &actual, nil
	}

	if into != nil {
		types, _, err := s.typer.ObjectKinds(into)
		switch {
		case runtime.IsNotRegisteredError(err):
			pb, ok := into.(proto.Message)
			if !ok {
				return nil, &actual, errNotMarshalable{reflect.TypeOf(into)}
			}
			if err := proto.Unmarshal(unk.Raw, pb); err != nil {
				return nil, &actual, err
			}
			return into, &actual, nil
		case err != nil:
			return nil, &actual, err
		default:
			copyKindDefaults(&actual, &types[0])
			// if the result of defaulting did not set a version or group, ensure that at least group is set
			// (copyKindDefaults will not assign Group if version is already set). This guarantees that the group
			// of into is set if there is no better information from the caller or object.
			if len(actual.Version) == 0 && len(actual.Group) == 0 {
				actual.Group = types[0].Group
			}
		}
	}

	if len(actual.Kind) == 0 {
		return nil, &actual, runtime.NewMissingKindErr(fmt.Sprintf("%#v", unk.TypeMeta))
	}
	if len(actual.Version) == 0 {
		return nil, &actual, runtime.NewMissingVersionErr(fmt.Sprintf("%#v", unk.TypeMeta))
	}

	return unmarshalToObject(s.typer, s.creater, &actual, into, unk.Raw)
}

// Encode serializes the provided object to the given writer.
func (s *Serializer) Encode(obj runtime.Object, w io.Writer) error {
	var unk runtime.Unknown
	kind := obj.GetObjectKind().GroupVersionKind()
	unk = runtime.Unknown{
		TypeMeta: runtime.TypeMeta{
			Kind:       kind.Kind,
			APIVersion: kind.GroupVersion().String(),
		},
	}

	prefixSize := uint64(len(s.prefix))

	switch t := obj.(type) {
	case bufferedMarshaller:
		// this path performs a single allocation during write but requires the caller to implement
		// the more efficient Size and MarshalTo methods
		encodedSize := uint64(t.Size())
		estimatedSize := prefixSize + estimateUnknownSize(&unk, encodedSize)
		data := make([]byte, estimatedSize)

		i, err := unk.NestedMarshalTo(data[prefixSize:], t, encodedSize)
		if err != nil {
			return err
		}

		copy(data, s.prefix)

		_, err = w.Write(data[:prefixSize+uint64(i)])
		return err

	case proto.Marshaler:
		// this path performs extra allocations
		data, err := t.Marshal()
		if err != nil {
			return err
		}
		unk.Raw = data

		estimatedSize := prefixSize + uint64(unk.Size())
		data = make([]byte, estimatedSize)

		i, err := unk.MarshalTo(data[prefixSize:])
		if err != nil {
			return err
		}

		copy(data, s.prefix)

		_, err = w.Write(data[:prefixSize+uint64(i)])
		return err

	default:
		// TODO: marshal with a different content type and serializer (JSON for third party objects)
		return errNotMarshalable{reflect.TypeOf(obj)}
	}
}

// RecognizesData implements the RecognizingDecoder interface.
func (s *Serializer) RecognizesData(peek io.Reader) (bool, error) {
	prefix := make([]byte, 4)
	n, err := peek.Read(prefix)
	if err != nil {
		if err == io.EOF {
			return false, nil
		}
		return false, err
	}
	if n != 4 {
		return false, nil
	}
	return bytes.Equal(s.prefix, prefix), nil
}

// copyKindDefaults defaults dst to the value in src if dst does not have a value set.
func copyKindDefaults(dst, src *unversioned.GroupVersionKind) {
	if src == nil {
		return
	}
	// apply kind and version defaulting from provided default
	if len(dst.Kind) == 0 {
		dst.Kind = src.Kind
	}
	if len(dst.Version) == 0 && len(src.Version) > 0 {
		dst.Group = src.Group
		dst.Version = src.Version
	}
}

// bufferedMarshaller describes a more efficient marshalling interface that can avoid allocating multiple
// byte buffers by pre-calculating the size of the final buffer needed.
type bufferedMarshaller interface {
	proto.Sizer
	runtime.ProtobufMarshaller
}

// estimateUnknownSize returns the expected bytes consumed by a given runtime.Unknown
// object with a nil RawJSON struct and the expected size of the provided buffer. The
// returned size will not be correct if RawJSOn is set on unk.
func estimateUnknownSize(unk *runtime.Unknown, byteSize uint64) uint64 {
	size := uint64(unk.Size())
	// protobuf uses 1 byte for the tag, a varint for the length of the array (at most 8 bytes - uint64 - here),
	// and the size of the array.
	size += 1 + 8 + byteSize
	return size
}

// NewRawSerializer creates a Protobuf serializer that handles encoding versioned objects into the proper wire form. If typer
// is not nil, the object has the group, version, and kind fields set. This serializer does not provide type information for the
// encoded object, and thus is not self describing (callers must know what type is being described in order to decode).
//
// This encoding scheme is experimental, and is subject to change at any time.
func NewRawSerializer(creater runtime.ObjectCreater, typer runtime.ObjectTyper, defaultContentType string) *RawSerializer {
	return &RawSerializer{
		creater:     creater,
		typer:       typer,
		contentType: defaultContentType,
	}
}

// RawSerializer encodes and decodes objects without adding a runtime.Unknown wrapper (objects are encoded without identifying
// type).
type RawSerializer struct {
	creater     runtime.ObjectCreater
	typer       runtime.ObjectTyper
	contentType string
}

var _ runtime.Serializer = &RawSerializer{}

// Decode attempts to convert the provided data into a protobuf message, extract the stored schema kind, apply the provided default
// gvk, and then load that data into an object matching the desired schema kind or the provided into. If into is *runtime.Unknown,
// the raw data will be extracted and no decoding will be performed. If into is not registered with the typer, then the object will
// be straight decoded using normal protobuf unmarshalling (the MarshalTo interface). If into is provided and the original data is
// not fully qualified with kind/version/group, the type of the into will be used to alter the returned gvk. On success or most
// errors, the method will return the calculated schema kind.
func (s *RawSerializer) Decode(originalData []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	if into == nil {
		return nil, nil, fmt.Errorf("this serializer requires an object to decode into: %#v", s)
	}

	if versioned, ok := into.(*runtime.VersionedObjects); ok {
		into = versioned.Last()
		obj, actual, err := s.Decode(originalData, gvk, into)
		if err != nil {
			return nil, actual, err
		}
		if into != nil && into != obj {
			versioned.Objects = []runtime.Object{obj, into}
		} else {
			versioned.Objects = []runtime.Object{obj}
		}
		return versioned, actual, err
	}

	if len(originalData) == 0 {
		// TODO: treat like decoding {} from JSON with defaulting
		return nil, nil, fmt.Errorf("empty data")
	}
	data := originalData

	actual := &unversioned.GroupVersionKind{}
	copyKindDefaults(actual, gvk)

	if intoUnknown, ok := into.(*runtime.Unknown); ok && intoUnknown != nil {
		intoUnknown.Raw = data
		intoUnknown.ContentEncoding = ""
		intoUnknown.ContentType = s.contentType
		intoUnknown.SetGroupVersionKind(*actual)
		return intoUnknown, actual, nil
	}

	types, _, err := s.typer.ObjectKinds(into)
	switch {
	case runtime.IsNotRegisteredError(err):
		pb, ok := into.(proto.Message)
		if !ok {
			return nil, actual, errNotMarshalable{reflect.TypeOf(into)}
		}
		if err := proto.Unmarshal(data, pb); err != nil {
			return nil, actual, err
		}
		return into, actual, nil
	case err != nil:
		return nil, actual, err
	default:
		copyKindDefaults(actual, &types[0])
		// if the result of defaulting did not set a version or group, ensure that at least group is set
		// (copyKindDefaults will not assign Group if version is already set). This guarantees that the group
		// of into is set if there is no better information from the caller or object.
		if len(actual.Version) == 0 && len(actual.Group) == 0 {
			actual.Group = types[0].Group
		}
	}

	if len(actual.Kind) == 0 {
		return nil, actual, runtime.NewMissingKindErr("<protobuf encoded body - must provide default type>")
	}
	if len(actual.Version) == 0 {
		return nil, actual, runtime.NewMissingVersionErr("<protobuf encoded body - must provide default type>")
	}

	return unmarshalToObject(s.typer, s.creater, actual, into, data)
}

// unmarshalToObject is the common code between decode in the raw and normal serializer.
func unmarshalToObject(typer runtime.ObjectTyper, creater runtime.ObjectCreater, actual *unversioned.GroupVersionKind, into runtime.Object, data []byte) (runtime.Object, *unversioned.GroupVersionKind, error) {
	// use the target if necessary
	obj, err := runtime.UseOrCreateObject(typer, creater, *actual, into)
	if err != nil {
		return nil, actual, err
	}

	pb, ok := obj.(proto.Message)
	if !ok {
		return nil, actual, errNotMarshalable{reflect.TypeOf(obj)}
	}
	if err := proto.Unmarshal(data, pb); err != nil {
		return nil, actual, err
	}
	return obj, actual, nil
}

// Encode serializes the provided object to the given writer. Overrides is ignored.
func (s *RawSerializer) Encode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case bufferedMarshaller:
		// this path performs a single allocation during write but requires the caller to implement
		// the more efficient Size and MarshalTo methods
		encodedSize := uint64(t.Size())
		data := make([]byte, encodedSize)

		n, err := t.MarshalTo(data)
		if err != nil {
			return err
		}
		_, err = w.Write(data[:n])
		return err

	case proto.Marshaler:
		// this path performs extra allocations
		data, err := t.Marshal()
		if err != nil {
			return err
		}
		_, err = w.Write(data)
		return err

	default:
		return errNotMarshalable{reflect.TypeOf(obj)}
	}
}

var LengthDelimitedFramer = lengthDelimitedFramer{}

type lengthDelimitedFramer struct{}

// NewFrameWriter implements stream framing for this serializer
func (lengthDelimitedFramer) NewFrameWriter(w io.Writer) io.Writer {
	return framer.NewLengthDelimitedFrameWriter(w)
}

// NewFrameReader implements stream framing for this serializer
func (lengthDelimitedFramer) NewFrameReader(r io.ReadCloser) io.ReadCloser {
	return framer.NewLengthDelimitedFrameReader(r)
}
