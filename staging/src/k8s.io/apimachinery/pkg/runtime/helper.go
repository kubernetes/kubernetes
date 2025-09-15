/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io"
	"reflect"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/errors"
)

// unsafeObjectConvertor implements ObjectConvertor using the unsafe conversion path.
type unsafeObjectConvertor struct {
	*Scheme
}

var _ ObjectConvertor = unsafeObjectConvertor{}

// ConvertToVersion converts in to the provided outVersion without copying the input first, which
// is only safe if the output object is not mutated or reused.
func (c unsafeObjectConvertor) ConvertToVersion(in Object, outVersion GroupVersioner) (Object, error) {
	return c.Scheme.UnsafeConvertToVersion(in, outVersion)
}

// UnsafeObjectConvertor performs object conversion without copying the object structure,
// for use when the converted object will not be reused or mutated. Primarily for use within
// versioned codecs, which use the external object for serialization but do not return it.
func UnsafeObjectConvertor(scheme *Scheme) ObjectConvertor {
	return unsafeObjectConvertor{scheme}
}

// SetField puts the value of src, into fieldName, which must be a member of v.
// The value of src must be assignable to the field.
func SetField(src interface{}, v reflect.Value, fieldName string) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %T", fieldName, v.Interface())
	}
	srcValue := reflect.ValueOf(src)
	if srcValue.Type().AssignableTo(field.Type()) {
		field.Set(srcValue)
		return nil
	}
	if srcValue.Type().ConvertibleTo(field.Type()) {
		field.Set(srcValue.Convert(field.Type()))
		return nil
	}
	return fmt.Errorf("couldn't assign/convert %v to %v", srcValue.Type(), field.Type())
}

// Field puts the value of fieldName, which must be a member of v, into dest,
// which must be a variable to which this field's value can be assigned.
func Field(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %T", fieldName, v.Interface())
	}
	destValue, err := conversion.EnforcePtr(dest)
	if err != nil {
		return err
	}
	if field.Type().AssignableTo(destValue.Type()) {
		destValue.Set(field)
		return nil
	}
	if field.Type().ConvertibleTo(destValue.Type()) {
		destValue.Set(field.Convert(destValue.Type()))
		return nil
	}
	return fmt.Errorf("couldn't assign/convert %v to %v", field.Type(), destValue.Type())
}

// FieldPtr puts the address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's
// address can be assigned.
func FieldPtr(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %T", fieldName, v.Interface())
	}
	v, err := conversion.EnforcePtr(dest)
	if err != nil {
		return err
	}
	field = field.Addr()
	if field.Type().AssignableTo(v.Type()) {
		v.Set(field)
		return nil
	}
	if field.Type().ConvertibleTo(v.Type()) {
		v.Set(field.Convert(v.Type()))
		return nil
	}
	return fmt.Errorf("couldn't assign/convert %v to %v", field.Type(), v.Type())
}

// EncodeList ensures that each object in an array is converted to a Unknown{} in serialized form.
// TODO: accept a content type.
func EncodeList(e Encoder, objects []Object) error {
	var errs []error
	for i := range objects {
		data, err := Encode(e, objects[i])
		if err != nil {
			errs = append(errs, err)
			continue
		}
		// TODO: Set ContentEncoding and ContentType.
		objects[i] = &Unknown{Raw: data}
	}
	return errors.NewAggregate(errs)
}

func decodeListItem(obj *Unknown, decoders []Decoder) (Object, error) {
	for _, decoder := range decoders {
		// TODO: Decode based on ContentType.
		obj, err := Decode(decoder, obj.Raw)
		if err != nil {
			if IsNotRegisteredError(err) {
				continue
			}
			return nil, err
		}
		return obj, nil
	}
	// could not decode, so leave the object as Unknown, but give the decoders the
	// chance to set Unknown.TypeMeta if it is available.
	for _, decoder := range decoders {
		if err := DecodeInto(decoder, obj.Raw, obj); err == nil {
			return obj, nil
		}
	}
	return obj, nil
}

// DecodeList alters the list in place, attempting to decode any objects found in
// the list that have the Unknown type. Any errors that occur are returned
// after the entire list is processed. Decoders are tried in order.
func DecodeList(objects []Object, decoders ...Decoder) []error {
	errs := []error(nil)
	for i, obj := range objects {
		switch t := obj.(type) {
		case *Unknown:
			decoded, err := decodeListItem(t, decoders)
			if err != nil {
				errs = append(errs, err)
				break
			}
			objects[i] = decoded
		}
	}
	return errs
}

// MultiObjectTyper returns the types of objects across multiple schemes in order.
type MultiObjectTyper []ObjectTyper

var _ ObjectTyper = MultiObjectTyper{}

func (m MultiObjectTyper) ObjectKinds(obj Object) (gvks []schema.GroupVersionKind, unversionedType bool, err error) {
	for _, t := range m {
		gvks, unversionedType, err = t.ObjectKinds(obj)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	for _, t := range m {
		if t.Recognizes(gvk) {
			return true
		}
	}
	return false
}

// SetZeroValue would set the object of objPtr to zero value of its type.
func SetZeroValue(objPtr Object) error {
	v, err := conversion.EnforcePtr(objPtr)
	if err != nil {
		return err
	}
	v.Set(reflect.Zero(v.Type()))
	return nil
}

// DefaultFramer is valid for any stream that can read objects serially without
// any separation in the stream.
var DefaultFramer = defaultFramer{}

type defaultFramer struct{}

func (defaultFramer) NewFrameReader(r io.ReadCloser) io.ReadCloser { return r }
func (defaultFramer) NewFrameWriter(w io.Writer) io.Writer         { return w }

// WithVersionEncoder serializes an object and ensures the GVK is set.
type WithVersionEncoder struct {
	Version GroupVersioner
	Encoder
	ObjectTyper
}

// Encode does not do conversion. It sets the gvk during serialization.
func (e WithVersionEncoder) Encode(obj Object, stream io.Writer) error {
	gvks, _, err := e.ObjectTyper.ObjectKinds(obj)
	if err != nil {
		if IsNotRegisteredError(err) {
			return e.Encoder.Encode(obj, stream)
		}
		return err
	}
	kind := obj.GetObjectKind()
	oldGVK := kind.GroupVersionKind()
	gvk := gvks[0]
	if e.Version != nil {
		preferredGVK, ok := e.Version.KindForGroupVersionKinds(gvks)
		if ok {
			gvk = preferredGVK
		}
	}

	// The gvk only needs to be set if not already as desired.
	if gvk != oldGVK {
		kind.SetGroupVersionKind(gvk)
		defer kind.SetGroupVersionKind(oldGVK)
	}

	return e.Encoder.Encode(obj, stream)
}

// WithoutVersionDecoder clears the group version kind of a deserialized object.
type WithoutVersionDecoder struct {
	Decoder
}

// Decode does not do conversion. It removes the gvk during deserialization.
func (d WithoutVersionDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into Object) (Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := d.Decoder.Decode(data, defaults, into)
	if obj != nil {
		kind := obj.GetObjectKind()
		// clearing the gvk is just a convention of a codec
		kind.SetGroupVersionKind(schema.GroupVersionKind{})
	}
	return obj, gvk, err
}

type encoderWithAllocator struct {
	encoder      EncoderWithAllocator
	memAllocator MemoryAllocator
}

// NewEncoderWithAllocator returns a new encoder
func NewEncoderWithAllocator(e EncoderWithAllocator, a MemoryAllocator) Encoder {
	return &encoderWithAllocator{
		encoder:      e,
		memAllocator: a,
	}
}

// Encode writes the provided object to the nested writer
func (e *encoderWithAllocator) Encode(obj Object, w io.Writer) error {
	return e.encoder.EncodeWithAllocator(obj, w, e.memAllocator)
}

// Identifier returns identifier of this encoder.
func (e *encoderWithAllocator) Identifier() Identifier {
	return e.encoder.Identifier()
}

type nondeterministicEncoderToEncoderAdapter struct {
	NondeterministicEncoder
}

func (e nondeterministicEncoderToEncoderAdapter) Encode(obj Object, w io.Writer) error {
	return e.EncodeNondeterministic(obj, w)
}

// UseNondeterministicEncoding returns an Encoder that encodes objects using the provided Encoder's
// EncodeNondeterministic method if it implements NondeterministicEncoder, otherwise it returns the
// provided Encoder as-is.
func UseNondeterministicEncoding(encoder Encoder) Encoder {
	if nondeterministic, ok := encoder.(NondeterministicEncoder); ok {
		return nondeterministicEncoderToEncoderAdapter{nondeterministic}
	}
	return encoder
}
