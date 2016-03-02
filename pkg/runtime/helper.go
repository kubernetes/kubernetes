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
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/util/errors"
)

type objectTyperToTyper struct {
	typer ObjectTyper
}

func (t objectTyperToTyper) ObjectKind(obj Object) (*unversioned.GroupVersionKind, bool, error) {
	gvk, err := t.typer.ObjectKind(obj)
	if err != nil {
		return nil, false, err
	}
	unversionedType, ok := t.typer.IsUnversioned(obj)
	if !ok {
		// ObjectTyper violates its contract
		return nil, false, fmt.Errorf("typer returned a kind for %v, but then reported it was not in the scheme with IsUnversioned", reflect.TypeOf(obj))
	}
	return &gvk, unversionedType, nil
}

func ObjectTyperToTyper(typer ObjectTyper) Typer {
	return objectTyperToTyper{typer: typer}
}

// fieldPtr puts the address of fieldName, which must be a member of v,
// into dest, which must be an address of a variable to which this field's
// address can be assigned.
func FieldPtr(v reflect.Value, fieldName string, dest interface{}) error {
	field := v.FieldByName(fieldName)
	if !field.IsValid() {
		return fmt.Errorf("couldn't find %v field in %#v", fieldName, v.Interface())
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
func EncodeList(e Encoder, objects []Object, overrides ...unversioned.GroupVersion) error {
	var errs []error
	for i := range objects {
		data, err := Encode(e, objects[i], overrides...)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		objects[i] = &Unknown{RawJSON: data}
	}
	return errors.NewAggregate(errs)
}

func decodeListItem(obj *Unknown, decoders []Decoder) (Object, error) {
	for _, decoder := range decoders {
		obj, err := Decode(decoder, obj.RawJSON)
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
		if err := DecodeInto(decoder, obj.RawJSON, obj); err == nil {
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

func (m MultiObjectTyper) ObjectKind(obj Object) (gvk unversioned.GroupVersionKind, err error) {
	for _, t := range m {
		gvk, err = t.ObjectKind(obj)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiObjectTyper) ObjectKinds(obj Object) (gvks []unversioned.GroupVersionKind, err error) {
	for _, t := range m {
		gvks, err = t.ObjectKinds(obj)
		if err == nil {
			return
		}
	}
	return
}

func (m MultiObjectTyper) Recognizes(gvk unversioned.GroupVersionKind) bool {
	for _, t := range m {
		if t.Recognizes(gvk) {
			return true
		}
	}
	return false
}

func (m MultiObjectTyper) IsUnversioned(obj Object) (bool, bool) {
	for _, t := range m {
		if unversioned, ok := t.IsUnversioned(obj); ok {
			return unversioned, true
		}
	}
	return false, false
}
