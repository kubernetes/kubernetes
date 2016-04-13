// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"fmt"
	"reflect"
	"time"

	"google.golang.org/appengine"
	pb "google.golang.org/appengine/internal/datastore"
)

var (
	typeOfBlobKey    = reflect.TypeOf(appengine.BlobKey(""))
	typeOfByteSlice  = reflect.TypeOf([]byte(nil))
	typeOfByteString = reflect.TypeOf(ByteString(nil))
	typeOfGeoPoint   = reflect.TypeOf(appengine.GeoPoint{})
	typeOfTime       = reflect.TypeOf(time.Time{})
)

// typeMismatchReason returns a string explaining why the property p could not
// be stored in an entity field of type v.Type().
func typeMismatchReason(p Property, v reflect.Value) string {
	entityType := "empty"
	switch p.Value.(type) {
	case int64:
		entityType = "int"
	case bool:
		entityType = "bool"
	case string:
		entityType = "string"
	case float64:
		entityType = "float"
	case *Key:
		entityType = "*datastore.Key"
	case time.Time:
		entityType = "time.Time"
	case appengine.BlobKey:
		entityType = "appengine.BlobKey"
	case appengine.GeoPoint:
		entityType = "appengine.GeoPoint"
	case ByteString:
		entityType = "datastore.ByteString"
	case []byte:
		entityType = "[]byte"
	}
	return fmt.Sprintf("type mismatch: %s versus %v", entityType, v.Type())
}

type propertyLoader struct {
	// m holds the number of times a substruct field like "Foo.Bar.Baz" has
	// been seen so far. The map is constructed lazily.
	m map[string]int
}

func (l *propertyLoader) load(codec *structCodec, structValue reflect.Value, p Property, requireSlice bool) string {
	var v reflect.Value
	// Traverse a struct's struct-typed fields.
	for name := p.Name; ; {
		decoder, ok := codec.byName[name]
		if !ok {
			return "no such struct field"
		}
		v = structValue.Field(decoder.index)
		if !v.IsValid() {
			return "no such struct field"
		}
		if !v.CanSet() {
			return "cannot set struct field"
		}

		if decoder.substructCodec == nil {
			break
		}

		if v.Kind() == reflect.Slice {
			if l.m == nil {
				l.m = make(map[string]int)
			}
			index := l.m[p.Name]
			l.m[p.Name] = index + 1
			for v.Len() <= index {
				v.Set(reflect.Append(v, reflect.New(v.Type().Elem()).Elem()))
			}
			structValue = v.Index(index)
			requireSlice = false
		} else {
			structValue = v
		}
		// Strip the "I." from "I.X".
		name = name[len(codec.byIndex[decoder.index].name):]
		codec = decoder.substructCodec
	}

	var slice reflect.Value
	if v.Kind() == reflect.Slice && v.Type().Elem().Kind() != reflect.Uint8 {
		slice = v
		v = reflect.New(v.Type().Elem()).Elem()
	} else if requireSlice {
		return "multiple-valued property requires a slice field type"
	}

	// Convert indexValues to a Go value with a meaning derived from the
	// destination type.
	pValue := p.Value
	if iv, ok := pValue.(indexValue); ok {
		meaning := pb.Property_NO_MEANING
		switch v.Type() {
		case typeOfBlobKey:
			meaning = pb.Property_BLOBKEY
		case typeOfByteSlice:
			meaning = pb.Property_BLOB
		case typeOfByteString:
			meaning = pb.Property_BYTESTRING
		case typeOfGeoPoint:
			meaning = pb.Property_GEORSS_POINT
		case typeOfTime:
			meaning = pb.Property_GD_WHEN
		}
		var err error
		pValue, err = propValue(iv.value, meaning)
		if err != nil {
			return err.Error()
		}
	}

	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		x, ok := pValue.(int64)
		if !ok && pValue != nil {
			return typeMismatchReason(p, v)
		}
		if v.OverflowInt(x) {
			return fmt.Sprintf("value %v overflows struct field of type %v", x, v.Type())
		}
		v.SetInt(x)
	case reflect.Bool:
		x, ok := pValue.(bool)
		if !ok && pValue != nil {
			return typeMismatchReason(p, v)
		}
		v.SetBool(x)
	case reflect.String:
		switch x := pValue.(type) {
		case appengine.BlobKey:
			v.SetString(string(x))
		case ByteString:
			v.SetString(string(x))
		case string:
			v.SetString(x)
		default:
			if pValue != nil {
				return typeMismatchReason(p, v)
			}
		}
	case reflect.Float32, reflect.Float64:
		x, ok := pValue.(float64)
		if !ok && pValue != nil {
			return typeMismatchReason(p, v)
		}
		if v.OverflowFloat(x) {
			return fmt.Sprintf("value %v overflows struct field of type %v", x, v.Type())
		}
		v.SetFloat(x)
	case reflect.Ptr:
		x, ok := pValue.(*Key)
		if !ok && pValue != nil {
			return typeMismatchReason(p, v)
		}
		if _, ok := v.Interface().(*Key); !ok {
			return typeMismatchReason(p, v)
		}
		v.Set(reflect.ValueOf(x))
	case reflect.Struct:
		switch v.Type() {
		case typeOfTime:
			x, ok := pValue.(time.Time)
			if !ok && pValue != nil {
				return typeMismatchReason(p, v)
			}
			v.Set(reflect.ValueOf(x))
		case typeOfGeoPoint:
			x, ok := pValue.(appengine.GeoPoint)
			if !ok && pValue != nil {
				return typeMismatchReason(p, v)
			}
			v.Set(reflect.ValueOf(x))
		default:
			return typeMismatchReason(p, v)
		}
	case reflect.Slice:
		x, ok := pValue.([]byte)
		if !ok {
			if y, yok := pValue.(ByteString); yok {
				x, ok = []byte(y), true
			}
		}
		if !ok && pValue != nil {
			return typeMismatchReason(p, v)
		}
		if v.Type().Elem().Kind() != reflect.Uint8 {
			return typeMismatchReason(p, v)
		}
		v.SetBytes(x)
	default:
		return typeMismatchReason(p, v)
	}
	if slice.IsValid() {
		slice.Set(reflect.Append(slice, v))
	}
	return ""
}

// loadEntity loads an EntityProto into PropertyLoadSaver or struct pointer.
func loadEntity(dst interface{}, src *pb.EntityProto) (err error) {
	props, err := protoToProperties(src)
	if err != nil {
		return err
	}
	if e, ok := dst.(PropertyLoadSaver); ok {
		return e.Load(props)
	}
	return LoadStruct(dst, props)
}

func (s structPLS) Load(props []Property) error {
	var fieldName, reason string
	var l propertyLoader
	for _, p := range props {
		if errStr := l.load(s.codec, s.v, p, p.Multiple); errStr != "" {
			// We don't return early, as we try to load as many properties as possible.
			// It is valid to load an entity into a struct that cannot fully represent it.
			// That case returns an error, but the caller is free to ignore it.
			fieldName, reason = p.Name, errStr
		}
	}
	if reason != "" {
		return &ErrFieldMismatch{
			StructType: s.v.Type(),
			FieldName:  fieldName,
			Reason:     reason,
		}
	}
	return nil
}

func protoToProperties(src *pb.EntityProto) ([]Property, error) {
	props, rawProps := src.Property, src.RawProperty
	out := make([]Property, 0, len(props)+len(rawProps))
	for {
		var (
			x       *pb.Property
			noIndex bool
		)
		if len(props) > 0 {
			x, props = props[0], props[1:]
		} else if len(rawProps) > 0 {
			x, rawProps = rawProps[0], rawProps[1:]
			noIndex = true
		} else {
			break
		}

		var value interface{}
		if x.Meaning != nil && *x.Meaning == pb.Property_INDEX_VALUE {
			value = indexValue{x.Value}
		} else {
			var err error
			value, err = propValue(x.Value, x.GetMeaning())
			if err != nil {
				return nil, err
			}
		}
		out = append(out, Property{
			Name:     x.GetName(),
			Value:    value,
			NoIndex:  noIndex,
			Multiple: x.GetMultiple(),
		})
	}
	return out, nil
}

// propValue returns a Go value that combines the raw PropertyValue with a
// meaning. For example, an Int64Value with GD_WHEN becomes a time.Time.
func propValue(v *pb.PropertyValue, m pb.Property_Meaning) (interface{}, error) {
	switch {
	case v.Int64Value != nil:
		if m == pb.Property_GD_WHEN {
			return fromUnixMicro(*v.Int64Value), nil
		} else {
			return *v.Int64Value, nil
		}
	case v.BooleanValue != nil:
		return *v.BooleanValue, nil
	case v.StringValue != nil:
		if m == pb.Property_BLOB {
			return []byte(*v.StringValue), nil
		} else if m == pb.Property_BLOBKEY {
			return appengine.BlobKey(*v.StringValue), nil
		} else if m == pb.Property_BYTESTRING {
			return ByteString(*v.StringValue), nil
		} else {
			return *v.StringValue, nil
		}
	case v.DoubleValue != nil:
		return *v.DoubleValue, nil
	case v.Referencevalue != nil:
		key, err := referenceValueToKey(v.Referencevalue)
		if err != nil {
			return nil, err
		}
		return key, nil
	case v.Pointvalue != nil:
		// NOTE: Strangely, latitude maps to X, longitude to Y.
		return appengine.GeoPoint{Lat: v.Pointvalue.GetX(), Lng: v.Pointvalue.GetY()}, nil
	}
	return nil, nil
}

// indexValue is a Property value that is created when entities are loaded from
// an index, such as from a projection query.
//
// Such Property values do not contain all of the metadata required to be
// faithfully represented as a Go value, and are instead represented as an
// opaque indexValue. Load the properties into a concrete struct type (e.g. by
// passing a struct pointer to Iterator.Next) to reconstruct actual Go values
// of type int, string, time.Time, etc.
type indexValue struct {
	value *pb.PropertyValue
}
