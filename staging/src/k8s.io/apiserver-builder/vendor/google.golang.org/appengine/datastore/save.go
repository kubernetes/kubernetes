// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	pb "google.golang.org/appengine/internal/datastore"
)

func toUnixMicro(t time.Time) int64 {
	// We cannot use t.UnixNano() / 1e3 because we want to handle times more than
	// 2^63 nanoseconds (which is about 292 years) away from 1970, and those cannot
	// be represented in the numerator of a single int64 divide.
	return t.Unix()*1e6 + int64(t.Nanosecond()/1e3)
}

func fromUnixMicro(t int64) time.Time {
	return time.Unix(t/1e6, (t%1e6)*1e3).UTC()
}

var (
	minTime = time.Unix(int64(math.MinInt64)/1e6, (int64(math.MinInt64)%1e6)*1e3)
	maxTime = time.Unix(int64(math.MaxInt64)/1e6, (int64(math.MaxInt64)%1e6)*1e3)
)

// valueToProto converts a named value to a newly allocated Property.
// The returned error string is empty on success.
func valueToProto(defaultAppID, name string, v reflect.Value, multiple bool) (p *pb.Property, errStr string) {
	var (
		pv          pb.PropertyValue
		unsupported bool
	)
	switch v.Kind() {
	case reflect.Invalid:
		// No-op.
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		pv.Int64Value = proto.Int64(v.Int())
	case reflect.Bool:
		pv.BooleanValue = proto.Bool(v.Bool())
	case reflect.String:
		pv.StringValue = proto.String(v.String())
	case reflect.Float32, reflect.Float64:
		pv.DoubleValue = proto.Float64(v.Float())
	case reflect.Ptr:
		if k, ok := v.Interface().(*Key); ok {
			if k != nil {
				pv.Referencevalue = keyToReferenceValue(defaultAppID, k)
			}
		} else {
			unsupported = true
		}
	case reflect.Struct:
		switch t := v.Interface().(type) {
		case time.Time:
			if t.Before(minTime) || t.After(maxTime) {
				return nil, "time value out of range"
			}
			pv.Int64Value = proto.Int64(toUnixMicro(t))
		case appengine.GeoPoint:
			if !t.Valid() {
				return nil, "invalid GeoPoint value"
			}
			// NOTE: Strangely, latitude maps to X, longitude to Y.
			pv.Pointvalue = &pb.PropertyValue_PointValue{X: &t.Lat, Y: &t.Lng}
		default:
			unsupported = true
		}
	case reflect.Slice:
		if b, ok := v.Interface().([]byte); ok {
			pv.StringValue = proto.String(string(b))
		} else {
			// nvToProto should already catch slice values.
			// If we get here, we have a slice of slice values.
			unsupported = true
		}
	default:
		unsupported = true
	}
	if unsupported {
		return nil, "unsupported datastore value type: " + v.Type().String()
	}
	p = &pb.Property{
		Name:     proto.String(name),
		Value:    &pv,
		Multiple: proto.Bool(multiple),
	}
	if v.IsValid() {
		switch v.Interface().(type) {
		case []byte:
			p.Meaning = pb.Property_BLOB.Enum()
		case ByteString:
			p.Meaning = pb.Property_BYTESTRING.Enum()
		case appengine.BlobKey:
			p.Meaning = pb.Property_BLOBKEY.Enum()
		case time.Time:
			p.Meaning = pb.Property_GD_WHEN.Enum()
		case appengine.GeoPoint:
			p.Meaning = pb.Property_GEORSS_POINT.Enum()
		}
	}
	return p, ""
}

// saveEntity saves an EntityProto into a PropertyLoadSaver or struct pointer.
func saveEntity(defaultAppID string, key *Key, src interface{}) (*pb.EntityProto, error) {
	var err error
	var props []Property
	if e, ok := src.(PropertyLoadSaver); ok {
		props, err = e.Save()
	} else {
		props, err = SaveStruct(src)
	}
	if err != nil {
		return nil, err
	}
	return propertiesToProto(defaultAppID, key, props)
}

func saveStructProperty(props *[]Property, name string, noIndex, multiple bool, v reflect.Value) error {
	p := Property{
		Name:     name,
		NoIndex:  noIndex,
		Multiple: multiple,
	}
	switch x := v.Interface().(type) {
	case *Key:
		p.Value = x
	case time.Time:
		p.Value = x
	case appengine.BlobKey:
		p.Value = x
	case appengine.GeoPoint:
		p.Value = x
	case ByteString:
		p.Value = x
	default:
		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			p.Value = v.Int()
		case reflect.Bool:
			p.Value = v.Bool()
		case reflect.String:
			p.Value = v.String()
		case reflect.Float32, reflect.Float64:
			p.Value = v.Float()
		case reflect.Slice:
			if v.Type().Elem().Kind() == reflect.Uint8 {
				p.NoIndex = true
				p.Value = v.Bytes()
			}
		case reflect.Struct:
			if !v.CanAddr() {
				return fmt.Errorf("datastore: unsupported struct field: value is unaddressable")
			}
			sub, err := newStructPLS(v.Addr().Interface())
			if err != nil {
				return fmt.Errorf("datastore: unsupported struct field: %v", err)
			}
			return sub.(structPLS).save(props, name, noIndex, multiple)
		}
	}
	if p.Value == nil {
		return fmt.Errorf("datastore: unsupported struct field type: %v", v.Type())
	}
	*props = append(*props, p)
	return nil
}

func (s structPLS) Save() ([]Property, error) {
	var props []Property
	if err := s.save(&props, "", false, false); err != nil {
		return nil, err
	}
	return props, nil
}

func (s structPLS) save(props *[]Property, prefix string, noIndex, multiple bool) error {
	for i, t := range s.codec.byIndex {
		if t.name == "-" {
			continue
		}
		name := t.name
		if prefix != "" {
			name = prefix + name
		}
		v := s.v.Field(i)
		if !v.IsValid() || !v.CanSet() {
			continue
		}
		noIndex1 := noIndex || t.noIndex
		// For slice fields that aren't []byte, save each element.
		if v.Kind() == reflect.Slice && v.Type().Elem().Kind() != reflect.Uint8 {
			for j := 0; j < v.Len(); j++ {
				if err := saveStructProperty(props, name, noIndex1, true, v.Index(j)); err != nil {
					return err
				}
			}
			continue
		}
		// Otherwise, save the field itself.
		if err := saveStructProperty(props, name, noIndex1, multiple, v); err != nil {
			return err
		}
	}
	return nil
}

func propertiesToProto(defaultAppID string, key *Key, props []Property) (*pb.EntityProto, error) {
	e := &pb.EntityProto{
		Key: keyToProto(defaultAppID, key),
	}
	if key.parent == nil {
		e.EntityGroup = &pb.Path{}
	} else {
		e.EntityGroup = keyToProto(defaultAppID, key.root()).Path
	}
	prevMultiple := make(map[string]bool)

	for _, p := range props {
		if pm, ok := prevMultiple[p.Name]; ok {
			if !pm || !p.Multiple {
				return nil, fmt.Errorf("datastore: multiple Properties with Name %q, but Multiple is false", p.Name)
			}
		} else {
			prevMultiple[p.Name] = p.Multiple
		}

		x := &pb.Property{
			Name:     proto.String(p.Name),
			Value:    new(pb.PropertyValue),
			Multiple: proto.Bool(p.Multiple),
		}
		switch v := p.Value.(type) {
		case int64:
			x.Value.Int64Value = proto.Int64(v)
		case bool:
			x.Value.BooleanValue = proto.Bool(v)
		case string:
			x.Value.StringValue = proto.String(v)
			if p.NoIndex {
				x.Meaning = pb.Property_TEXT.Enum()
			}
		case float64:
			x.Value.DoubleValue = proto.Float64(v)
		case *Key:
			if v != nil {
				x.Value.Referencevalue = keyToReferenceValue(defaultAppID, v)
			}
		case time.Time:
			if v.Before(minTime) || v.After(maxTime) {
				return nil, fmt.Errorf("datastore: time value out of range")
			}
			x.Value.Int64Value = proto.Int64(toUnixMicro(v))
			x.Meaning = pb.Property_GD_WHEN.Enum()
		case appengine.BlobKey:
			x.Value.StringValue = proto.String(string(v))
			x.Meaning = pb.Property_BLOBKEY.Enum()
		case appengine.GeoPoint:
			if !v.Valid() {
				return nil, fmt.Errorf("datastore: invalid GeoPoint value")
			}
			// NOTE: Strangely, latitude maps to X, longitude to Y.
			x.Value.Pointvalue = &pb.PropertyValue_PointValue{X: &v.Lat, Y: &v.Lng}
			x.Meaning = pb.Property_GEORSS_POINT.Enum()
		case []byte:
			x.Value.StringValue = proto.String(string(v))
			x.Meaning = pb.Property_BLOB.Enum()
			if !p.NoIndex {
				return nil, fmt.Errorf("datastore: cannot index a []byte valued Property with Name %q", p.Name)
			}
		case ByteString:
			x.Value.StringValue = proto.String(string(v))
			x.Meaning = pb.Property_BYTESTRING.Enum()
		default:
			if p.Value != nil {
				return nil, fmt.Errorf("datastore: invalid Value type for a Property with Name %q", p.Name)
			}
		}

		if p.NoIndex {
			e.RawProperty = append(e.RawProperty, x)
		} else {
			e.Property = append(e.Property, x)
			if len(e.Property) > maxIndexedProperties {
				return nil, errors.New("datastore: too many indexed properties")
			}
		}
	}
	return e, nil
}
