// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datastore

import (
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/golang/protobuf/proto"
	pb "google.golang.org/cloud/internal/datastore"
)

// saveEntity saves an EntityProto into a PropertyLoadSaver or struct pointer.
func saveEntity(key *Key, src interface{}) (*pb.Entity, error) {
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
	return propertiesToProto(key, props)
}

func saveStructProperty(props *[]Property, name string, noIndex, multiple bool, v reflect.Value) error {
	p := Property{
		Name:     name,
		NoIndex:  noIndex,
		Multiple: multiple,
	}

	switch x := v.Interface().(type) {
	case *Key, time.Time:
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

func propertiesToProto(key *Key, props []Property) (*pb.Entity, error) {
	e := &pb.Entity{
		Key: keyToProto(key),
	}
	indexedProps := 0
	prevMultiple := make(map[string]*pb.Property)
	for _, p := range props {
		val, err := interfaceToProto(p.Value)
		if err != "" {
			return nil, fmt.Errorf("datastore: %s for a Property with Name %q", err, p.Name)
		}
		if !p.NoIndex {
			rVal := reflect.ValueOf(p.Value)
			if rVal.Kind() == reflect.Slice && rVal.Type().Elem().Kind() != reflect.Uint8 {
				indexedProps += rVal.Len()
			} else {
				indexedProps++
			}
		}
		if indexedProps > maxIndexedProperties {
			return nil, errors.New("datastore: too many indexed properties")
		}
		switch v := p.Value.(type) {
		case string:
			if len(v) > 1500 && !p.NoIndex {
				return nil, fmt.Errorf("datastore: Property with Name %q is too long to index", p.Name)
			}
		case []byte:
			if len(v) > 1500 && !p.NoIndex {
				return nil, fmt.Errorf("datastore: Property with Name %q is too long to index", p.Name)
			}
		}
		val.Indexed = proto.Bool(!p.NoIndex)
		if p.Multiple {
			x, ok := prevMultiple[p.Name]
			if !ok {
				x = &pb.Property{
					Name:  proto.String(p.Name),
					Value: &pb.Value{},
				}
				prevMultiple[p.Name] = x
				e.Property = append(e.Property, x)
			}
			x.Value.ListValue = append(x.Value.ListValue, val)
		} else {
			e.Property = append(e.Property, &pb.Property{
				Name:  proto.String(p.Name),
				Value: val,
			})
		}
	}
	return e, nil
}

func interfaceToProto(iv interface{}) (p *pb.Value, errStr string) {
	val := new(pb.Value)
	switch v := iv.(type) {
	case int:
		val.IntegerValue = proto.Int64(int64(v))
	case int32:
		val.IntegerValue = proto.Int64(int64(v))
	case int64:
		val.IntegerValue = proto.Int64(v)
	case bool:
		val.BooleanValue = proto.Bool(v)
	case string:
		val.StringValue = proto.String(v)
	case float32:
		val.DoubleValue = proto.Float64(float64(v))
	case float64:
		val.DoubleValue = proto.Float64(v)
	case *Key:
		if v != nil {
			val.KeyValue = keyToProto(v)
		}
	case time.Time:
		if v.Before(minTime) || v.After(maxTime) {
			return nil, fmt.Sprintf("time value out of range")
		}
		val.TimestampMicrosecondsValue = proto.Int64(toUnixMicro(v))
	case []byte:
		val.BlobValue = v
	default:
		if iv != nil {
			return nil, fmt.Sprintf("invalid Value type %t", iv)
		}
	}
	// TODO(jbd): Support ListValue and EntityValue.
	// TODO(jbd): Support types whose underlying type is one of the types above.
	return val, ""
}
