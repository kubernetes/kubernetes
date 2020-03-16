/*
Copyright 2019 The Kubernetes Authors.

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

package value

import (
	"bytes"
	"fmt"

	jsoniter "github.com/json-iterator/go"
)

var (
	readPool  = jsoniter.NewIterator(jsoniter.ConfigCompatibleWithStandardLibrary).Pool()
	writePool = jsoniter.NewStream(jsoniter.ConfigCompatibleWithStandardLibrary, nil, 1024).Pool()
)

// FromJSONFast is a helper function for reading a JSON document
func FromJSONFast(input []byte) (Value, error) {
	iter := readPool.BorrowIterator(input)
	defer readPool.ReturnIterator(iter)
	return ReadJSONIter(iter)
}

func ReadJSONIter(iter *jsoniter.Iterator) (Value, error) {
	next := iter.WhatIsNext()
	switch next {
	case jsoniter.InvalidValue:
		iter.ReportError("reading an object", "got invalid token")
		return Value{}, iter.Error
	case jsoniter.StringValue:
		str := String(iter.ReadString())
		return Value{StringValue: &str}, nil
	case jsoniter.NumberValue:
		number := iter.ReadNumber()
		isFloat := false
		for _, c := range number {
			if c == 'e' || c == 'E' || c == '.' {
				isFloat = true
				break
			}
		}
		if isFloat {
			f, err := number.Float64()
			if err != nil {
				iter.ReportError("parsing as float", err.Error())
				return Value{}, err
			}
			return Value{FloatValue: (*Float)(&f)}, nil
		}
		i, err := number.Int64()
		if err != nil {
			iter.ReportError("parsing as float", err.Error())
			return Value{}, err
		}
		return Value{IntValue: (*Int)(&i)}, nil
	case jsoniter.NilValue:
		iter.ReadNil()
		return Value{Null: true}, nil
	case jsoniter.BoolValue:
		b := Boolean(iter.ReadBool())
		return Value{BooleanValue: &b}, nil
	case jsoniter.ArrayValue:
		list := &List{}
		iter.ReadArrayCB(func(iter *jsoniter.Iterator) bool {
			v, err := ReadJSONIter(iter)
			if err != nil {
				iter.Error = err
				return false
			}
			list.Items = append(list.Items, v)
			return true
		})
		return Value{ListValue: list}, iter.Error
	case jsoniter.ObjectValue:
		m := &Map{}
		iter.ReadObjectCB(func(iter *jsoniter.Iterator, key string) bool {
			v, err := ReadJSONIter(iter)
			if err != nil {
				iter.Error = err
				return false
			}
			m.Items = append(m.Items, Field{Name: key, Value: v})
			return true
		})
		return Value{MapValue: m}, iter.Error
	default:
		return Value{}, fmt.Errorf("unexpected object type %v", next)
	}
}

// ToJSONFast is a helper function for producing a JSon document.
func (v *Value) ToJSONFast() ([]byte, error) {
	buf := bytes.Buffer{}
	stream := writePool.BorrowStream(&buf)
	defer writePool.ReturnStream(stream)
	v.WriteJSONStream(stream)
	err := stream.Flush()
	return buf.Bytes(), err
}

func (v *Value) WriteJSONStream(stream *jsoniter.Stream) {
	switch {
	case v.Null:
		stream.WriteNil()
	case v.FloatValue != nil:
		stream.WriteFloat64(float64(*v.FloatValue))
	case v.IntValue != nil:
		stream.WriteInt64(int64(*v.IntValue))
	case v.BooleanValue != nil:
		stream.WriteBool(bool(*v.BooleanValue))
	case v.StringValue != nil:
		stream.WriteString(string(*v.StringValue))
	case v.ListValue != nil:
		stream.WriteArrayStart()
		for i := range v.ListValue.Items {
			if i > 0 {
				stream.WriteMore()
			}
			v.ListValue.Items[i].WriteJSONStream(stream)
		}
		stream.WriteArrayEnd()
	case v.MapValue != nil:
		stream.WriteObjectStart()
		for i := range v.MapValue.Items {
			if i > 0 {
				stream.WriteMore()
			}
			stream.WriteObjectField(v.MapValue.Items[i].Name)
			v.MapValue.Items[i].Value.WriteJSONStream(stream)
		}
		stream.WriteObjectEnd()
	default:
		stream.Write([]byte("invalid_value"))
	}
}
