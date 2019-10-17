/*
Copyright 2018 The Kubernetes Authors.

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
	"io"
	"sort"
	"strings"

	jsoniter "github.com/json-iterator/go"
)

var (
	readPool  = jsoniter.NewIterator(jsoniter.ConfigCompatibleWithStandardLibrary).Pool()
	writePool = jsoniter.NewStream(jsoniter.ConfigCompatibleWithStandardLibrary, nil, 1024).Pool()
)

func FromJSON(input []byte) (Value, error) {
	return FromJSONFast(input)
}

func ToJSON(val Value) ([]byte, error) {
	buf := bytes.Buffer{}
	stream := writePool.BorrowStream(&buf)
	defer writePool.ReturnStream(stream)
	WriteJSONStream(val, stream)
	b := stream.Buffer()
	err := stream.Flush()
	// Help jsoniter manage its buffers--without this, the next
	// use of the stream is likely to require an allocation. Look
	// at the jsoniter stream code to understand why. They were probably
	// optimizing for folks using the buffer directly.
	stream.SetBuffer(b[:0])
	return buf.Bytes(), err
}

func FromJSONFast(input []byte) (Value, error) {
	iter := readPool.BorrowIterator(input)
	defer readPool.ReturnIterator(iter)
	return ReadJSONIter(iter)
}

func ReadJSONIter(iter *jsoniter.Iterator) (Value, error) {
	v := iter.Read()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil, iter.Error
	}
	return v, nil
}

func WriteJSONStream(v Value, stream *jsoniter.Stream) {
	stream.WriteVal(v)
}

type Value interface{}

func Copy(v Value) Value {
	if IsList(v) {
		l := make([]interface{}, 0, len(ValueList(v)))
		for _, item := range ValueList(v) {
			l = append(l, Copy(item))
		}
		return l
	}
	if IsMap(v) {
		m := make(map[string]interface{}, ValueMap(v).Length())
		ValueMap(v).Iterate(func(key string, item Value) bool {
			m[key] = Copy(item)
			return true
		})
		return m
	}
	// Scalars don't have to be copied
	return v
}

func ToString(v Value) string {
	if v == nil {
		return "null"
	}
	switch {
	case IsFloat(v):
		return fmt.Sprintf("%v", ValueFloat(v))
	case IsInt(v):
		return fmt.Sprintf("%v", ValueInt(v))
	case IsString(v):
		return fmt.Sprintf("%q", ValueString(v))
	case IsBool(v):
		return fmt.Sprintf("%v", ValueBool(v))
	case IsList(v):
		strs := []string{}
		for _, item := range ValueList(v) {
			strs = append(strs, ToString(item))
		}
		return "[" + strings.Join(strs, ",") + "]"
	case IsMap(v):
		strs := []string{}
		ValueMap(v).Iterate(func(k string, v Value) bool {
			strs = append(strs, fmt.Sprintf("%v=%v", k, ToString(v)))
			return true
		})
		return "{" + strings.Join(strs, ";") + "}"
	}
	return fmt.Sprintf("{{undefined(%#v)}}", v)
}

// Equals returns true iff the two values are equal.
func Equals(lhs, rhs Value) bool {
	return !Less(lhs, rhs) && !Less(rhs, lhs)
}

// Less provides a total ordering for Value (so that they can be sorted, even
// if they are of different types).
func Less(lhs, rhs Value) bool {
	if IsFloat(lhs) {
		if !IsFloat(rhs) {
			// Extra: compare floats and ints numerically.
			if IsInt(rhs) {
				return ValueFloat(lhs) < float64(ValueInt(rhs))
			}
			return true
		}
		return ValueFloat(lhs) < ValueFloat(rhs)
	} else if IsFloat(rhs) {
		// Extra: compare floats and ints numerically.
		if IsInt(lhs) {
			return float64(ValueInt(lhs)) < ValueFloat(rhs)
		}
		return false
	}

	if IsInt(lhs) {
		if !IsInt(rhs) {
			return true
		}
		return ValueInt(lhs) < ValueInt(rhs)
	} else if IsInt(rhs) {
		return false
	}

	if IsString(lhs) {
		if !IsString(rhs) {
			return true
		}
		return ValueString(lhs) < ValueString(rhs)
	} else if IsString(rhs) {
		return false
	}

	if IsBool(lhs) {
		if !IsBool(rhs) {
			return true
		}
		if ValueBool(lhs) == ValueBool(rhs) {
			return false
		}
		return ValueBool(lhs) == false
	} else if IsBool(rhs) {
		return false
	}

	if IsList(lhs) {
		if !IsList(rhs) {
			return true
		}
		return ListLess(ValueList(lhs), ValueList(rhs))
	} else if IsList(rhs) {
		return false
	}
	if IsMap(lhs) {
		if !IsMap(rhs) {
			return true
		}
		return MapLess(ValueMap(lhs), ValueMap(rhs))
	} else if IsMap(rhs) {
		return false
	}
	if IsNull(lhs) {
		if !IsNull(rhs) {
			return true
		}
		return false
	} else if IsNull(rhs) {
		return false
	}

	// Invalid Value-- nothing is set.
	return false
}

func ListLess(lhs, rhs []interface{}) bool {
	i := 0
	for {
		if i >= len(lhs) && i >= len(rhs) {
			// Lists are the same length and all items are equal.
			return false
		}
		if i >= len(lhs) {
			// LHS is shorter.
			return true
		}
		if i >= len(rhs) {
			// RHS is shorter.
			return false
		}
		if Less(lhs[i], rhs[i]) {
			// LHS is less; return
			return true
		}
		if Less(rhs[i], lhs[i]) {
			// RHS is less; return
			return false
		}
		// The items are equal; continue.
		i++
	}
}

func MapLess(lhs, rhs Map) bool {
	lorder := []string{}
	lhs.Iterate(func(key string, _ Value) bool {
		lorder = append(lorder, key)
		return true
	})
	sort.Strings(lorder)
	rorder := []string{}
	rhs.Iterate(func(key string, _ Value) bool {
		rorder = append(rorder, key)
		return true
	})
	sort.Strings(rorder)

	i := 0
	for {
		if i >= len(lorder) && i >= len(rorder) {
			// Maps are the same length and all items are equal.
			return false
		}
		if i >= len(lorder) {
			// LHS is shorter.
			return true
		}
		if i >= len(rorder) {
			// RHS is shorter.
			return false
		}
		aname, bname := lorder[i], rorder[i]
		if aname != bname {
			// the map having the field name that sorts lexically less is "less"
			return aname < bname
		}
		aval, _ := lhs.Get(aname)
		bval, _ := rhs.Get(bname)
		if Less(aval, bval) {
			// LHS is less; return
			return true
		}
		if Less(bval, aval) {
			// RHS is less; return
			return false
		}
		// The items are equal; continue.
		i++
	}
}

func IsMap(v Value) bool {
	if _, ok := v.(map[string]interface{}); ok {
		return true
	}
	if _, ok := v.(map[interface{}]interface{}); ok {
		return true
	}
	return false
}

func ValueMap(v Value) Map {
	if v == nil {
		return MapString(nil)
	}
	switch t := v.(type) {
	case map[string]interface{}:
		return MapString(t)
	case map[interface{}]interface{}:
		return MapInterface(t)
	}
	panic(fmt.Errorf("not a map: %#v", v))
}

func IsList(v Value) bool {
	if v == nil {
		return false
	}
	_, ok := v.([]interface{})
	return ok
}

func ValueList(v Value) []interface{} {
	return v.([]interface{})
}

func IsFloat(v Value) bool {
	if v == nil {
		return false
	} else if _, ok := v.(float64); ok {
		return true
	} else if _, ok := v.(float32); ok {
		return true
	}
	return false
}

func ValueFloat(v Value) float64 {
	if f, ok := v.(float32); ok {
		return float64(f)
	}
	return v.(float64)
}

func IsInt(v Value) bool {
	if v == nil {
		return false
	} else if _, ok := v.(int); ok {
		return true
	} else if _, ok := v.(int8); ok {
		return true
	} else if _, ok := v.(int16); ok {
		return true
	} else if _, ok := v.(int32); ok {
		return true
	} else if _, ok := v.(int64); ok {
		return true
	}
	return false
}

func ValueInt(v Value) int64 {
	if i, ok := v.(int); ok {
		return int64(i)
	} else if i, ok := v.(int8); ok {
		return int64(i)
	} else if i, ok := v.(int16); ok {
		return int64(i)
	} else if i, ok := v.(int32); ok {
		return int64(i)
	}
	return v.(int64)
}

func IsString(v Value) bool {
	if v == nil {
		return false
	}
	_, ok := v.(string)
	return ok
}

func ValueString(v Value) string {
	return v.(string)
}

func IsBool(v Value) bool {
	if v == nil {
		return false
	}
	_, ok := v.(bool)
	return ok
}

func ValueBool(v Value) bool {
	return v.(bool)
}

func IsNull(v Value) bool {
	return v == nil
}

type Map interface {
	Length() int
	Get(string) (Value, bool)
	Set(string, Value)
	Delete(string)
	Iterate(func(string, Value) bool)
}

type MapString map[string]interface{}

func (m MapString) Length() int {
	return len(m)
}

func (m MapString) Get(key string) (Value, bool) {
	val, ok := m[key]
	return val, ok
}

func (m MapString) Set(key string, val Value) {
	m[key] = val
}

func (m MapString) Delete(key string) {
	delete(m, key)
}

func (m MapString) Iterate(iter func(string, Value) bool) {
	for key := range m {
		if !iter(key, m[key]) {
			return
		}
	}
}

type MapInterface map[interface{}]interface{}

func (m MapInterface) Length() int {
	return len(m)
}

func (m MapInterface) Get(key string) (Value, bool) {
	val, ok := m[key]
	return val, ok
}

func (m MapInterface) Set(key string, val Value) {
	m[key] = val
}

func (m MapInterface) Delete(key string) {
	delete(m, key)
}

func (m MapInterface) Iterate(iter func(string, Value) bool) {
	for key := range m {
		vk, ok := key.(string)
		if !ok {
			continue
		}
		if !iter(vk, m[key]) {
			return
		}
	}
}
