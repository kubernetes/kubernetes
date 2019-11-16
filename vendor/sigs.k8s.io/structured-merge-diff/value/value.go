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

func FromJSONFast(input []byte) (Value, error) {
	iter := readPool.BorrowIterator(input)
	defer readPool.ReturnIterator(iter)
	return ReadJSONIter(iter)
}

func ToJSON(v Value) ([]byte, error) {
	buf := bytes.Buffer{}
	stream := writePool.BorrowStream(&buf)
	defer writePool.ReturnStream(stream)
	WriteJSONStream(v, stream)
	b := stream.Buffer()
	err := stream.Flush()
	// Help jsoniter manage its buffers--without this, the next
	// use of the stream is likely to require an allocation. Look
	// at the jsoniter stream code to understand why. They were probably
	// optimizing for folks using the buffer directly.
	stream.SetBuffer(b[:0])
	return buf.Bytes(), err
}

func ReadJSONIter(iter *jsoniter.Iterator) (Value, error) {
	v := iter.Read()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil, iter.Error
	}
	return ValueInterface{Value: v}, nil
}

func WriteJSONStream(v Value, stream *jsoniter.Stream) {
	stream.WriteVal(v.Interface())
}

// Field is an individual key-value pair.
type Field struct {
	Name  string
	Value Value
}

// FieldList is a list of key-value pairs. Each field is expected to
// have a different name.
type FieldList []Field

// Sort sorts the field list by Name.
func (f FieldList) Sort() {
	if len(f) < 2 {
		return
	}
	if len(f) == 2 {
		if f[1].Name < f[0].Name {
			f[0], f[1] = f[1], f[0]
		}
		return
	}
	sort.SliceStable(f, func(i, j int) bool {
		return f[i].Name < f[j].Name
	})
}

// Less compares two lists lexically.
func (f FieldList) Less(rhs FieldList) bool {
	i := 0
	for {
		if i >= len(f) && i >= len(rhs) {
			// Maps are the same length and all items are equal.
			return false
		}
		if i >= len(f) {
			// F is shorter.
			return true
		}
		if i >= len(rhs) {
			// RHS is shorter.
			return false
		}
		if f[i].Name != rhs[i].Name {
			// the map having the field name that sorts lexically less is "less"
			return f[i].Name < rhs[i].Name
		}
		if Less(f[i].Value, rhs[i].Value) {
			// F is less; return
			return true
		}
		if Less(rhs[i].Value, f[i].Value) {
			// RHS is less; return
			return false
		}
		// The items are equal; continue.
		i++
	}
}

type Value interface {
	IsMap() bool
	IsList() bool
	IsBool() bool
	IsInt() bool
	IsFloat() bool
	IsString() bool
	IsNull() bool

	Map() Map
	List() List
	Bool() bool
	Int() int64
	Float() float64
	String() string

	Interface() interface{}
}

type List interface {
	Interface() []interface{}
	Length() int
	Iterate(func(int, Value))
	At(int) Value
}

type ValueInterface struct {
	Value interface{}
}

func (v ValueInterface) IsMap() bool {
	if _, ok := v.Value.(map[string]interface{}); ok {
		return true
	}
	if _, ok := v.Value.(map[interface{}]interface{}); ok {
		return true
	}
	return false
}

func (v ValueInterface) Map() Map {
	if v.Value == nil {
		return MapString(nil)
	}
	switch t := v.Value.(type) {
	case map[string]interface{}:
		return MapString(t)
	case map[interface{}]interface{}:
		return MapInterface(t)
	}
	panic(fmt.Errorf("not a map: %#v", v))
}

func (v ValueInterface) IsList() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.([]interface{})
	return ok
}

func (v ValueInterface) List() List {
	return ListInterface(v.Value.([]interface{}))
}

func (v ValueInterface) IsFloat() bool {
	if v.Value == nil {
		return false
	} else if _, ok := v.Value.(float64); ok {
		return true
	} else if _, ok := v.Value.(float32); ok {
		return true
	}
	return false
}

func (v ValueInterface) Float() float64 {
	if f, ok := v.Value.(float32); ok {
		return float64(f)
	}
	return v.Value.(float64)
}

func (v ValueInterface) IsInt() bool {
	if v.Value == nil {
		return false
	} else if _, ok := v.Value.(int); ok {
		return true
	} else if _, ok := v.Value.(int8); ok {
		return true
	} else if _, ok := v.Value.(int16); ok {
		return true
	} else if _, ok := v.Value.(int32); ok {
		return true
	} else if _, ok := v.Value.(int64); ok {
		return true
	}
	return false
}

func (v ValueInterface) Int() int64 {
	if i, ok := v.Value.(int); ok {
		return int64(i)
	} else if i, ok := v.Value.(int8); ok {
		return int64(i)
	} else if i, ok := v.Value.(int16); ok {
		return int64(i)
	} else if i, ok := v.Value.(int32); ok {
		return int64(i)
	}
	return v.Value.(int64)
}

func (v ValueInterface) IsString() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(string)
	return ok
}

func (v ValueInterface) String() string {
	return v.Value.(string)
}

func (v ValueInterface) IsBool() bool {
	if v.Value == nil {
		return false
	}
	_, ok := v.Value.(bool)
	return ok
}

func (v ValueInterface) Bool() bool {
	return v.Value.(bool)
}

func (v ValueInterface) IsNull() bool {
	return v.Value == nil
}

func (v ValueInterface) Interface() interface{} {
	return v.Value
}

type ListInterface []interface{}

func (l ListInterface) Interface() []interface{} {
	return l
}

func (l ListInterface) Length() int {
	return len(l)
}

func (l ListInterface) At(i int) Value {
	return ValueInterface{Value: l[i]}
}

func (l ListInterface) Iterate(iter func(int, Value)) {
	for i, val := range l {
		iter(i, ValueInterface{Value: val})
	}
}

func Copy(v Value) Value {
	if v.IsList() {
		l := make([]interface{}, 0, v.List().Length())
		v.List().Iterate(func(_ int, item Value) {
			l = append(l, Copy(ValueInterface{Value: item}))
		})
		return ValueInterface{Value: l}
	}
	if v.IsMap() {
		m := make(map[string]interface{}, v.Map().Length())
		v.Map().Iterate(func(key string, item Value) bool {
			m[key] = Copy(item)
			return true
		})
		return ValueInterface{Value: m}
	}
	// Scalars don't have to be copied
	return v
}

func ToString(v Value) string {
	if v.IsNull() {
		return "null"
	}
	switch {
	case v.IsFloat():
		return fmt.Sprintf("%v", v.Float())
	case v.IsInt():
		return fmt.Sprintf("%v", v.Int())
	case v.IsString():
		return fmt.Sprintf("%q", v.String())
	case v.IsBool():
		return fmt.Sprintf("%v", v.Bool())
	case v.IsList():
		strs := []string{}
		v.List().Iterate(func(_ int, item Value) {
			strs = append(strs, ToString(item))
		})
		return "[" + strings.Join(strs, ",") + "]"
	case v.IsMap():
		strs := []string{}
		v.Map().Iterate(func(k string, v Value) bool {
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
	if lhs.IsFloat() {
		if !rhs.IsFloat() {
			// Extra: compare floats and ints numerically.
			if rhs.IsInt() {
				return lhs.Float() < float64(rhs.Int())
			}
			return true
		}
		return lhs.Float() < rhs.Float()
	} else if rhs.IsFloat() {
		// Extra: compare floats and ints numerically.
		if lhs.IsInt() {
			return float64(lhs.Int()) < rhs.Float()
		}
		return false
	}

	if lhs.IsInt() {
		if !rhs.IsInt() {
			return true
		}
		return lhs.Int() < rhs.Int()
	} else if rhs.IsInt() {
		return false
	}

	if lhs.IsString() {
		if !rhs.IsString() {
			return true
		}
		return lhs.String() < rhs.String()
	} else if rhs.IsString() {
		return false
	}

	if lhs.IsBool() {
		if !rhs.IsBool() {
			return true
		}
		if lhs.Bool() == rhs.Bool() {
			return false
		}
		return lhs.Bool() == false
	} else if rhs.IsBool() {
		return false
	}

	if lhs.IsList() {
		if !rhs.IsList() {
			return true
		}
		return ListLess(lhs.List(), rhs.List())
	} else if rhs.IsList() {
		return false
	}
	if lhs.IsMap() {
		if !rhs.IsMap() {
			return true
		}
		return MapLess(lhs.Map(), rhs.Map())
	} else if rhs.IsMap() {
		return false
	}
	if lhs.IsNull() {
		if !rhs.IsNull() {
			return true
		}
		return false
	} else if rhs.IsNull() {
		return false
	}

	// Invalid Value-- nothing is set.
	return false
}

func ListLess(lhs, rhs List) bool {
	i := 0
	for {
		if i >= lhs.Length() && i >= rhs.Length() {
			// Lists are the same length and all items are equal.
			return false
		}
		if i >= lhs.Length() {
			// LHS is shorter.
			return true
		}
		if i >= rhs.Length() {
			// RHS is shorter.
			return false
		}
		if Less(lhs.At(i), rhs.At(i)) {
			// LHS is less; return
			return true
		}
		if Less(rhs.At(i), lhs.At(i)) {
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
	return ValueInterface{Value: val}, ok
}

func (m MapString) Set(key string, val Value) {
	m[key] = val.Interface()
}

func (m MapString) Delete(key string) {
	delete(m, key)
}

func (m MapString) Iterate(iter func(string, Value) bool) {
	for key, value := range m {
		if !iter(key, ValueInterface{Value: value}) {
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
	return ValueInterface{Value: val}, ok
}

func (m MapInterface) Set(key string, val Value) {
	m[key] = val.Interface()
}

func (m MapInterface) Delete(key string) {
	delete(m, key)
}

func (m MapInterface) Iterate(iter func(string, Value) bool) {
	for key, value := range m {
		vk, ok := key.(string)
		if !ok {
			continue
		}
		if !iter(vk, ValueInterface{Value: value}) {
			return
		}
	}
}
