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
	"strings"

	jsoniter "github.com/json-iterator/go"
)

var (
	readPool  = jsoniter.NewIterator(jsoniter.ConfigCompatibleWithStandardLibrary).Pool()
	writePool = jsoniter.NewStream(jsoniter.ConfigCompatibleWithStandardLibrary, nil, 1024).Pool()
)

// FromJSON is a helper function for reading a JSON document.
func FromJSON(input []byte) (Value, error) {
	return FromJSONFast(input)
}

// FromJSONFast is a helper function for reading a JSON document.
func FromJSONFast(input []byte) (Value, error) {
	iter := readPool.BorrowIterator(input)
	defer readPool.ReturnIterator(iter)
	return ReadJSONIter(iter)
}

// ToJSON is a helper function for producing a JSon document.
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

// ReadJSONIter reads a Value from a JSON iterator.
func ReadJSONIter(iter *jsoniter.Iterator) (Value, error) {
	v := iter.Read()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil, iter.Error
	}
	return NewValueInterface(v), nil
}

// WriteJSONStream writes a value into a JSON stream.
func WriteJSONStream(v Value, stream Stream) {
	switch {
	case v.IsNull():
		stream.WriteNil()
	case v.IsFloat():
		stream.WriteFloat64(v.Float())
	case v.IsInt():
		stream.WriteInt64(v.Int())
	case v.IsBool():
		stream.WriteBool(v.Bool())
	case v.IsString():
		stream.WriteString(v.String())
	case v.IsList():
		stream.WriteArrayStart()
		for i := 0; i < v.List().Length(); i++ {
			if i > 0 {
				stream.WriteRaw(",")
			}
			WriteJSONStream(v.List().At(i), stream)
		}
		stream.WriteArrayEnd()
	case v.IsMap():
		stream.WriteObjectStart()
		i := 0
		v.Map().Iterate(func(k string, v Value) bool {
			if i > 0 {
				stream.WriteRaw(",")
			}
			stream.WriteObjectField(k)
			WriteJSONStream(v, stream)
			i++
			return true
		})
		stream.WriteObjectEnd()
	default:
		stream.Write([]byte("invalid_value"))
	}
}

// A Value corresponds to an 'atom' in the schema.
type Value interface {
	// IsMap returns true if the Value can be converted to a Map,
	// false otherwise.
	IsMap() bool
	// IsList returns true if the Value can be converted to a List,
	// false otherwise.
	IsList() bool
	// IsBool returns true if the Value can be converted to a bool,
	// false otherwise.
	IsBool() bool
	// IsInt returns true if the Value can be converted to a int64,
	// false otherwise.
	IsInt() bool
	// IsFloat returns true if the Value can be converted to a
	// float64, false otherwise.
	IsFloat() bool
	// IsString returns true if the Value can be converted to a
	// string, false otherwise.
	IsString() bool
	// IsMap returns true if the Value is null, false otherwise.
	IsNull() bool

	// Map converts the Value into a Map (or panic if the type
	// doesn't allow it).
	Map() Map
	// List converts the Value into a List (or panic if the type
	// doesn't allow it).
	List() List
	// Bool converts the Value into a bool (or panic if the type
	// doesn't allow it).
	Bool() bool
	// Int converts the Value into an int64 (or panic if the type
	// doesn't allow it).
	Int() int64
	// Float converts the Value into a float64 (or panic if the type
	// doesn't allow it).
	Float() float64
	// String converts the Value into a string (or panic if the type
	// doesn't allow it).
	String() string

	// Returns a value of this type that is no longer needed. The
	// value shouldn't be used after this call.
	Recycle()

	// Converts the Value into an interface{}.
	Interface() interface{}
}

// Equals returns true iff the two values are equal.
func Equals(lhs, rhs Value) bool {
	if lhs.IsFloat() || rhs.IsFloat() {
		var lf float64
		if lhs.IsFloat() {
			lf = lhs.Float()
		} else if lhs.IsInt() {
			lf = float64(lhs.Int())
		} else {
			return false
		}
		var rf float64
		if rhs.IsFloat() {
			rf = rhs.Float()
		} else if rhs.IsInt() {
			rf = float64(rhs.Int())
		} else {
			return false
		}
		return lf == rf
	}
	if lhs.IsInt() {
		if rhs.IsInt() {
			return lhs.Int() == rhs.Int()
		}
		return false
	}
	// if rc, ok := lhs.(reflectConverted); ok {
	// 	if rcrhs, ok := rhs.(reflectConverted); ok {
	// 		lval := rc.Value
	// 		if lval.Kind() == reflect.Ptr {
	// 			lval = lval.Elem()
	// 		}
	// 		rval := rcrhs.Value
	// 		if rval.Kind() == reflect.Ptr {
	// 			rval = rval.Elem()
	// 		}
	// 		return rc.Converter.Equal(lval, rval)
	// 	}
	// }
	if lhs.IsString() {
		if rhs.IsString() {
			return lhs.String() == rhs.String()
		}
		return false
	}
	if lhs.IsBool() {
		if rhs.IsBool() {
			return lhs.Bool() == rhs.Bool()
		}
		return false
	}
	if lhs.IsList() {
		if rhs.IsList() {
			return ListEquals(lhs.List(), rhs.List())
		}
		return false
	}
	if lhs.IsMap() {
		if rhs.IsMap() {
			return lhs.Map().Equals(rhs.Map())
		}
		return false
	}
	if lhs.IsNull() {
		if rhs.IsNull() {
			return true
		}
		return false
	}
	// No field is set, on either objects.
	return true
}

// String returns a human-readable representation of the value.
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
		for i := 0; i < v.List().Length(); i++ {
			strs = append(strs, ToString(v.List().At(i)))
		}
		return "[" + strings.Join(strs, ",") + "]"
	case v.IsMap():
		strs := []string{}
		m := v.Map()
		m.Iterate(func(k string, v Value) bool {
			strs = append(strs, fmt.Sprintf("%v=%v", k, ToString(v)))
			return true
		})
		return "{" + strings.Join(strs, ",") + "}"
	}
	// No field is set, on either objects.
	return "{{undefined}}"
}

// Less provides a total ordering for Value (so that they can be sorted, even
// if they are of different types).
func Less(lhs, rhs Value) bool {
	return Compare(lhs, rhs) == -1
}

// Compare provides a total ordering for Value (so that they can be
// sorted, even if they are of different types). The result will be 0 if
// v==rhs, -1 if v < rhs, and +1 if v > rhs.
func Compare(lhs, rhs Value) int {
	if lhs.IsFloat() {
		if !rhs.IsFloat() {
			// Extra: compare floats and ints numerically.
			if rhs.IsInt() {
				return FloatCompare(lhs.Float(), float64(rhs.Int()))
			}
			return -1
		}
		return FloatCompare(lhs.Float(), rhs.Float())
	} else if rhs.IsFloat() {
		// Extra: compare floats and ints numerically.
		if lhs.IsInt() {
			return FloatCompare(float64(lhs.Int()), rhs.Float())
		}
		return 1
	}

	if lhs.IsInt() {
		if !rhs.IsInt() {
			return -1
		}
		return IntCompare(lhs.Int(), rhs.Int())
	} else if rhs.IsInt() {
		return 1
	}

	if lhs.IsString() {
		if !rhs.IsString() {
			return -1
		}
		return strings.Compare(lhs.String(), rhs.String())
	} else if rhs.IsString() {
		return 1
	}

	if lhs.IsBool() {
		if !rhs.IsBool() {
			return -1
		}
		return BoolCompare(lhs.Bool(), rhs.Bool())
	} else if rhs.IsBool() {
		return 1
	}

	if lhs.IsList() {
		if !rhs.IsList() {
			return -1
		}
		return ListCompare(lhs.List(), rhs.List())
	} else if rhs.IsList() {
		return 1
	}
	if lhs.IsMap() {
		if !rhs.IsMap() {
			return -1
		}
		return MapCompare(lhs.Map(), rhs.Map())
	} else if rhs.IsMap() {
		return 1
	}
	if lhs.IsNull() {
		if !rhs.IsNull() {
			return -1
		}
		return 0
	} else if rhs.IsNull() {
		return 1
	}

	// Invalid Value-- nothing is set.
	return 0
}

type Stream interface {
	Write([]byte) (int, error)
	WriteRaw(string)
	WriteString(string)
	WriteBool(bool)
	WriteInt(int)
	WriteInt64(int64)
	WriteFloat64(float64)
	WriteNil()
	WriteArrayStart()
	WriteArrayEnd()
	WriteObjectStart()
	WriteObjectEnd()
	WriteObjectField(string)
	Buffer() []byte
	Flush() error
	SetBuffer(buf []byte)
}
