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
	yaml "sigs.k8s.io/yaml/goyaml.v2"
)

var (
	readPool  = jsoniter.NewIterator(jsoniter.ConfigCompatibleWithStandardLibrary).Pool()
	writePool = jsoniter.NewStream(jsoniter.ConfigCompatibleWithStandardLibrary, nil, 1024).Pool()
)

// A Value corresponds to an 'atom' in the schema. It should return true
// for at least one of the IsXXX methods below, or the value is
// considered "invalid"
type Value interface {
	// IsMap returns true if the Value is a Map, false otherwise.
	IsMap() bool
	// IsList returns true if the Value is a List, false otherwise.
	IsList() bool
	// IsBool returns true if the Value is a bool, false otherwise.
	IsBool() bool
	// IsInt returns true if the Value is a int64, false otherwise.
	IsInt() bool
	// IsFloat returns true if the Value is a float64, false
	// otherwise.
	IsFloat() bool
	// IsString returns true if the Value is a string, false
	// otherwise.
	IsString() bool
	// IsMap returns true if the Value is null, false otherwise.
	IsNull() bool

	// AsMap converts the Value into a Map (or panic if the type
	// doesn't allow it).
	AsMap() Map
	// AsMapUsing uses the provided allocator and converts the Value
	// into a Map (or panic if the type doesn't allow it).
	AsMapUsing(Allocator) Map
	// AsList converts the Value into a List (or panic if the type
	// doesn't allow it).
	AsList() List
	// AsListUsing uses the provided allocator and converts the Value
	// into a List (or panic if the type doesn't allow it).
	AsListUsing(Allocator) List
	// AsBool converts the Value into a bool (or panic if the type
	// doesn't allow it).
	AsBool() bool
	// AsInt converts the Value into an int64 (or panic if the type
	// doesn't allow it).
	AsInt() int64
	// AsFloat converts the Value into a float64 (or panic if the type
	// doesn't allow it).
	AsFloat() float64
	// AsString converts the Value into a string (or panic if the type
	// doesn't allow it).
	AsString() string

	// Unstructured converts the Value into an Unstructured interface{}.
	Unstructured() interface{}
}

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
func WriteJSONStream(v Value, stream *jsoniter.Stream) {
	stream.WriteVal(v.Unstructured())
}

// ToYAML marshals a value as YAML.
func ToYAML(v Value) ([]byte, error) {
	return yaml.Marshal(v.Unstructured())
}

// Equals returns true iff the two values are equal.
func Equals(lhs, rhs Value) bool {
	return EqualsUsing(HeapAllocator, lhs, rhs)
}

// EqualsUsing uses the provided allocator and returns true iff the two values are equal.
func EqualsUsing(a Allocator, lhs, rhs Value) bool {
	if lhs.IsFloat() || rhs.IsFloat() {
		var lf float64
		if lhs.IsFloat() {
			lf = lhs.AsFloat()
		} else if lhs.IsInt() {
			lf = float64(lhs.AsInt())
		} else {
			return false
		}
		var rf float64
		if rhs.IsFloat() {
			rf = rhs.AsFloat()
		} else if rhs.IsInt() {
			rf = float64(rhs.AsInt())
		} else {
			return false
		}
		return lf == rf
	}
	if lhs.IsInt() {
		if rhs.IsInt() {
			return lhs.AsInt() == rhs.AsInt()
		}
		return false
	} else if rhs.IsInt() {
		return false
	}
	if lhs.IsString() {
		if rhs.IsString() {
			return lhs.AsString() == rhs.AsString()
		}
		return false
	} else if rhs.IsString() {
		return false
	}
	if lhs.IsBool() {
		if rhs.IsBool() {
			return lhs.AsBool() == rhs.AsBool()
		}
		return false
	} else if rhs.IsBool() {
		return false
	}
	if lhs.IsList() {
		if rhs.IsList() {
			lhsList := lhs.AsListUsing(a)
			defer a.Free(lhsList)
			rhsList := rhs.AsListUsing(a)
			defer a.Free(rhsList)
			return lhsList.EqualsUsing(a, rhsList)
		}
		return false
	} else if rhs.IsList() {
		return false
	}
	if lhs.IsMap() {
		if rhs.IsMap() {
			lhsList := lhs.AsMapUsing(a)
			defer a.Free(lhsList)
			rhsList := rhs.AsMapUsing(a)
			defer a.Free(rhsList)
			return lhsList.EqualsUsing(a, rhsList)
		}
		return false
	} else if rhs.IsMap() {
		return false
	}
	if lhs.IsNull() {
		if rhs.IsNull() {
			return true
		}
		return false
	} else if rhs.IsNull() {
		return false
	}
	// No field is set, on either objects.
	return true
}

// ToString returns a human-readable representation of the value.
func ToString(v Value) string {
	if v.IsNull() {
		return "null"
	}
	switch {
	case v.IsFloat():
		return fmt.Sprintf("%v", v.AsFloat())
	case v.IsInt():
		return fmt.Sprintf("%v", v.AsInt())
	case v.IsString():
		return fmt.Sprintf("%q", v.AsString())
	case v.IsBool():
		return fmt.Sprintf("%v", v.AsBool())
	case v.IsList():
		strs := []string{}
		list := v.AsList()
		for i := 0; i < list.Length(); i++ {
			strs = append(strs, ToString(list.At(i)))
		}
		return "[" + strings.Join(strs, ",") + "]"
	case v.IsMap():
		strs := []string{}
		v.AsMap().Iterate(func(k string, v Value) bool {
			strs = append(strs, fmt.Sprintf("%v=%v", k, ToString(v)))
			return true
		})
		return strings.Join(strs, "")
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
	return CompareUsing(HeapAllocator, lhs, rhs)
}

// CompareUsing uses the provided allocator and provides a total
// ordering for Value (so that they can be sorted, even if they
// are of different types). The result will be 0 if v==rhs, -1
// if v < rhs, and +1 if v > rhs.
func CompareUsing(a Allocator, lhs, rhs Value) int {
	if lhs.IsFloat() {
		if !rhs.IsFloat() {
			// Extra: compare floats and ints numerically.
			if rhs.IsInt() {
				return FloatCompare(lhs.AsFloat(), float64(rhs.AsInt()))
			}
			return -1
		}
		return FloatCompare(lhs.AsFloat(), rhs.AsFloat())
	} else if rhs.IsFloat() {
		// Extra: compare floats and ints numerically.
		if lhs.IsInt() {
			return FloatCompare(float64(lhs.AsInt()), rhs.AsFloat())
		}
		return 1
	}

	if lhs.IsInt() {
		if !rhs.IsInt() {
			return -1
		}
		return IntCompare(lhs.AsInt(), rhs.AsInt())
	} else if rhs.IsInt() {
		return 1
	}

	if lhs.IsString() {
		if !rhs.IsString() {
			return -1
		}
		return strings.Compare(lhs.AsString(), rhs.AsString())
	} else if rhs.IsString() {
		return 1
	}

	if lhs.IsBool() {
		if !rhs.IsBool() {
			return -1
		}
		return BoolCompare(lhs.AsBool(), rhs.AsBool())
	} else if rhs.IsBool() {
		return 1
	}

	if lhs.IsList() {
		if !rhs.IsList() {
			return -1
		}
		lhsList := lhs.AsListUsing(a)
		defer a.Free(lhsList)
		rhsList := rhs.AsListUsing(a)
		defer a.Free(rhsList)
		return ListCompareUsing(a, lhsList, rhsList)
	} else if rhs.IsList() {
		return 1
	}
	if lhs.IsMap() {
		if !rhs.IsMap() {
			return -1
		}
		lhsMap := lhs.AsMapUsing(a)
		defer a.Free(lhsMap)
		rhsMap := rhs.AsMapUsing(a)
		defer a.Free(rhsMap)
		return MapCompareUsing(a, lhsMap, rhsMap)
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
