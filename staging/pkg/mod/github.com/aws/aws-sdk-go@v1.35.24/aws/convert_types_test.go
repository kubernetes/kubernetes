package aws

import (
	"reflect"
	"testing"
	"time"
)

var testCasesStringSlice = [][]string{
	{"a", "b", "c", "d", "e"},
	{"a", "b", "", "", "e"},
}

func TestStringSlice(t *testing.T) {
	for idx, in := range testCasesStringSlice {
		if in == nil {
			continue
		}
		out := StringSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := StringValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesStringValueSlice = [][]*string{
	{String("a"), String("b"), nil, String("c")},
}

func TestStringValueSlice(t *testing.T) {
	for idx, in := range testCasesStringValueSlice {
		if in == nil {
			continue
		}
		out := StringValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != "" {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := StringSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != "" {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *in[i], *out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesStringMap = []map[string]string{
	{"a": "1", "b": "2", "c": "3"},
}

func TestStringMap(t *testing.T) {
	for idx, in := range testCasesStringMap {
		if in == nil {
			continue
		}
		out := StringMap(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := StringValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesBoolSlice = [][]bool{
	{true, true, false, false},
}

func TestBoolSlice(t *testing.T) {
	for idx, in := range testCasesBoolSlice {
		if in == nil {
			continue
		}
		out := BoolSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := BoolValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesBoolValueSlice = [][]*bool{}

func TestBoolValueSlice(t *testing.T) {
	for idx, in := range testCasesBoolValueSlice {
		if in == nil {
			continue
		}
		out := BoolValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := BoolSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesBoolMap = []map[string]bool{
	{"a": true, "b": false, "c": true},
}

func TestBoolMap(t *testing.T) {
	for idx, in := range testCasesBoolMap {
		if in == nil {
			continue
		}
		out := BoolMap(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := BoolValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUintSlice = [][]uint{
	{1, 2, 3, 4},
}

func TestUintSlice(t *testing.T) {
	for idx, in := range testCasesUintSlice {
		if in == nil {
			continue
		}
		out := UintSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := UintValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUintValueSlice = [][]*uint{}

func TestUintValueSlice(t *testing.T) {
	for idx, in := range testCasesUintValueSlice {
		if in == nil {
			continue
		}
		out := UintValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := UintSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesUintMap = []map[string]uint{
	{"a": 3, "b": 2, "c": 1},
}

func TestUintMap(t *testing.T) {
	for idx, in := range testCasesUintMap {
		if in == nil {
			continue
		}
		out := UintMap(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := UintValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesIntSlice = [][]int{
	{1, 2, 3, 4},
}

func TestIntSlice(t *testing.T) {
	for idx, in := range testCasesIntSlice {
		if in == nil {
			continue
		}
		out := IntSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := IntValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesIntValueSlice = [][]*int{}

func TestIntValueSlice(t *testing.T) {
	for idx, in := range testCasesIntValueSlice {
		if in == nil {
			continue
		}
		out := IntValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := IntSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesIntMap = []map[string]int{
	{"a": 3, "b": 2, "c": 1},
}

func TestIntMap(t *testing.T) {
	for idx, in := range testCasesIntMap {
		if in == nil {
			continue
		}
		out := IntMap(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := IntValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt8Slice = [][]int8{
	{1, 2, 3, 4},
}

func TestInt8Slice(t *testing.T) {
	for idx, in := range testCasesInt8Slice {
		if in == nil {
			continue
		}
		out := Int8Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int8ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt8ValueSlice = [][]*int8{}

func TestInt8ValueSlice(t *testing.T) {
	for idx, in := range testCasesInt8ValueSlice {
		if in == nil {
			continue
		}
		out := Int8ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Int8Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesInt8Map = []map[string]int8{
	{"a": 3, "b": 2, "c": 1},
}

func TestInt8Map(t *testing.T) {
	for idx, in := range testCasesInt8Map {
		if in == nil {
			continue
		}
		out := Int8Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int8ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt16Slice = [][]int16{
	{1, 2, 3, 4},
}

func TestInt16Slice(t *testing.T) {
	for idx, in := range testCasesInt16Slice {
		if in == nil {
			continue
		}
		out := Int16Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int16ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt16ValueSlice = [][]*int16{}

func TestInt16ValueSlice(t *testing.T) {
	for idx, in := range testCasesInt16ValueSlice {
		if in == nil {
			continue
		}
		out := Int16ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Int16Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesInt16Map = []map[string]int16{
	{"a": 3, "b": 2, "c": 1},
}

func TestInt16Map(t *testing.T) {
	for idx, in := range testCasesInt16Map {
		if in == nil {
			continue
		}
		out := Int16Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int16ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt32Slice = [][]int32{
	{1, 2, 3, 4},
}

func TestInt32Slice(t *testing.T) {
	for idx, in := range testCasesInt32Slice {
		if in == nil {
			continue
		}
		out := Int32Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int32ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt32ValueSlice = [][]*int32{}

func TestInt32ValueSlice(t *testing.T) {
	for idx, in := range testCasesInt32ValueSlice {
		if in == nil {
			continue
		}
		out := Int32ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Int32Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesInt32Map = []map[string]int32{
	{"a": 3, "b": 2, "c": 1},
}

func TestInt32Map(t *testing.T) {
	for idx, in := range testCasesInt32Map {
		if in == nil {
			continue
		}
		out := Int32Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int32ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt64Slice = [][]int64{
	{1, 2, 3, 4},
}

func TestInt64Slice(t *testing.T) {
	for idx, in := range testCasesInt64Slice {
		if in == nil {
			continue
		}
		out := Int64Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int64ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesInt64ValueSlice = [][]*int64{}

func TestInt64ValueSlice(t *testing.T) {
	for idx, in := range testCasesInt64ValueSlice {
		if in == nil {
			continue
		}
		out := Int64ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Int64Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesInt64Map = []map[string]int64{
	{"a": 3, "b": 2, "c": 1},
}

func TestInt64Map(t *testing.T) {
	for idx, in := range testCasesInt64Map {
		if in == nil {
			continue
		}
		out := Int64Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Int64ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint8Slice = [][]uint8{
	{1, 2, 3, 4},
}

func TestUint8Slice(t *testing.T) {
	for idx, in := range testCasesUint8Slice {
		if in == nil {
			continue
		}
		out := Uint8Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint8ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint8ValueSlice = [][]*uint8{}

func TestUint8ValueSlice(t *testing.T) {
	for idx, in := range testCasesUint8ValueSlice {
		if in == nil {
			continue
		}
		out := Uint8ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Uint8Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesUint8Map = []map[string]uint8{
	{"a": 3, "b": 2, "c": 1},
}

func TestUint8Map(t *testing.T) {
	for idx, in := range testCasesUint8Map {
		if in == nil {
			continue
		}
		out := Uint8Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint8ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint16Slice = [][]uint16{
	{1, 2, 3, 4},
}

func TestUint16Slice(t *testing.T) {
	for idx, in := range testCasesUint16Slice {
		if in == nil {
			continue
		}
		out := Uint16Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint16ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint16ValueSlice = [][]*uint16{}

func TestUint16ValueSlice(t *testing.T) {
	for idx, in := range testCasesUint16ValueSlice {
		if in == nil {
			continue
		}
		out := Uint16ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Uint16Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesUint16Map = []map[string]uint16{
	{"a": 3, "b": 2, "c": 1},
}

func TestUint16Map(t *testing.T) {
	for idx, in := range testCasesUint16Map {
		if in == nil {
			continue
		}
		out := Uint16Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint16ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint32Slice = [][]uint32{
	{1, 2, 3, 4},
}

func TestUint32Slice(t *testing.T) {
	for idx, in := range testCasesUint32Slice {
		if in == nil {
			continue
		}
		out := Uint32Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint32ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint32ValueSlice = [][]*uint32{}

func TestUint32ValueSlice(t *testing.T) {
	for idx, in := range testCasesUint32ValueSlice {
		if in == nil {
			continue
		}
		out := Uint32ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Uint32Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesUint32Map = []map[string]uint32{
	{"a": 3, "b": 2, "c": 1},
}

func TestUint32Map(t *testing.T) {
	for idx, in := range testCasesUint32Map {
		if in == nil {
			continue
		}
		out := Uint32Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint32ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint64Slice = [][]uint64{
	{1, 2, 3, 4},
}

func TestUint64Slice(t *testing.T) {
	for idx, in := range testCasesUint64Slice {
		if in == nil {
			continue
		}
		out := Uint64Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint64ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesUint64ValueSlice = [][]*uint64{}

func TestUint64ValueSlice(t *testing.T) {
	for idx, in := range testCasesUint64ValueSlice {
		if in == nil {
			continue
		}
		out := Uint64ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Uint64Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesUint64Map = []map[string]uint64{
	{"a": 3, "b": 2, "c": 1},
}

func TestUint64Map(t *testing.T) {
	for idx, in := range testCasesUint64Map {
		if in == nil {
			continue
		}
		out := Uint64Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Uint64ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesFloat32Slice = [][]float32{
	{1, 2, 3, 4},
}

func TestFloat32Slice(t *testing.T) {
	for idx, in := range testCasesFloat32Slice {
		if in == nil {
			continue
		}
		out := Float32Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Float32ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesFloat32ValueSlice = [][]*float32{}

func TestFloat32ValueSlice(t *testing.T) {
	for idx, in := range testCasesFloat32ValueSlice {
		if in == nil {
			continue
		}
		out := Float32ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Float32Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesFloat32Map = []map[string]float32{
	{"a": 3, "b": 2, "c": 1},
}

func TestFloat32Map(t *testing.T) {
	for idx, in := range testCasesFloat32Map {
		if in == nil {
			continue
		}
		out := Float32Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Float32ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesFloat64Slice = [][]float64{
	{1, 2, 3, 4},
}

func TestFloat64Slice(t *testing.T) {
	for idx, in := range testCasesFloat64Slice {
		if in == nil {
			continue
		}
		out := Float64Slice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Float64ValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesFloat64ValueSlice = [][]*float64{}

func TestFloat64ValueSlice(t *testing.T) {
	for idx, in := range testCasesFloat64ValueSlice {
		if in == nil {
			continue
		}
		out := Float64ValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if out[i] != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := Float64Slice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if *(out2[i]) != 0 {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesFloat64Map = []map[string]float64{
	{"a": 3, "b": 2, "c": 1},
}

func TestFloat64Map(t *testing.T) {
	for idx, in := range testCasesFloat64Map {
		if in == nil {
			continue
		}
		out := Float64Map(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := Float64ValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesTimeSlice = [][]time.Time{
	{time.Now(), time.Now().AddDate(100, 0, 0)},
}

func TestTimeSlice(t *testing.T) {
	for idx, in := range testCasesTimeSlice {
		if in == nil {
			continue
		}
		out := TimeSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := TimeValueSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

var testCasesTimeValueSlice = [][]*time.Time{}

func TestTimeValueSlice(t *testing.T) {
	for idx, in := range testCasesTimeValueSlice {
		if in == nil {
			continue
		}
		out := TimeValueSlice(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if in[i] == nil {
				if !out[i].IsZero() {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := *(in[i]), out[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}

		out2 := TimeSlice(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out2 {
			if in[i] == nil {
				if !(out2[i]).IsZero() {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			} else {
				if e, a := in[i], out2[i]; e != a {
					t.Errorf("Unexpected value at idx %d", idx)
				}
			}
		}
	}
}

var testCasesTimeMap = []map[string]time.Time{
	{"a": time.Now().AddDate(-100, 0, 0), "b": time.Now()},
}

func TestTimeMap(t *testing.T) {
	for idx, in := range testCasesTimeMap {
		if in == nil {
			continue
		}
		out := TimeMap(in)
		if e, a := len(out), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		for i := range out {
			if e, a := in[i], *(out[i]); e != a {
				t.Errorf("Unexpected value at idx %d", idx)
			}
		}

		out2 := TimeValueMap(out)
		if e, a := len(out2), len(in); e != a {
			t.Errorf("Unexpected len at idx %d", idx)
		}
		if e, a := in, out2; !reflect.DeepEqual(e, a) {
			t.Errorf("Unexpected value at idx %d", idx)
		}
	}
}

type TimeValueTestCase struct {
	in        int64
	outSecs   time.Time
	outMillis time.Time
}

var testCasesTimeValue = []TimeValueTestCase{
	{
		in:        int64(1501558289000),
		outSecs:   time.Unix(1501558289, 0),
		outMillis: time.Unix(1501558289, 0),
	},
	{
		in:        int64(1501558289001),
		outSecs:   time.Unix(1501558289, 0),
		outMillis: time.Unix(1501558289, 1*1000000),
	},
}

func TestSecondsTimeValue(t *testing.T) {
	for idx, testCase := range testCasesTimeValue {
		out := SecondsTimeValue(&testCase.in)
		if e, a := testCase.outSecs, out; e != a {
			t.Errorf("Unexpected value for time value at %d", idx)
		}
	}
}

func TestMillisecondsTimeValue(t *testing.T) {
	for idx, testCase := range testCasesTimeValue {
		out := MillisecondsTimeValue(&testCase.in)
		if e, a := testCase.outMillis, out; e != a {
			t.Errorf("Unexpected value for time value at %d", idx)
		}
	}
}
