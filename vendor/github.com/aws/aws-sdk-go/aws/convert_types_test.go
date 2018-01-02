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
				if !(*(out2[i])).IsZero() {
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
