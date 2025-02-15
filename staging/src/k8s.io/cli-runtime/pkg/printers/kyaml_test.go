/*
Copyright 2025 The Kubernetes Authors.

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

package printers

import (
	"bytes"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/randfill"
	"sigs.k8s.io/yaml"
)

type AllTypesStruct struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Basic types
	String  string      `json:"string,omitempty"`
	Bool    bool        `json:"bool,omitempty"`
	Int     int         `json:"int,omitempty"`
	Int8    int8        `json:"int8,omitempty"`
	Int16   int16       `json:"int16,omitempty"`
	Int32   int32       `json:"int32,omitempty"`
	Int64   int64       `json:"int64,omitempty"`
	Uint    uint        `json:"uint,omitempty"`
	Uint8   uint8       `json:"uint8,omitempty"`
	Uint16  uint16      `json:"uint16,omitempty"`
	Uint32  uint32      `json:"uint32,omitempty"`
	Uint64  uint64      `json:"uint64,omitempty"`
	Float32 float32     `json:"float32,omitempty"`
	Float64 float64     `json:"float64,omitempty"`
	Time    metav1.Time `json:"time,omitempty"`
	Bytes   []byte      `json:"bytes,omitempty"`

	// Pointers to basic types
	StringPtr  *string      `json:"stringPtr,omitempty"`
	BoolPtr    *bool        `json:"boolPtr,omitempty"`
	IntPtr     *int         `json:"intPtr,omitempty"`
	Int8Ptr    *int8        `json:"int8Ptr,omitempty"`
	Int16Ptr   *int16       `json:"int16Ptr,omitempty"`
	Int32Ptr   *int32       `json:"int32Ptr,omitempty"`
	Int64Ptr   *int64       `json:"int64Ptr,omitempty"`
	UintPtr    *uint        `json:"uintPtr,omitempty"`
	Uint8Ptr   *uint8       `json:"uint8Ptr,omitempty"`
	Uint16Ptr  *uint16      `json:"uint16Ptr,omitempty"`
	Uint32Ptr  *uint32      `json:"uint32Ptr,omitempty"`
	Uint64Ptr  *uint64      `json:"uint64Ptr,omitempty"`
	Float32Ptr *float32     `json:"float32Ptr,omitempty"`
	Float64Ptr *float64     `json:"float64Ptr,omitempty"`
	TimePtr    *metav1.Time `json:"timePtr,omitempty"`

	// Slices of basic types
	StringSlice  []string      `json:"stringSlice,omitempty"`
	BoolSlice    []bool        `json:"boolSlice,omitempty"`
	IntSlice     []int         `json:"intSlice,omitempty"`
	Int8Slice    []int8        `json:"int8Slice,omitempty"`
	Int16Slice   []int16       `json:"int16Slice,omitempty"`
	Int32Slice   []int32       `json:"int32Slice,omitempty"`
	Int64Slice   []int64       `json:"int64Slice,omitempty"`
	UintSlice    []uint        `json:"uintSlice,omitempty"`
	Uint8Slice   []uint8       `json:"uint8Slice,omitempty"`
	Uint16Slice  []uint16      `json:"uint16Slice,omitempty"`
	Uint32Slice  []uint32      `json:"uint32Slice,omitempty"`
	Uint64Slice  []uint64      `json:"uint64Slice,omitempty"`
	Float32Slice []float32     `json:"float32Slice,omitempty"`
	Float64Slice []float64     `json:"float64Slice,omitempty"`
	TimeSlice    []metav1.Time `json:"timeSlice,omitempty"`

	// Maps of string to basic types
	StringMap  map[string]string      `json:"stringMap,omitempty"`
	BoolMap    map[string]bool        `json:"boolMap,omitempty"`
	IntMap     map[string]int         `json:"intMap,omitempty"`
	Int8Map    map[string]int8        `json:"int8Map,omitempty"`
	Int16Map   map[string]int16       `json:"int16Map,omitempty"`
	Int32Map   map[string]int32       `json:"int32Map,omitempty"`
	Int64Map   map[string]int64       `json:"int64Map,omitempty"`
	UintMap    map[string]uint        `json:"uintMap,omitempty"`
	Uint8Map   map[string]uint8       `json:"uint8Map,omitempty"`
	Uint16Map  map[string]uint16      `json:"uint16Map,omitempty"`
	Uint32Map  map[string]uint32      `json:"uint32Map,omitempty"`
	Uint64Map  map[string]uint64      `json:"uint64Map,omitempty"`
	Float32Map map[string]float32     `json:"float32Map,omitempty"`
	Float64Map map[string]float64     `json:"float64Map,omitempty"`
	TimeMap    map[string]metav1.Time `json:"timeMap,omitempty"`

	// Slice of slices
	StringSliceSlice  [][]string      `json:"stringSliceSlice,omitempty"`
	BoolSliceSlice    [][]bool        `json:"boolSliceSlice,omitempty"`
	IntSliceSlice     [][]int         `json:"intSliceSlice,omitempty"`
	Int8SliceSlice    [][]int8        `json:"int8SliceSlice,omitempty"`
	Int16SliceSlice   [][]int16       `json:"int16SliceSlice,omitempty"`
	Int32SliceSlice   [][]int32       `json:"int32SliceSlice,omitempty"`
	Int64SliceSlice   [][]int64       `json:"int64SliceSlice,omitempty"`
	UintSliceSlice    [][]uint        `json:"uintSliceSlice,omitempty"`
	Uint8SliceSlice   [][]uint8       `json:"uint8SliceSlice,omitempty"`
	Uint16SliceSlice  [][]uint16      `json:"uint16SliceSlice,omitempty"`
	Uint32SliceSlice  [][]uint32      `json:"uint32SliceSlice,omitempty"`
	Uint64SliceSlice  [][]uint64      `json:"uint64SliceSlice,omitempty"`
	Float32SliceSlice [][]float32     `json:"float32SliceSlice,omitempty"`
	Float64SliceSlice [][]float64     `json:"float64SliceSlice,omitempty"`
	TimeSliceSlice    [][]metav1.Time `json:"timeSliceSlice,omitempty"`

	// Slice of maps
	StringSliceMap  []map[string]string      `json:"stringSliceMap,omitempty"`
	BoolSliceMap    []map[string]bool        `json:"boolSliceMap,omitempty"`
	IntSliceMap     []map[string]int         `json:"intSliceMap,omitempty"`
	Int8SliceMap    []map[string]int8        `json:"int8SliceMap,omitempty"`
	Int16SliceMap   []map[string]int16       `json:"int16SliceMap,omitempty"`
	Int32SliceMap   []map[string]int32       `json:"int32SliceMap,omitempty"`
	Int64SliceMap   []map[string]int64       `json:"int64SliceMap,omitempty"`
	UintSliceMap    []map[string]uint        `json:"uintSliceMap,omitempty"`
	Uint8SliceMap   []map[string]uint8       `json:"uint8SliceMap,omitempty"`
	Uint16SliceMap  []map[string]uint16      `json:"uint16SliceMap,omitempty"`
	Uint32SliceMap  []map[string]uint32      `json:"uint32SliceMap,omitempty"`
	Uint64SliceMap  []map[string]uint64      `json:"uint64SliceMap,omitempty"`
	Float32SliceMap []map[string]float32     `json:"float32SliceMap,omitempty"`
	Float64SliceMap []map[string]float64     `json:"float64SliceMap,omitempty"`
	TimeSliceMap    []map[string]metav1.Time `json:"timeSliceMap,omitempty"`

	// Map of string to slices
	StringMapSlice  map[string][]string      `json:"stringMapSlice,omitempty"`
	BoolMapSlice    map[string][]bool        `json:"boolMapSlice,omitempty"`
	IntMapSlice     map[string][]int         `json:"intMapSlice,omitempty"`
	Int8MapSlice    map[string][]int8        `json:"int8MapSlice,omitempty"`
	Int16MapSlice   map[string][]int16       `json:"int16MapSlice,omitempty"`
	Int32MapSlice   map[string][]int32       `json:"int32MapSlice,omitempty"`
	Int64MapSlice   map[string][]int64       `json:"int64MapSlice,omitempty"`
	UintMapSlice    map[string][]uint        `json:"uintMapSlice,omitempty"`
	Uint8MapSlice   map[string][]uint8       `json:"uint8MapSlice,omitempty"`
	Uint16MapSlice  map[string][]uint16      `json:"uint16MapSlice,omitempty"`
	Uint32MapSlice  map[string][]uint32      `json:"uint32MapSlice,omitempty"`
	Uint64MapSlice  map[string][]uint64      `json:"uint64MapSlice,omitempty"`
	Float32MapSlice map[string][]float32     `json:"float32MapSlice,omitempty"`
	Float64MapSlice map[string][]float64     `json:"float64MapSlice,omitempty"`
	TimeMapSlice    map[string][]metav1.Time `json:"timeMapSlice,omitempty"`

	// Map of string to maps
	StringMapMap  map[string]map[string]string      `json:"stringMapMap,omitempty"`
	BoolMapMap    map[string]map[string]bool        `json:"boolMapMap,omitempty"`
	IntMapMap     map[string]map[string]int         `json:"intMapMap,omitempty"`
	Int8MapMap    map[string]map[string]int8        `json:"int8MapMap,omitempty"`
	Int16MapMap   map[string]map[string]int16       `json:"int16MapMap,omitempty"`
	Int32MapMap   map[string]map[string]int32       `json:"int32MapMap,omitempty"`
	Int64MapMap   map[string]map[string]int64       `json:"int64MapMap,omitempty"`
	UintMapMap    map[string]map[string]uint        `json:"uintMapMap,omitempty"`
	Uint8MapMap   map[string]map[string]uint8       `json:"uint8MapMap,omitempty"`
	Uint16MapMap  map[string]map[string]uint16      `json:"uint16MapMap,omitempty"`
	Uint32MapMap  map[string]map[string]uint32      `json:"uint32MapMap,omitempty"`
	Uint64MapMap  map[string]map[string]uint64      `json:"uint64MapMap,omitempty"`
	Float32MapMap map[string]map[string]float32     `json:"float32MapMap,omitempty"`
	Float64MapMap map[string]map[string]float64     `json:"float64MapMap,omitempty"`
	TimeMapMap    map[string]map[string]metav1.Time `json:"timeMapMap,omitempty"`

	// Recursive types
	Self           *AllTypesStruct                       `json:"self,omitempty"`
	SelfSlice      []*AllTypesStruct                     `json:"selfSlice,omitempty"`
	SelfMap        map[string]*AllTypesStruct            `json:"selfMap,omitempty"`
	SelfSliceSlice [][](*AllTypesStruct)                 `json:"selfSliceSlice,omitempty"`
	SelfSliceMap   []map[string]*AllTypesStruct          `json:"selfSliceMap,omitempty"`
	SelfMapSlice   map[string][]*AllTypesStruct          `json:"selfMapSlice,omitempty"`
	SelfMapMap     map[string]map[string]*AllTypesStruct `json:"selfMapMap,omitempty"`
}

func TestKYAMLPrinterRoundTrip(t *testing.T) {
	for i := 0; i < 1000; i++ {
		// Create and fill an instance.
		original := &AllTypesStruct{}
		f := randfill.New().NilChance(0.5).NumElements(1, 5).MaxDepth(3)
		f.Fill(original)

		// Render to YAML.
		var buf bytes.Buffer
		printer := &KYAMLPrinter{}
		if err := printer.fromAny(original, &buf); err != nil {
			t.Fatalf("iteration %d: failed to render KYAML: %v", i, err)
		}

		// Parse back from YAML with the standard parser.
		parsed := &AllTypesStruct{}
		if err := yaml.Unmarshal(buf.Bytes(), parsed); err != nil {
			t.Fatalf("iteration %d: failed to parse KYAML: %v\nKYAML:\n%s", i, err, buf.String())
		}

		// Compare.
		if diff := cmp.Diff(original, parsed, cmpopts.EquateEmpty()); diff != "" {
			t.Fatalf("iteration %d: objects differ after round trip (-original +parsed):\n%s\nKYAML:\n%s", i, diff, buf.String())
		}
	}
}

type SelfMarshalStruct struct {
	String string `json:"string,omitempty"`
}

func (s SelfMarshalStruct) MarshalJSON() ([]byte, error) {
	return []byte(`{"key":"value"}`), nil
}

func TestKYAMLSelfMarshal(t *testing.T) {
	original := &SelfMarshalStruct{String: "string"}
	var buf bytes.Buffer
	printer := &KYAMLPrinter{}
	if err := printer.fromAny(original, &buf); err != nil {
		t.Fatalf("failed to render KYAML: %v", err)
	}
	expected := "{\n  key: \"value\",\n}\n"
	if buf.String() != expected {
		t.Fatalf("wrong result:\nexpected: %q\n     got: %q", expected, buf.String())
	}
}

func TestYamlGuessesWrong(t *testing.T) {
	type testCase struct {
		name     string
		input    string
		expected bool
	}

	tests := []testCase{
		// Regular strings that should not need quotes
		{"regular string", "regular", false},
		{"alphanumeric", "abc123", false},
		{"underscore", "hello_world", false},
		{"hyphen", "hello-world", false},

		// Boolean-like strings
		{"yes", "yes", true},
		{"YES", "YES", true},
		{"y", "y", true},
		{"Y", "Y", true},
		{"no", "no", true},
		{"NO", "NO", true},
		{"n", "n", true},
		{"N", "N", true},
		{"on", "on", true},
		{"ON", "ON", true},
		{"off", "off", true},
		{"OFF", "OFF", true},

		// Numbers should stay strings
		{"decimal", "1234", true},
		{"underscores", "_1_2_3_4_", true},
		{"leading zero", "0123", true},
		{"plus sign", "+123", true},
		{"negative sign", "-123", true},
		{"large decimal", "123456789012345678901234567890", true},
		{"octal 0", "0777", true},
		{"octal 0o", "0o777", true},
		{"hex lowercase", "0xff", true},
		{"hex uppercase", "0xFF", true},

		// Infinity and NaN
		{"infinity", ".inf", true},
		{"negative infinity", "-.inf", true},
		{"positive infinity", "+.inf", true},
		{"not a number", ".nan", true},
		{"uppercase infinity", ".INF", true},
		{"uppercase nan", ".NAN", true},

		// Scientific notation
		{"scientific", "1e10", true},
		{"scientific uppercase", "1E10", true},

		// Timestamp-like strings
		{"year", "2006", true},
		{"date", "2006-1-2", true},
		{"RCF3339Nano with short date", "2006-1-2T15:4:5.999999999-08:00", true},
		{"RCF3339Nano with lowercase t", "2006-1-2t15:4:5.999999999-08:00", true},
		{"space separated", "2006-1-2 14:4:5.999999999", true},

		// Null-like strings
		{"null lowercase", "null", true},
		{"null mixed", "Null", true},
		{"null uppercase", "NULL", true},
		{"tilde", "~", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := yamlGuessesWrong(tt.input)
			if result != tt.expected {
				t.Errorf("yamlGuessesWrong(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}
