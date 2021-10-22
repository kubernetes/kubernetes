// Copyright 2016 Google LLC
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

package valuecollector

import (
	"fmt"
	"testing"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	"cloud.google.com/go/internal/testutil"
	cd "google.golang.org/api/clouddebugger/v2"
)

const (
	// Some arbitrary type IDs for the test, for use in debug.Var's TypeID field.
	// A TypeID of 0 means the type is unknown, so we start at 1.
	int16Type = iota + 1
	stringType
	structType
	pointerType
	arrayType
	int32Type
	debugStringType
	mapType
	channelType
	sliceType
)

func TestValueCollector(t *testing.T) {
	// Construct the collector.
	c := NewCollector(&Program{}, 26)
	// Add some variables of various types, whose values we want the collector to read.
	variablesToAdd := []debug.LocalVar{
		{Name: "a", Var: debug.Var{TypeID: int16Type, Address: 0x1}},
		{Name: "b", Var: debug.Var{TypeID: stringType, Address: 0x2}},
		{Name: "c", Var: debug.Var{TypeID: structType, Address: 0x3}},
		{Name: "d", Var: debug.Var{TypeID: pointerType, Address: 0x4}},
		{Name: "e", Var: debug.Var{TypeID: arrayType, Address: 0x5}},
		{Name: "f", Var: debug.Var{TypeID: debugStringType, Address: 0x6}},
		{Name: "g", Var: debug.Var{TypeID: mapType, Address: 0x7}},
		{Name: "h", Var: debug.Var{TypeID: channelType, Address: 0x8}},
		{Name: "i", Var: debug.Var{TypeID: sliceType, Address: 0x9}},
	}
	expectedResults := []*cd.Variable{
		{Name: "a", VarTableIndex: 1},
		{Name: "b", VarTableIndex: 2},
		{Name: "c", VarTableIndex: 3},
		{Name: "d", VarTableIndex: 4},
		{Name: "e", VarTableIndex: 5},
		{Name: "f", VarTableIndex: 6},
		{Name: "g", VarTableIndex: 7},
		{Name: "h", VarTableIndex: 8},
		{Name: "i", VarTableIndex: 9},
	}
	for i, v := range variablesToAdd {
		added := c.AddVariable(v)
		if !testutil.Equal(added, expectedResults[i]) {
			t.Errorf("AddVariable: got %+v want %+v", *added, *expectedResults[i])
		}
	}
	// Read the values, compare the output to what we expect.
	v := c.ReadValues()
	expectedValues := []*cd.Variable{
		{},
		{Value: "1"},
		{Value: `"hello"`},
		{
			Members: []*cd.Variable{
				{Name: "x", VarTableIndex: 1},
				{Name: "y", VarTableIndex: 2},
			},
		},
		{
			Members: []*cd.Variable{
				{VarTableIndex: 1},
			},
			Value: "0x1",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 10},
				{Name: "[1]", VarTableIndex: 11},
				{Name: "[2]", VarTableIndex: 12},
				{Name: "[3]", VarTableIndex: 13},
			},
			Value: "len = 4",
		},
		{Value: `"world"`},
		{
			Members: []*cd.Variable{
				{Name: "⚫", VarTableIndex: 14},
				{Name: "⚫", VarTableIndex: 15},
				{Name: "⚫", VarTableIndex: 16},
			},
			Value: "len = 3",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 17},
				{Name: "[1]", VarTableIndex: 18},
			},
			Value: "len = 2",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 19},
				{Name: "[1]", VarTableIndex: 20},
			},
			Value: "len = 2",
		},
		{Value: "100"},
		{Value: "104"},
		{Value: "108"},
		{Value: "112"},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 21},
				{Name: "value", VarTableIndex: 22},
			},
		},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 23},
				{Name: "value", VarTableIndex: 24},
			},
		},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 25},
				{
					Name: "value",
					Status: &cd.StatusMessage{
						Description: &cd.FormatMessage{
							Format:     "$0",
							Parameters: []string{"Not captured"},
						},
						IsError:  true,
						RefersTo: "VARIABLE_NAME",
					},
				},
			},
		},
		{Value: "246"},
		{Value: "210"},
		{Value: "300"},
		{Value: "304"},
		{Value: "400"},
		{Value: "404"},
		{Value: "1400"},
		{Value: "1404"},
		{Value: "2400"},
	}
	if !testutil.Equal(v, expectedValues) {
		t.Errorf("ReadValues: got %v want %v", v, expectedValues)
		// Do element-by-element comparisons, for more useful error messages.
		for i := range v {
			if i < len(expectedValues) && !testutil.Equal(v[i], expectedValues[i]) {
				t.Errorf("element %d: got %+v want %+v", i, *v[i], *expectedValues[i])
			}
		}
	}
}

// Program implements the similarly-named interface in x/debug.
// ValueCollector should only call its Value and MapElement methods.
type Program struct {
	debug.Program
}

func (p *Program) Value(v debug.Var) (debug.Value, error) {
	// We determine what to return using v.TypeID.
	switch v.TypeID {
	case int16Type:
		// We use the address as the value, so that we're testing whether the right
		// address was calculated.
		return int16(v.Address), nil
	case stringType:
		// A string.
		return "hello", nil
	case structType:
		// A struct with two elements.
		return debug.Struct{
			Fields: []debug.StructField{
				{
					Name: "x",
					Var:  debug.Var{TypeID: int16Type, Address: 0x1},
				},
				{
					Name: "y",
					Var:  debug.Var{TypeID: stringType, Address: 0x2},
				},
			},
		}, nil
	case pointerType:
		// A pointer to the first variable above.
		return debug.Pointer{TypeID: int16Type, Address: 0x1}, nil
	case arrayType:
		// An array of 4 32-bit-wide elements.
		return debug.Array{
			ElementTypeID: int32Type,
			Address:       0x64,
			Length:        4,
			StrideBits:    32,
		}, nil
	case debugStringType:
		return debug.String{
			Length: 5,
			String: "world",
		}, nil
	case mapType:
		return debug.Map{
			TypeID:  99,
			Address: 0x100,
			Length:  3,
		}, nil
	case channelType:
		return debug.Channel{
			ElementTypeID: int32Type,
			Address:       200,
			Buffer:        210,
			Length:        2,
			Capacity:      10,
			Stride:        4,
			BufferStart:   9,
		}, nil
	case sliceType:
		// A slice of 2 32-bit-wide elements.
		return debug.Slice{
			Array: debug.Array{
				ElementTypeID: int32Type,
				Address:       300,
				Length:        2,
				StrideBits:    32,
			},
			Capacity: 50,
		}, nil
	case int32Type:
		// We use the address as the value, so that we're testing whether the right
		// address was calculated.
		return int32(v.Address), nil
	}
	return nil, fmt.Errorf("unexpected Value request")
}

func (p *Program) MapElement(m debug.Map, index uint64) (debug.Var, debug.Var, error) {
	return debug.Var{TypeID: int16Type, Address: 1000*index + 400},
		debug.Var{TypeID: int32Type, Address: 1000*index + 404},
		nil
}

func TestLogString(t *testing.T) {
	bp := cd.Breakpoint{
		Action:           "LOG",
		LogMessageFormat: "$0 hello, $$7world! $1 $2 $3 $4 $5$6 $7 $8",
		EvaluatedExpressions: []*cd.Variable{
			{Name: "a", VarTableIndex: 1},
			{Name: "b", VarTableIndex: 2},
			{Name: "c", VarTableIndex: 3},
			{Name: "d", VarTableIndex: 4},
			{Name: "e", VarTableIndex: 5},
			{Name: "f", VarTableIndex: 6},
			{Name: "g", VarTableIndex: 7},
			{Name: "h", VarTableIndex: 8},
			{Name: "i", VarTableIndex: 9},
		},
	}
	varTable := []*cd.Variable{
		{},
		{Value: "1"},
		{Value: `"hello"`},
		{
			Members: []*cd.Variable{
				{Name: "x", Value: "1"},
				{Name: "y", Value: `"hello"`},
				{Name: "z", VarTableIndex: 3},
			},
		},
		{
			Members: []*cd.Variable{
				{VarTableIndex: 1},
			},
			Value: "0x1",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 10},
				{Name: "[1]", VarTableIndex: 11},
				{Name: "[2]", VarTableIndex: 12},
				{Name: "[3]", VarTableIndex: 13},
			},
			Value: "len = 4",
		},
		{Value: `"world"`},
		{
			Members: []*cd.Variable{
				{Name: "⚫", VarTableIndex: 14},
				{Name: "⚫", VarTableIndex: 15},
				{Name: "⚫", VarTableIndex: 16},
			},
			Value: "len = 3",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 17},
				{Name: "[1]", VarTableIndex: 18},
			},
			Value: "len = 2",
		},
		{
			Members: []*cd.Variable{
				{Name: "[0]", VarTableIndex: 19},
				{Name: "[1]", VarTableIndex: 20},
			},
			Value: "len = 2",
		},
		{Value: "100"},
		{Value: "104"},
		{Value: "108"},
		{Value: "112"},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 21},
				{Name: "value", VarTableIndex: 22},
			},
		},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 23},
				{Name: "value", VarTableIndex: 24},
			},
		},
		{
			Members: []*cd.Variable{
				{Name: "key", VarTableIndex: 25},
				{
					Name: "value",
					Status: &cd.StatusMessage{
						Description: &cd.FormatMessage{
							Format:     "$0",
							Parameters: []string{"Not captured"},
						},
						IsError:  true,
						RefersTo: "VARIABLE_NAME",
					},
				},
			},
		},
		{Value: "246"},
		{Value: "210"},
		{Value: "300"},
		{Value: "304"},
		{Value: "400"},
		{Value: "404"},
		{Value: "1400"},
		{Value: "1404"},
		{Value: "2400"},
	}
	s := LogString(bp.LogMessageFormat, bp.EvaluatedExpressions, varTable)
	expected := `LOGPOINT: 1 hello, $7world! "hello" {x:1, y:"hello", z:...} ` +
		`0x1 {100, 104, 108, 112} "world"{400:404, 1400:1404, 2400:(Not captured)} ` +
		`{246, 210} {300, 304}`
	if s != expected {
		t.Errorf("LogString: got %q want %q", s, expected)
	}
}

func TestParseToken(t *testing.T) {
	for _, c := range []struct {
		s   string
		max int
		num int
		n   int
		ok  bool
	}{
		{"", 0, 0, 0, false},
		{".", 0, 0, 0, false},
		{"0", 0, 0, 1, true},
		{"0", 1, 0, 1, true},
		{"00", 0, 0, 2, true},
		{"1.", 1, 1, 1, true},
		{"1.", 0, 0, 0, false},
		{"10", 10, 10, 2, true},
		{"10..", 10, 10, 2, true},
		{"10", 11, 10, 2, true},
		{"10..", 11, 10, 2, true},
		{"10", 9, 0, 0, false},
		{"10..", 9, 0, 0, false},
		{" 10", 10, 0, 0, false},
		{"010", 10, 10, 3, true},
		{"123456789", 123456789, 123456789, 9, true},
		{"123456789", 123456788, 0, 0, false},
		{"123456789123456789123456789", 999999999, 0, 0, false},
	} {
		num, n, ok := parseToken(c.s, c.max)
		if ok != c.ok {
			t.Errorf("parseToken(%q, %d): got ok=%t want ok=%t", c.s, c.max, ok, c.ok)
			continue
		}
		if !ok {
			continue
		}
		if num != c.num || n != c.n {
			t.Errorf("parseToken(%q, %d): got %d,%d,%t want %d,%d,%t", c.s, c.max, num, n, ok, c.num, c.n, c.ok)
		}
	}
}
