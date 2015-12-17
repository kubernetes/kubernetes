package tsdb

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdb/influxdb/influxql"
)

import "sort"

// type testPoint struct {
// 	time   int64
// 	value  interface{}
// 	fields map[string]interface{}
// 	tags   map[string]string
// }

func TestMapMeanNoValues(t *testing.T) {
	if got := MapMean(&MapInput{}); got != nil {
		t.Errorf("output mismatch: exp nil got %v", got)
	}
}

func TestMapMean(t *testing.T) {

	tests := []struct {
		input  *MapInput
		output *meanMapOutput
	}{
		{ // Single point
			input: &MapInput{
				Items: []MapItem{
					{Timestamp: 1, Value: 1.0},
				},
			},
			output: &meanMapOutput{1, 1, Float64Type},
		},
		{ // Two points
			input: &MapInput{
				Items: []MapItem{
					{Timestamp: 1, Value: float64(2.0)},
					{Timestamp: 2, Value: float64(8.0)},
				},
			},
			output: &meanMapOutput{2, 10.0, Float64Type},
		},
	}

	for _, test := range tests {
		got := MapMean(test.input)
		if got == nil {
			t.Fatalf("MapMean(%v): output mismatch: exp %v got %v", test.input, test.output, got)
		}

		if got.(*meanMapOutput).Count != test.output.Count || got.(*meanMapOutput).Total != test.output.Total {
			t.Errorf("output mismatch: exp %v got %v", test.output, got)
		}
	}
}

func TestInitializeMapFuncDerivative(t *testing.T) {

	for _, fn := range []string{"derivative", "non_negative_derivative"} {
		// Single field arg should return MapEcho
		c := &influxql.Call{
			Name: fn,
			Args: []influxql.Expr{
				&influxql.VarRef{Val: " field1"},
				&influxql.DurationLiteral{Val: time.Hour},
			},
		}

		_, err := initializeMapFunc(c)
		if err != nil {
			t.Errorf("InitializeMapFunc(%v) unexpected error.  got %v", c, err)
		}

		// Nested Aggregate func should return the map func for the nested aggregate
		c = &influxql.Call{
			Name: fn,
			Args: []influxql.Expr{
				&influxql.Call{Name: "mean", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}},
				&influxql.DurationLiteral{Val: time.Hour},
			},
		}

		_, err = initializeMapFunc(c)
		if err != nil {
			t.Errorf("InitializeMapFunc(%v) unexpected error.  got %v", c, err)
		}
	}
}

func TestReducePercentileNil(t *testing.T) {

	input := []interface{}{
		nil,
	}

	// ReducePercentile should ignore nil values when calculating the percentile
	got := ReducePercentile(input, 100)
	if got != nil {
		t.Fatalf("ReducePercentile(100) returned wrong type. exp nil got %v", got)
	}
}

func TestMapDistinct(t *testing.T) {
	const ( // prove that we're ignoring time
		timeId1 = iota + 1
		timeId2
		timeId3
		timeId4
		timeId5
		timeId6
	)

	input := &MapInput{
		Items: []MapItem{
			{Timestamp: timeId1, Value: uint64(1)},
			{Timestamp: timeId2, Value: uint64(1)},
			{Timestamp: timeId3, Value: "1"},
			{Timestamp: timeId4, Value: uint64(1)},
			{Timestamp: timeId5, Value: float64(1.0)},
			{Timestamp: timeId6, Value: "1"},
		},
	}

	values := MapDistinct(input).(InterfaceValues)

	if exp, got := 3, len(values); exp != got {
		t.Errorf("Wrong number of values. exp %v got %v", exp, got)
	}

	sort.Sort(values)

	exp := InterfaceValues{
		"1",
		uint64(1),
		float64(1),
	}

	if !reflect.DeepEqual(values, exp) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(exp), spew.Sdump(values))
	}
}

func TestMapDistinctNil(t *testing.T) {
	values := MapDistinct(&MapInput{})

	if values != nil {
		t.Errorf("Wrong values. exp nil got %v", spew.Sdump(values))
	}
}

func TestReduceDistinct(t *testing.T) {
	v1 := InterfaceValues{
		"2",
		"1",
		float64(2.0),
		float64(1),
		uint64(2),
		uint64(1),
		true,
		false,
	}

	expect := InterfaceValues{
		"1",
		"2",
		false,
		true,
		uint64(1),
		float64(1),
		uint64(2),
		float64(2),
	}

	got := ReduceDistinct([]interface{}{v1, v1, expect})

	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(expect), spew.Sdump(got))
	}
}

func TestReduceDistinctNil(t *testing.T) {
	tests := []struct {
		name   string
		values []interface{}
	}{
		{
			name:   "nil values",
			values: nil,
		},
		{
			name:   "nil mapper",
			values: []interface{}{nil},
		},
		{
			name:   "no mappers",
			values: []interface{}{},
		},
		{
			name:   "empty mappper (len 1)",
			values: []interface{}{InterfaceValues{}},
		},
		{
			name:   "empty mappper (len 2)",
			values: []interface{}{InterfaceValues{}, InterfaceValues{}},
		},
	}

	for _, test := range tests {
		t.Log(test.name)
		got := ReduceDistinct(test.values)
		if got != nil {
			t.Errorf("Wrong values. exp nil got %v", spew.Sdump(got))
		}
	}
}

func Test_distinctValues_Sort(t *testing.T) {
	values := InterfaceValues{
		"2",
		"1",
		float64(2.0),
		float64(1),
		uint64(2),
		uint64(1),
		true,
		false,
	}

	expect := InterfaceValues{
		"1",
		"2",
		false,
		true,
		uint64(1),
		float64(1),
		uint64(2),
		float64(2),
	}

	sort.Sort(values)

	if !reflect.DeepEqual(values, expect) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(expect), spew.Sdump(values))
	}
}

func TestMapCountDistinct(t *testing.T) {
	const ( // prove that we're ignoring time
		timeId1 = iota + 1
		timeId2
		timeId3
		timeId4
		timeId5
		timeId6
		timeId7
	)

	input := &MapInput{
		Items: []MapItem{
			{Timestamp: timeId1, Value: uint64(1)},
			{Timestamp: timeId2, Value: uint64(1)},
			{Timestamp: timeId3, Value: "1"},
			{Timestamp: timeId4, Value: uint64(1)},
			{Timestamp: timeId5, Value: float64(1.0)},
			{Timestamp: timeId6, Value: "1"},
			{Timestamp: timeId7, Value: true},
		},
	}

	values := MapCountDistinct(input).(map[interface{}]struct{})

	if exp, got := 4, len(values); exp != got {
		t.Errorf("Wrong number of values. exp %v got %v", exp, got)
	}

	exp := map[interface{}]struct{}{
		uint64(1):  struct{}{},
		float64(1): struct{}{},
		"1":        struct{}{},
		true:       struct{}{},
	}

	if !reflect.DeepEqual(values, exp) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(exp), spew.Sdump(values))
	}
}

func TestMapCountDistinctNil(t *testing.T) {
	if values := MapCountDistinct(&MapInput{}); values != nil {
		t.Errorf("Wrong values. exp nil got %v", spew.Sdump(values))
	}
}

func TestReduceCountDistinct(t *testing.T) {
	v1 := map[interface{}]struct{}{
		"2":          struct{}{},
		"1":          struct{}{},
		float64(2.0): struct{}{},
		float64(1):   struct{}{},
		uint64(2):    struct{}{},
		uint64(1):    struct{}{},
		true:         struct{}{},
		false:        struct{}{},
	}

	v2 := map[interface{}]struct{}{
		uint64(1):  struct{}{},
		float64(1): struct{}{},
		uint64(2):  struct{}{},
		float64(2): struct{}{},
		false:      struct{}{},
		true:       struct{}{},
		"1":        struct{}{},
		"2":        struct{}{},
	}

	exp := 8
	got := ReduceCountDistinct([]interface{}{v1, v1, v2})

	if !reflect.DeepEqual(got, exp) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(exp), spew.Sdump(got))
	}
}

func TestReduceCountDistinctNil(t *testing.T) {
	emptyResults := make(map[interface{}]struct{})
	tests := []struct {
		name   string
		values []interface{}
	}{
		{
			name:   "nil values",
			values: nil,
		},
		{
			name:   "nil mapper",
			values: []interface{}{nil},
		},
		{
			name:   "no mappers",
			values: []interface{}{},
		},
		{
			name:   "empty mappper (len 1)",
			values: []interface{}{emptyResults},
		},
		{
			name:   "empty mappper (len 2)",
			values: []interface{}{emptyResults, emptyResults},
		},
	}

	for _, test := range tests {
		t.Log(test.name)
		got := ReduceCountDistinct(test.values)
		if got != 0 {
			t.Errorf("Wrong values. exp nil got %v", spew.Sdump(got))
		}
	}
}

var getSortedRangeData = []float64{
	60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
	20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
	10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
	50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
	30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
}

var getSortedRangeTests = []struct {
	name     string
	data     []float64
	start    int
	count    int
	expected []float64
}{
	{"first 5", getSortedRangeData, 0, 5, []float64{0, 1, 2, 3, 4}},
	{"0 length", getSortedRangeData, 8, 0, []float64{}},
	{"past end of data", getSortedRangeData, len(getSortedRangeData) - 3, 5, []float64{67, 68, 69}},
}

func TestGetSortedRange(t *testing.T) {
	for _, tt := range getSortedRangeTests {
		results := getSortedRange(tt.data, tt.start, tt.count)
		if len(results) != len(tt.expected) {
			t.Errorf("Test %s error.  Expected getSortedRange to return %v but got %v", tt.name, tt.expected, results)
		}
		for i, testPoint := range tt.expected {
			if testPoint != results[i] {
				t.Errorf("Test %s error. getSortedRange returned wrong result for index %v.  Expected %v but got %v", tt.name, i, testPoint, results[i])
			}
		}
	}
}

var benchGetSortedRangeResults []float64

func BenchmarkGetSortedRangeByPivot(b *testing.B) {
	data := make([]float64, len(getSortedRangeData))
	var results []float64
	for i := 0; i < b.N; i++ {
		copy(data, getSortedRangeData)
		results = getSortedRange(data, 8, 15)
	}
	benchGetSortedRangeResults = results
}

func BenchmarkGetSortedRangeBySort(b *testing.B) {
	data := make([]float64, len(getSortedRangeData))
	var results []float64
	for i := 0; i < b.N; i++ {
		copy(data, getSortedRangeData)
		sort.Float64s(data)
		results = data[8:23]
	}
	benchGetSortedRangeResults = results
}

func TestMapTopBottom(t *testing.T) {
	tests := []struct {
		name  string
		skip  bool
		input *MapInput
		exp   positionOut
		call  *influxql.Call
	}{
		{
			name: "top int64 - basic",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: int64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(53), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top int64 - tie on value, resolve based on time",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 20, Value: int64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(99), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				callArgs: []string{"host"},
				points: PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, int64(99), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top mixed numerics - ints",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: uint64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, uint64(88), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top mixed numerics - ints & floats",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: float64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: uint64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, float64(99), nil, map[string]string{"host": "a"}},
					{20, uint64(88), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top mixed numerics - ints, floats, & strings",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: float64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: "88", Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, float64(99), nil, map[string]string{"host": "a"}},
					{10, int64(53), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top bools",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: true, Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: true, Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: false, Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, true, nil, map[string]string{"host": "a"}},
					{10, true, nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - basic",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: int64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "a"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - tie on value, resolve based on time",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: int64(53), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				callArgs: []string{"host"},
				points: PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "a"}},
					{20, int64(53), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom mixed numerics - ints",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: uint64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "a"}},
					{20, uint64(88), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom mixed numerics - ints & floats",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: int64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: float64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: uint64(88), Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, float64(53), nil, map[string]string{"host": "a"}},
					{20, uint64(88), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom mixed numerics - ints, floats, & strings",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: float64(99), Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: int64(53), Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: "88", Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "a"}},
					{10, float64(99), nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom bools",
			input: &MapInput{
				TMin: -1,
				Items: []MapItem{
					{Timestamp: 10, Value: true, Tags: map[string]string{"host": "a"}},
					{Timestamp: 10, Value: true, Tags: map[string]string{"host": "a"}},
					{Timestamp: 20, Value: false, Tags: map[string]string{"host": "a"}},
				},
			},
			exp: positionOut{
				points: PositionPoints{
					{20, false, nil, map[string]string{"host": "a"}},
					{10, true, nil, map[string]string{"host": "a"}},
				},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
	}

	for _, test := range tests {
		if test.skip {
			continue
		}
		lit, _ := test.call.Args[len(test.call.Args)-1].(*influxql.NumberLiteral)
		limit := int(lit.Val)
		fields := topCallArgs(test.call)

		values := MapTopBottom(test.input, limit, fields, len(test.call.Args), test.call.Name).(PositionPoints)
		t.Logf("Test: %s", test.name)
		if exp, got := len(test.exp.points), len(values); exp != got {
			t.Errorf("Wrong number of values. exp %v got %v", exp, got)
		}
		if !reflect.DeepEqual(values, test.exp.points) {
			t.Errorf("Wrong values. \nexp\n %v\ngot\n %v", spew.Sdump(test.exp.points), spew.Sdump(values))
		}
	}
}

func TestReduceTopBottom(t *testing.T) {
	tests := []struct {
		name   string
		skip   bool
		values []interface{}
		exp    PositionPoints
		call   *influxql.Call
	}{
		{
			name: "top int64 - single map",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(53), nil, map[string]string{"host": "b"}},
				},
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top int64 - double map",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
				PositionPoints{
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(53), nil, map[string]string{"host": "b"}},
				},
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top int64 - double map with nil",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(53), nil, map[string]string{"host": "b"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "top int64 - double map with non-matching tags and tag selected",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, int64(88), nil, map[string]string{}},
					{10, int64(53), nil, map[string]string{"host": "b"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(88), nil, map[string]string{}},
			},
			call: &influxql.Call{Name: "top", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.VarRef{Val: "host"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			skip: true,
			name: "top int64 - double map with non-matching tags",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
					{20, int64(88), nil, map[string]string{}},
					{10, int64(53), nil, map[string]string{"host": "b"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(55), nil, map[string]string{"host": "b"}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - single map",
			values: []interface{}{
				PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "b"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
			},
			exp: PositionPoints{
				PositionPoint{10, int64(53), nil, map[string]string{"host": "b"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - double map",
			values: []interface{}{
				PositionPoints{
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
				PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "b"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
				},
			},
			exp: PositionPoints{
				PositionPoint{10, int64(53), nil, map[string]string{"host": "b"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - double map with nil",
			values: []interface{}{
				PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "b"}},
					{20, int64(88), nil, map[string]string{"host": "a"}},
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(53), nil, map[string]string{"host": "b"}},
				PositionPoint{20, int64(88), nil, map[string]string{"host": "a"}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			name: "bottom int64 - double map with non-matching tags and tag selected",
			values: []interface{}{
				PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "b"}},
					{20, int64(88), nil, map[string]string{}},
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(53), nil, map[string]string{"host": "b"}},
				PositionPoint{20, int64(88), nil, map[string]string{}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.VarRef{Val: "host"}, &influxql.NumberLiteral{Val: 2}}},
		},
		{
			skip: true,
			name: "bottom int64 - double map with non-matching tags",
			values: []interface{}{
				PositionPoints{
					{10, int64(53), nil, map[string]string{"host": "b"}},
					{20, int64(88), nil, map[string]string{}},
					{10, int64(99), nil, map[string]string{"host": "a"}},
				},
				nil,
			},
			exp: PositionPoints{
				PositionPoint{10, int64(99), nil, map[string]string{"host": "a"}},
				PositionPoint{20, int64(55), nil, map[string]string{"host": "b"}},
			},
			call: &influxql.Call{Name: "bottom", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.NumberLiteral{Val: 2}}},
		},
	}

	for _, test := range tests {
		if test.skip {
			continue
		}
		lit, _ := test.call.Args[len(test.call.Args)-1].(*influxql.NumberLiteral)
		limit := int(lit.Val)
		fields := topCallArgs(test.call)
		values := ReduceTopBottom(test.values, limit, fields, test.call.Name)
		t.Logf("Test: %s", test.name)
		if values != nil {
			v, _ := values.(PositionPoints)
			if exp, got := len(test.exp), len(v); exp != got {
				t.Errorf("Wrong number of values. exp %v got %v", exp, got)
			}
		}
		if !reflect.DeepEqual(values, test.exp) {
			t.Errorf("Wrong values. \nexp\n %v\ngot\n %v", spew.Sdump(test.exp), spew.Sdump(values))
		}
	}
}

func TestInitializeUnmarshallerMaxMin(t *testing.T) {
	tests := []struct {
		Name   string
		input  []byte
		output interface{}
		call   *influxql.Call
	}{
		{
			Name:  "max - one point",
			input: []byte(`{"Time":1447729856247384906,"Val":1,"Type":0,"Fields":{"":1},"Tags":{}}`),
			output: PositionPoint{
				Time:   int64(1447729856247384906),
				Value:  float64(1),
				Fields: map[string]interface{}{"": float64(1)},
				Tags:   map[string]string{},
			},
			call: &influxql.Call{Name: "max"},
		},
		{
			Name:   "max - nil point",
			input:  []byte(`null`),
			output: nil,
			call:   &influxql.Call{Name: "max"},
		},
		{
			Name:  "min - one point",
			input: []byte(`{"Time":1447729856247384906,"Val":1,"Type":0,"Fields":{"":1},"Tags":{}}`),
			output: PositionPoint{
				Time:   int64(1447729856247384906),
				Value:  float64(1),
				Fields: map[string]interface{}{"": float64(1)},
				Tags:   map[string]string{},
			},
			call: &influxql.Call{Name: "min"},
		},
		{
			Name:   "min - nil point",
			input:  []byte(`null`),
			output: nil,
			call:   &influxql.Call{Name: "min"},
		},
	}
	for _, test := range tests {
		unmarshaller, err := InitializeUnmarshaller(test.call)
		if err != nil {
			t.Errorf("initialize unmarshaller for %v, got error:%v", test.Name, err)
		}

		// unmarshaller take bytes recieved from remote server as input,
		// unmarshal it into an interface the reducer can use
		unmarshallOutput, err := unmarshaller(test.input)
		if err != nil {
			t.Errorf("unmarshaller unmarshal %v fail with error:%v", &test.input, err)
		}

		//if input is "null" then the unmarshal output is expect to be nil
		if string(test.input) == "null" && unmarshallOutput != nil {
			t.Errorf("initialize unmarshaller, \nexp\n %v\ngot\n %v", nil, spew.Sdump(unmarshallOutput))
			continue
		}

		// initialize a reducer that can take the output of unmarshaller as input
		reducer, err := initializeReduceFunc(test.call)
		if err != nil {
			t.Errorf("initialize %v reduce function fail with error:%v", test.Name, err)
		}

		output := reducer([]interface{}{unmarshallOutput})
		if !reflect.DeepEqual(output, test.output) {
			t.Errorf("Wrong output. \nexp\n %v\ngot\n %v", spew.Sdump(test.output), spew.Sdump(output))
		}
	}
}

func TestInitializeUnmarshallerTopBottom(t *testing.T) {
	tests := []struct {
		Name   string
		input  []byte
		output interface{}
		call   *influxql.Call
	}{
		{
			Name:  "top - one point",
			input: []byte(`[{"Time":1447729856247384906,"Value":1,"Fields":{"":1},"Tags":{}}]`),
			output: PositionPoints{
				{int64(1447729856247384906), float64(1), map[string]interface{}{"": float64(1)}, map[string]string{}},
			},
			call: &influxql.Call{
				Name: "top",
				Args: []influxql.Expr{
					&influxql.VarRef{Val: "field1"},
					&influxql.NumberLiteral{Val: 1},
				},
			},
		},
		{
			Name:  "bottom - one point",
			input: []byte(`[{"Time":1447729856247384906,"Value":1,"Fields":{"":1},"Tags":{}}]`),
			output: PositionPoints{
				{int64(1447729856247384906), float64(1), map[string]interface{}{"": float64(1)}, map[string]string{}},
			},
			call: &influxql.Call{
				Name: "bottom",
				Args: []influxql.Expr{
					&influxql.VarRef{Val: "field1"},
					&influxql.NumberLiteral{Val: 1},
				},
			},
		},
	}
	for _, test := range tests {
		unmarshaller, err := InitializeUnmarshaller(test.call)
		if err != nil {
			t.Errorf("initialize unmarshaller for %v, got error:%v", test.Name, err)
		}

		// unmarshaller take bytes recieved from remote server as input,
		// unmarshal it into an interface the reducer can use
		output, err := unmarshaller(test.input)
		if err != nil {
			t.Errorf("unmarshaller unmarshal %v fail with error:%v", &test.input, err)
		}

		if !reflect.DeepEqual(output, test.output) {
			t.Errorf("Wrong output. \nexp\n %v\ngot\n %v", spew.Sdump(test.output), spew.Sdump(output))
		}
	}
}
