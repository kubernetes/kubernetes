package influxql

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
)

import "sort"

type point struct {
	seriesKey string
	time      int64
	value     interface{}
}

type testIterator struct {
	values []point
}

func (t *testIterator) Next() (timestamp int64, value interface{}) {
	if len(t.values) > 0 {
		v := t.values[0]
		t.values = t.values[1:]
		return v.time, v.value
	}

	return -1, nil
}

func TestMapMeanNoValues(t *testing.T) {
	iter := &testIterator{}
	if got := MapMean(iter); got != nil {
		t.Errorf("output mismatch: exp nil got %v", got)
	}
}

func TestMapMean(t *testing.T) {

	tests := []struct {
		input  []point
		output *meanMapOutput
	}{
		{ // Single point
			input:  []point{point{"0", 1, 1.0}},
			output: &meanMapOutput{1, 1, Float64Type},
		},
		{ // Two points
			input: []point{
				point{"0", 1, 2.0},
				point{"0", 2, 8.0},
			},
			output: &meanMapOutput{2, 5.0, Float64Type},
		},
	}

	for _, test := range tests {
		iter := &testIterator{
			values: test.input,
		}

		got := MapMean(iter)
		if got == nil {
			t.Fatalf("MapMean(%v): output mismatch: exp %v got %v", test.input, test.output, got)
		}

		if got.(*meanMapOutput).Count != test.output.Count || got.(*meanMapOutput).Mean != test.output.Mean {
			t.Errorf("output mismatch: exp %v got %v", test.output, got)
		}
	}
}
func TestInitializeMapFuncPercentile(t *testing.T) {
	// No args
	c := &Call{
		Name: "percentile",
		Args: []Expr{},
	}
	_, err := InitializeMapFunc(c)
	if err == nil {
		t.Errorf("InitializeMapFunc(%v) expected error. got nil", c)
	}

	if exp := "expected two arguments for percentile()"; err.Error() != exp {
		t.Errorf("InitializeMapFunc(%v) mismatch. exp %v got %v", c, exp, err.Error())
	}

	// No percentile arg
	c = &Call{
		Name: "percentile",
		Args: []Expr{
			&VarRef{Val: "field1"},
		},
	}

	_, err = InitializeMapFunc(c)
	if err == nil {
		t.Errorf("InitializeMapFunc(%v) expected error. got nil", c)
	}

	if exp := "expected two arguments for percentile()"; err.Error() != exp {
		t.Errorf("InitializeMapFunc(%v) mismatch. exp %v got %v", c, exp, err.Error())
	}
}

func TestInitializeMapFuncDerivative(t *testing.T) {

	for _, fn := range []string{"derivative", "non_negative_derivative"} {
		// No args should fail
		c := &Call{
			Name: fn,
			Args: []Expr{},
		}

		_, err := InitializeMapFunc(c)
		if err == nil {
			t.Errorf("InitializeMapFunc(%v) expected error.  got nil", c)
		}

		// Single field arg should return MapEcho
		c = &Call{
			Name: fn,
			Args: []Expr{
				&VarRef{Val: " field1"},
				&DurationLiteral{Val: time.Hour},
			},
		}

		_, err = InitializeMapFunc(c)
		if err != nil {
			t.Errorf("InitializeMapFunc(%v) unexpected error.  got %v", c, err)
		}

		// Nested Aggregate func should return the map func for the nested aggregate
		c = &Call{
			Name: fn,
			Args: []Expr{
				&Call{Name: "mean", Args: []Expr{&VarRef{Val: "field1"}}},
				&DurationLiteral{Val: time.Hour},
			},
		}

		_, err = InitializeMapFunc(c)
		if err != nil {
			t.Errorf("InitializeMapFunc(%v) unexpected error.  got %v", c, err)
		}
	}
}

func TestInitializeReduceFuncPercentile(t *testing.T) {
	// No args
	c := &Call{
		Name: "percentile",
		Args: []Expr{},
	}
	_, err := InitializeReduceFunc(c)
	if err == nil {
		t.Errorf("InitializedReduceFunc(%v) expected error. got nil", c)
	}

	if exp := "expected float argument in percentile()"; err.Error() != exp {
		t.Errorf("InitializedReduceFunc(%v) mismatch. exp %v got %v", c, exp, err.Error())
	}

	// No percentile arg
	c = &Call{
		Name: "percentile",
		Args: []Expr{
			&VarRef{Val: "field1"},
		},
	}

	_, err = InitializeReduceFunc(c)
	if err == nil {
		t.Errorf("InitializedReduceFunc(%v) expected error. got nil", c)
	}

	if exp := "expected float argument in percentile()"; err.Error() != exp {
		t.Errorf("InitializedReduceFunc(%v) mismatch. exp %v got %v", c, exp, err.Error())
	}
}

func TestReducePercentileNil(t *testing.T) {

	// ReducePercentile should ignore nil values when calculating the percentile
	fn := ReducePercentile(100)
	input := []interface{}{
		nil,
	}

	got := fn(input)
	if got != nil {
		t.Fatalf("ReducePercentile(100) returned wrong type. exp nil got %v", got)
	}
}

func TestMapDistinct(t *testing.T) {
	const ( // prove that we're ignoring seriesKey
		seriesKey1 = "1"
		seriesKey2 = "2"
	)

	const ( // prove that we're ignoring time
		timeId1 = iota + 1
		timeId2
		timeId3
		timeId4
		timeId5
		timeId6
	)

	iter := &testIterator{
		values: []point{
			{seriesKey1, timeId1, uint64(1)},
			{seriesKey1, timeId2, uint64(1)},
			{seriesKey1, timeId3, "1"},
			{seriesKey2, timeId4, uint64(1)},
			{seriesKey2, timeId5, float64(1.0)},
			{seriesKey2, timeId6, "1"},
		},
	}

	values := MapDistinct(iter).(distinctValues)

	if exp, got := 3, len(values); exp != got {
		t.Errorf("Wrong number of values. exp %v got %v", exp, got)
	}

	sort.Sort(values)

	exp := distinctValues{
		uint64(1),
		float64(1),
		"1",
	}

	if !reflect.DeepEqual(values, exp) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(exp), spew.Sdump(values))
	}
}

func TestMapDistinctNil(t *testing.T) {
	iter := &testIterator{
		values: []point{},
	}

	values := MapDistinct(iter)

	if values != nil {
		t.Errorf("Wrong values. exp nil got %v", spew.Sdump(values))
	}
}

func TestReduceDistinct(t *testing.T) {
	v1 := distinctValues{
		"2",
		"1",
		float64(2.0),
		float64(1),
		uint64(2),
		uint64(1),
		true,
		false,
	}

	expect := distinctValues{
		uint64(1),
		float64(1),
		uint64(2),
		float64(2),
		false,
		true,
		"1",
		"2",
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
			values: []interface{}{distinctValues{}},
		},
		{
			name:   "empty mappper (len 2)",
			values: []interface{}{distinctValues{}, distinctValues{}},
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
	values := distinctValues{
		"2",
		"1",
		float64(2.0),
		float64(1),
		uint64(2),
		uint64(1),
		true,
		false,
	}

	expect := distinctValues{
		uint64(1),
		float64(1),
		uint64(2),
		float64(2),
		false,
		true,
		"1",
		"2",
	}

	sort.Sort(values)

	if !reflect.DeepEqual(values, expect) {
		t.Errorf("Wrong values. exp %v got %v", spew.Sdump(expect), spew.Sdump(values))
	}
}

func TestMapCountDistinct(t *testing.T) {
	const ( // prove that we're ignoring seriesKey
		seriesKey1 = "1"
		seriesKey2 = "2"
	)

	const ( // prove that we're ignoring time
		timeId1 = iota + 1
		timeId2
		timeId3
		timeId4
		timeId5
		timeId6
		timeId7
	)

	iter := &testIterator{
		values: []point{
			{seriesKey1, timeId1, uint64(1)},
			{seriesKey1, timeId2, uint64(1)},
			{seriesKey1, timeId3, "1"},
			{seriesKey2, timeId4, uint64(1)},
			{seriesKey2, timeId5, float64(1.0)},
			{seriesKey2, timeId6, "1"},
			{seriesKey2, timeId7, true},
		},
	}

	values := MapCountDistinct(iter).(map[interface{}]struct{})

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
	iter := &testIterator{
		values: []point{},
	}

	values := MapCountDistinct(iter)

	if values != nil {
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
		for i, point := range tt.expected {
			if point != results[i] {
				t.Errorf("Test %s error. getSortedRange returned wrong result for index %v.  Expected %v but got %v", tt.name, i, point, results[i])
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
