package influxql_test

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/pkg/deep"
)

// Second represents a helper for type converting durations.
const Second = int64(time.Second)

// Ensure a SELECT min() query can be executed.
func TestSelect_Min(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		if !reflect.DeepEqual(opt.Expr, MustParseExpr(`min(value)`)) {
			t.Fatalf("unexpected expr: %s", spew.Sdump(opt.Expr))
		}

		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT min(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100, Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT distinct() query can be executed.
func TestSelect_Distinct_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: 2},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT distinct(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT distinct() query can be executed.
func TestSelect_Distinct_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: 2},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT distinct(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT distinct() query can be executed.
func TestSelect_Distinct_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &StringIterator{Points: []influxql.StringPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: "a"},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: "b"},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: "c"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: "b"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: "d"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: "d"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: "d"},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT distinct(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: "a"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: "b"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: "c"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: "d"}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT distinct() query can be executed.
func TestSelect_Distinct_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &BooleanIterator{Points: []influxql.BooleanPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: true},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT distinct(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: true}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: true}},
	}) {
		t.Errorf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mean() query can be executed.
func TestSelect_Mean_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 3.2, Aggregated: 5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mean() query can be executed.
func TestSelect_Mean_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 3.2, Aggregated: 5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mean() query cannot be executed on strings.
func TestSelect_Mean_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&StringIterator{}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err == nil || err.Error() != "unsupported mean iterator type: *influxql_test.StringIterator" {
		t.Errorf("unexpected error: %s", err)
	}

	if itrs != nil {
		influxql.Iterators(itrs).Close()
	}
}

// Ensure a SELECT mean() query cannot be executed on booleans.
func TestSelect_Mean_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&BooleanIterator{}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err == nil || err.Error() != "unsupported mean iterator type: *influxql_test.BooleanIterator" {
		t.Errorf("unexpected error: %s", err)
	}

	if itrs != nil {
		influxql.Iterators(itrs).Close()
	}
}

// Ensure a SELECT median() query can be executed.
func TestSelect_Median_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT median(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19.5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2.5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT median() query can be executed.
func TestSelect_Median_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT median(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19.5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2.5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT median() query cannot be executed on strings.
func TestSelect_Median_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &StringIterator{}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT median(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err == nil || err.Error() != "unsupported median iterator type: *influxql_test.StringIterator" {
		t.Errorf("unexpected error: %s", err)
	}

	if itrs != nil {
		influxql.Iterators(itrs).Close()
	}
}

// Ensure a SELECT median() query cannot be executed on booleans.
func TestSelect_Median_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &BooleanIterator{}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT median(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err == nil || err.Error() != "unsupported median iterator type: *influxql_test.BooleanIterator" {
		t.Errorf("unexpected error: %s", err)
	}

	if itrs != nil {
		influxql.Iterators(itrs).Close()
	}
}

// Ensure a SELECT mode() query can be executed.
func TestSelect_Mode_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mode(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mode() query can be executed.
func TestSelect_Mode_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 54 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mode(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mode() query cannot be executed on strings.
func TestSelect_Mode_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &StringIterator{Points: []influxql.StringPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: "a"},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: "a"},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: "cxxx"},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 6 * Second, Value: "zzzz"},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 7 * Second, Value: "zzzz"},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 8 * Second, Value: "zxxx"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: "b"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: "d"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: "d"},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: "d"},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mode(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: "a"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: "zzzz"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: "d"}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT mode() query cannot be executed on booleans.
func TestSelect_Mode_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &BooleanIterator{Points: []influxql.BooleanPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 2 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 6 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: true},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mode(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected point: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: true}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: true}},
	}) {
		t.Errorf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed.
func TestSelect_Top_NoTags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed.
func TestSelect_Top_NoTags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 5}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed with tags.
func TestSelect_Top_Tags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value::float, host::tag, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "B"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 30 * Second, Value: 5, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "B"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed with tags.
func TestSelect_Top_Tags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value::integer, host::tag, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 10, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "B"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 30 * Second, Value: 5, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "B"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed with tags and group by.
func TestSelect_Top_GroupByTags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value::float, host::tag, 1) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY region, time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: 19, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: "A"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT top() query can be executed with tags and group by.
func TestSelect_Top_GroupByTags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT top(value::integer, host::tag, 1) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY region, time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: 19, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: "A"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed.
func TestSelect_Bottom_NoTags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::float, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 3}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed.
func TestSelect_Bottom_NoTags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::integer, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 2}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 30 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed with tags.
func TestSelect_Bottom_Tags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::float, host::tag, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 2, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "B"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 30 * Second, Value: 1, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "B"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "A"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed with tags.
func TestSelect_Bottom_Tags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::integer, host::tag, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 2, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 10, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 0 * Second, Value: "B"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 30 * Second, Value: 1, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "B"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Time: 30 * Second, Value: 100, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Time: 30 * Second, Value: "A"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed with tags and group by.
func TestSelect_Bottom_GroupByTags_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::float, host::tag, 1) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY region, time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: 2, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: 3, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: 1, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: "B"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT bottom() query can be executed with tags and group by.
func TestSelect_Bottom_GroupByTags_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3, Aux: []interface{}{"A"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100, Aux: []interface{}{"A"}},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4, Aux: []interface{}{"B"}},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5, Aux: []interface{}{"B"}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT bottom(value::float, host::tag, 1) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY region, time(30s) fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: 2, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=east"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: 3, Aux: []interface{}{"A"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 0 * Second, Value: "A"},
		},
		{
			&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: 1, Aux: []interface{}{"B"}},
			&influxql.StringPoint{Name: "cpu", Tags: ParseTags("region=west"), Time: 30 * Second, Value: "B"},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT query with a fill(null) statement can be executed.
func TestSelect_Fill_Null_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 2},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(null)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Nil: true}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT query with a fill(<number>) statement can be executed.
func TestSelect_Fill_Number_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 2},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(1)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Value: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT query with a fill(previous) statement can be executed.
func TestSelect_Fill_Previous_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 2},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(previous)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Value: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT query with a fill(linear) statement can be executed.
func TestSelect_Fill_Linear_Float_One(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 32 * Second, Value: 4},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(linear)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 3}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 4, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Nil: true}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Fill_Linear_Float_Many(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 62 * Second, Value: 7},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT mean(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(linear)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 3}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 4}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Value: 5}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Value: 6}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 60 * Second, Value: 7, Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT query with a fill(linear) statement can be executed for integers.
func TestSelect_Fill_Linear_Integer_One(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 32 * Second, Value: 4},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT max(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:00Z' GROUP BY host, time(10s) fill(linear)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 1, Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 2}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 4, Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Nil: true}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Nil: true}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Fill_Linear_Integer_Many(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 12 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("host=A"), Time: 72 * Second, Value: 10},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT max(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:01:20Z' GROUP BY host, time(10s) fill(linear)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Nil: true}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 1, Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 20 * Second, Value: 2}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 40 * Second, Value: 5}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 50 * Second, Value: 7}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 60 * Second, Value: 8}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 70 * Second, Value: 10, Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT stddev() query can be executed.
func TestSelect_Stddev_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT stddev(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 0.7071067811865476}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 0.7071067811865476}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 1.5811388300841898}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT stddev() query can be executed.
func TestSelect_Stddev_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT stddev(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 0.7071067811865476}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 0.7071067811865476}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Nil: true}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 1.5811388300841898}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT spread() query can be executed.
func TestSelect_Spread_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT spread(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 0}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 0}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT spread() query can be executed.
func TestSelect_Spread_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 1},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 5},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT spread(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 0}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 0}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT percentile() query can be executed.
func TestSelect_Percentile_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 9},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 8},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 7},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 54 * Second, Value: 6},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 55 * Second, Value: 5},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 56 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 57 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 58 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 59 * Second, Value: 1},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT percentile(value, 90) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 3}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 9}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT percentile() query can be executed.
func TestSelect_Percentile_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},

			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 50 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 51 * Second, Value: 9},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 52 * Second, Value: 8},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 53 * Second, Value: 7},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 54 * Second, Value: 6},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 55 * Second, Value: 5},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 56 * Second, Value: 4},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 57 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 58 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 59 * Second, Value: 1},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT percentile(value, 90) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 50 * Second, Value: 9}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT sample() query can be executed.
func TestSelect_Sample_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 10 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 15 * Second, Value: 2},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT sample(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 5 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 10 * Second, Value: 19}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 15 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT sample() query can be executed.
func TestSelect_Sample_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 10 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 15 * Second, Value: 2},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT sample(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 5 * Second, Value: 10}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 10 * Second, Value: 19}},
		{&influxql.IntegerPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 15 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT sample() query can be executed.
func TestSelect_Sample_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &BooleanIterator{Points: []influxql.BooleanPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: true},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 5 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 10 * Second, Value: false},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 15 * Second, Value: true},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT sample(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: true}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 5 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 10 * Second, Value: false}},
		{&influxql.BooleanPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 15 * Second, Value: true}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT sample() query can be executed.
func TestSelect_Sample_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &StringIterator{Points: []influxql.StringPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: "a"},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 5 * Second, Value: "b"},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 10 * Second, Value: "c"},
			{Name: "cpu", Tags: ParseTags("region=east,host=B"), Time: 15 * Second, Value: "d"},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT sample(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: "a"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 5 * Second, Value: "b"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 10 * Second, Value: "c"}},
		{&influxql.StringPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 15 * Second, Value: "d"}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a simple raw SELECT statement can be executed.
func TestSelect_Raw(t *testing.T) {
	// Mock two iterators -- one for each value in the query.
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		if !reflect.DeepEqual(opt.Aux, []influxql.VarRef{{Val: "v1", Type: influxql.Float}, {Val: "v2", Type: influxql.Float}}) {
			t.Fatalf("unexpected options: %s", spew.Sdump(opt.Expr))

		}
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Time: 0, Aux: []interface{}{float64(1), nil}},
			{Time: 1, Aux: []interface{}{nil, float64(2)}},
			{Time: 5, Aux: []interface{}{float64(3), float64(4)}},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT v1::float, v2::float FROM cpu`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{
			&influxql.FloatPoint{Time: 0, Value: 1},
			&influxql.FloatPoint{Time: 0, Nil: true},
		},
		{
			&influxql.FloatPoint{Time: 1, Nil: true},
			&influxql.FloatPoint{Time: 1, Value: 2},
		},
		{
			&influxql.FloatPoint{Time: 5, Value: 3},
			&influxql.FloatPoint{Time: 5, Value: 4},
		},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure a SELECT binary expr queries can be executed as floats.
func TestSelect_BinaryExpr_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		makeAuxFields := func(value float64) []interface{} {
			aux := make([]interface{}, len(opt.Aux))
			for i := range aux {
				aux[i] = value
			}
			return aux
		}
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20, Aux: makeAuxFields(20)},
			{Name: "cpu", Time: 5 * Second, Value: 10, Aux: makeAuxFields(10)},
			{Name: "cpu", Time: 9 * Second, Value: 19, Aux: makeAuxFields(19)},
		}}, nil
	}
	ic.FieldDimensionsFn = func(sources influxql.Sources) (map[string]influxql.DataType, map[string]struct{}, error) {
		return map[string]influxql.DataType{"value": influxql.Float}, nil, nil
	}

	for _, test := range []struct {
		Name      string
		Statement string
		Points    [][]influxql.Point
	}{
		{
			Name:      "rhs binary add number",
			Statement: `SELECT value + 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "rhs binary add integer",
			Statement: `SELECT value + 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "lhs binary add number",
			Statement: `SELECT 2.0 + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "lhs binary add integer",
			Statement: `SELECT 2 + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "two variable binary add",
			Statement: `SELECT value + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "rhs binary multiply number",
			Statement: `SELECT value * 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "rhs binary multiply integer",
			Statement: `SELECT value * 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "lhs binary multiply number",
			Statement: `SELECT 2.0 * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "lhs binary multiply integer",
			Statement: `SELECT 2 * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "two variable binary multiply",
			Statement: `SELECT value * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 400}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 100}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 361}},
			},
		},
		{
			Name:      "rhs binary subtract number",
			Statement: `SELECT value - 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 17}},
			},
		},
		{
			Name:      "rhs binary subtract integer",
			Statement: `SELECT value - 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 17}},
			},
		},
		{
			Name:      "lhs binary subtract number",
			Statement: `SELECT 2.0 - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: -18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: -8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: -17}},
			},
		},
		{
			Name:      "lhs binary subtract integer",
			Statement: `SELECT 2 - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: -18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: -8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: -17}},
			},
		},
		{
			Name:      "two variable binary subtract",
			Statement: `SELECT value - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 0}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 0}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 0}},
			},
		},
		{
			Name:      "rhs binary division number",
			Statement: `SELECT value / 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: float64(19) / 2}},
			},
		},
		{
			Name:      "rhs binary division integer",
			Statement: `SELECT value / 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: float64(19) / 2}},
			},
		},
		{
			Name:      "lhs binary division number",
			Statement: `SELECT 38.0 / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1.9}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 3.8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 2}},
			},
		},
		{
			Name:      "lhs binary division integer",
			Statement: `SELECT 38 / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1.9}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 3.8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 2}},
			},
		},
		{
			Name:      "two variable binary division",
			Statement: `SELECT value / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 1}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 1}},
			},
		},
	} {
		stmt, err := MustParseSelectStatement(test.Statement).RewriteFields(&ic)
		if err != nil {
			t.Errorf("%s: rewrite error: %s", test.Name, err)
		}

		itrs, err := influxql.Select(stmt, &ic, nil)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.Name, err)
		} else if a, err := Iterators(itrs).ReadAll(); err != nil {
			t.Fatalf("%s: unexpected error: %s", test.Name, err)
		} else if !deep.Equal(a, test.Points) {
			t.Errorf("%s: unexpected points: %s", test.Name, spew.Sdump(a))
		}
	}
}

// Ensure a SELECT binary expr queries can be executed as integers.
func TestSelect_BinaryExpr_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		makeAuxFields := func(value int64) []interface{} {
			aux := make([]interface{}, len(opt.Aux))
			for i := range aux {
				aux[i] = value
			}
			return aux
		}
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20, Aux: makeAuxFields(20)},
			{Name: "cpu", Time: 5 * Second, Value: 10, Aux: makeAuxFields(10)},
			{Name: "cpu", Time: 9 * Second, Value: 19, Aux: makeAuxFields(19)},
		}}, nil
	}
	ic.FieldDimensionsFn = func(sources influxql.Sources) (map[string]influxql.DataType, map[string]struct{}, error) {
		return map[string]influxql.DataType{"value": influxql.Integer}, nil, nil
	}

	for _, test := range []struct {
		Name      string
		Statement string
		Points    [][]influxql.Point
	}{
		{
			Name:      "rhs binary add number",
			Statement: `SELECT value + 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "rhs binary add integer",
			Statement: `SELECT value + 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "lhs binary add number",
			Statement: `SELECT 2.0 + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "lhs binary add integer",
			Statement: `SELECT 2 + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 22}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 12}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 21}},
			},
		},
		{
			Name:      "two variable binary add",
			Statement: `SELECT value + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "rhs binary multiply number",
			Statement: `SELECT value * 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "rhs binary multiply integer",
			Statement: `SELECT value * 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "lhs binary multiply number",
			Statement: `SELECT 2.0 * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "lhs binary multiply integer",
			Statement: `SELECT 2 * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 40}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 20}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 38}},
			},
		},
		{
			Name:      "two variable binary multiply",
			Statement: `SELECT value * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 400}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 100}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 361}},
			},
		},
		{
			Name:      "rhs binary subtract number",
			Statement: `SELECT value - 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 17}},
			},
		},
		{
			Name:      "rhs binary subtract integer",
			Statement: `SELECT value - 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 18}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 8}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 17}},
			},
		},
		{
			Name:      "lhs binary subtract number",
			Statement: `SELECT 2.0 - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: -18}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: -8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: -17}},
			},
		},
		{
			Name:      "lhs binary subtract integer",
			Statement: `SELECT 2 - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: -18}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: -8}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: -17}},
			},
		},
		{
			Name:      "two variable binary subtract",
			Statement: `SELECT value - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 0}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 5 * Second, Value: 0}},
				{&influxql.IntegerPoint{Name: "cpu", Time: 9 * Second, Value: 0}},
			},
		},
		{
			Name:      "rhs binary division number",
			Statement: `SELECT value / 2.0 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 9.5}},
			},
		},
		{
			Name:      "rhs binary division integer",
			Statement: `SELECT value / 2 FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: float64(19) / 2}},
			},
		},
		{
			Name:      "lhs binary division number",
			Statement: `SELECT 38.0 / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1.9}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 3.8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 2.0}},
			},
		},
		{
			Name:      "lhs binary division integer",
			Statement: `SELECT 38 / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1.9}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 3.8}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 2}},
			},
		},
		{
			Name:      "two variable binary division",
			Statement: `SELECT value / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 1}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 1}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 1}},
			},
		},
	} {
		stmt, err := MustParseSelectStatement(test.Statement).RewriteFields(&ic)
		if err != nil {
			t.Errorf("%s: rewrite error: %s", test.Name, err)
		}

		itrs, err := influxql.Select(stmt, &ic, nil)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.Name, err)
		} else if a, err := Iterators(itrs).ReadAll(); err != nil {
			t.Fatalf("%s: unexpected error: %s", test.Name, err)
		} else if !deep.Equal(a, test.Points) {
			t.Errorf("%s: unexpected points: %s", test.Name, spew.Sdump(a))
		}
	}
}

// Ensure a SELECT binary expr queries can be executed on mixed iterators.
func TestSelect_BinaryExpr_Mixed(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20, Aux: []interface{}{float64(20), int64(10)}},
			{Name: "cpu", Time: 5 * Second, Value: 10, Aux: []interface{}{float64(10), int64(15)}},
			{Name: "cpu", Time: 9 * Second, Value: 19, Aux: []interface{}{float64(19), int64(5)}},
		}}, nil
	}
	ic.FieldDimensionsFn = func(sources influxql.Sources) (map[string]influxql.DataType, map[string]struct{}, error) {
		return map[string]influxql.DataType{
			"total": influxql.Float,
			"value": influxql.Integer,
		}, nil, nil
	}

	for _, test := range []struct {
		Name      string
		Statement string
		Points    [][]influxql.Point
	}{
		{
			Name:      "mixed binary add",
			Statement: `SELECT total + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 30}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 25}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 24}},
			},
		},
		{
			Name:      "mixed binary subtract",
			Statement: `SELECT total - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 10}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: -5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 14}},
			},
		},
		{
			Name:      "mixed binary multiply",
			Statement: `SELECT total * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 200}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 150}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: 95}},
			},
		},
		{
			Name:      "mixed binary division",
			Statement: `SELECT total / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 2}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: float64(10) / float64(15)}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Value: float64(19) / float64(5)}},
			},
		},
	} {
		stmt, err := MustParseSelectStatement(test.Statement).RewriteFields(&ic)
		if err != nil {
			t.Errorf("%s: rewrite error: %s", test.Name, err)
		}

		itrs, err := influxql.Select(stmt, &ic, nil)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.Name, err)
		} else if a, err := Iterators(itrs).ReadAll(); err != nil {
			t.Fatalf("%s: unexpected error: %s", test.Name, err)
		} else if !deep.Equal(a, test.Points) {
			t.Errorf("%s: unexpected points: %s", test.Name, spew.Sdump(a))
		}
	}
}

// Ensure a SELECT binary expr with nil values can be executed.
// Nil values may be present when a field is missing from one iterator,
// but not the other.
func TestSelect_BinaryExpr_NilValues(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20, Aux: []interface{}{float64(20), nil}},
			{Name: "cpu", Time: 5 * Second, Value: 10, Aux: []interface{}{float64(10), float64(15)}},
			{Name: "cpu", Time: 9 * Second, Value: 19, Aux: []interface{}{nil, float64(5)}},
		}}, nil
	}
	ic.FieldDimensionsFn = func(sources influxql.Sources) (map[string]influxql.DataType, map[string]struct{}, error) {
		return map[string]influxql.DataType{
			"total": influxql.Float,
			"value": influxql.Float,
		}, nil, nil
	}

	for _, test := range []struct {
		Name      string
		Statement string
		Points    [][]influxql.Point
	}{
		{
			Name:      "nil binary add",
			Statement: `SELECT total + value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Nil: true}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 25}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Nil: true}},
			},
		},
		{
			Name:      "nil binary subtract",
			Statement: `SELECT total - value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Nil: true}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: -5}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Nil: true}},
			},
		},
		{
			Name:      "nil binary multiply",
			Statement: `SELECT total * value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Nil: true}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: 150}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Nil: true}},
			},
		},
		{
			Name:      "nil binary division",
			Statement: `SELECT total / value FROM cpu`,
			Points: [][]influxql.Point{
				{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Nil: true}},
				{&influxql.FloatPoint{Name: "cpu", Time: 5 * Second, Value: float64(10) / float64(15)}},
				{&influxql.FloatPoint{Name: "cpu", Time: 9 * Second, Nil: true}},
			},
		},
	} {
		stmt, err := MustParseSelectStatement(test.Statement).RewriteFields(&ic)
		if err != nil {
			t.Errorf("%s: rewrite error: %s", test.Name, err)
		}

		itrs, err := influxql.Select(stmt, &ic, nil)
		if err != nil {
			t.Errorf("%s: parse error: %s", test.Name, err)
		} else if a, err := Iterators(itrs).ReadAll(); err != nil {
			t.Fatalf("%s: unexpected error: %s", test.Name, err)
		} else if !deep.Equal(a, test.Points) {
			t.Errorf("%s: unexpected points: %s", test.Name, spew.Sdump(a))
		}
	}
}

// Ensure a SELECT (...) query can be executed.
func TestSelect_ParenExpr(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		if !reflect.DeepEqual(opt.Expr, MustParseExpr(`min(value)`)) {
			t.Fatalf("unexpected expr: %s", spew.Sdump(opt.Expr))
		}

		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 11 * Second, Value: 3},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 31 * Second, Value: 100},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT (min(value)) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10, Aggregated: 1}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 30 * Second, Value: 100, Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}

	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 0 * Second, Value: 20},
			{Name: "cpu", Tags: ParseTags("region=west,host=A"), Time: 1 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=west,host=B"), Time: 5 * Second, Value: 10},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 9 * Second, Value: 19},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 10 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 11 * Second, Value: 2},
			{Name: "cpu", Tags: ParseTags("region=east,host=A"), Time: 12 * Second, Value: 2},
		}}, nil
	}

	// Execute selection.
	itrs, err = influxql.Select(MustParseSelectStatement(`SELECT (distinct(value)) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-02T00:00:00Z' GROUP BY time(10s), host fill(none)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 0 * Second, Value: 19}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=B"), Time: 0 * Second, Value: 10}},
		{&influxql.FloatPoint{Name: "cpu", Tags: ParseTags("host=A"), Time: 10 * Second, Value: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.5}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 2.25}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: -4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.5}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 2.25}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: -4}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Desc_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 12 * Second, Value: 3},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 0 * Second, Value: 20},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z' ORDER BY desc`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Errorf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.25}},
		{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 2.5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Desc_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 12 * Second, Value: 3},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 0 * Second, Value: 20},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z' ORDER BY desc`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Errorf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.25}},
		{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 2.5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Duplicate_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Derivative_Duplicate_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT derivative(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -2.5}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Difference_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT difference(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -10}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 9}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: -16}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Difference_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT difference(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: -10}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 9}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 12 * Second, Value: -16}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Difference_Duplicate_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT difference(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: -10}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Difference_Duplicate_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT difference(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: -10}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Elapsed_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 11 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT elapsed(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 11 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Elapsed_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 11 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT elapsed(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 11 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Elapsed_String(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &StringIterator{Points: []influxql.StringPoint{
			{Name: "cpu", Time: 0 * Second, Value: "a"},
			{Name: "cpu", Time: 4 * Second, Value: "b"},
			{Name: "cpu", Time: 8 * Second, Value: "c"},
			{Name: "cpu", Time: 11 * Second, Value: "d"},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT elapsed(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 11 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_Elapsed_Boolean(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &BooleanIterator{Points: []influxql.BooleanPoint{
			{Name: "cpu", Time: 0 * Second, Value: true},
			{Name: "cpu", Time: 4 * Second, Value: false},
			{Name: "cpu", Time: 8 * Second, Value: false},
			{Name: "cpu", Time: 11 * Second, Value: true},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT elapsed(value, 1s) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 4}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 11 * Second, Value: 3}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_MovingAverage_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT moving_average(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: 15, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 14.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: 11, Aggregated: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_MovingAverage_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT moving_average(value, 2) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: 15, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 14.5, Aggregated: 2}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: 11, Aggregated: 2}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_CumulativeSum_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT cumulative_sum(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: 30}},
		{&influxql.FloatPoint{Name: "cpu", Time: 8 * Second, Value: 49}},
		{&influxql.FloatPoint{Name: "cpu", Time: 12 * Second, Value: 52}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_CumulativeSum_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 8 * Second, Value: 19},
			{Name: "cpu", Time: 12 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT cumulative_sum(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 30}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 8 * Second, Value: 49}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 12 * Second, Value: 52}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_CumulativeSum_Duplicate_Float(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT cumulative_sum(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 20}},
		{&influxql.FloatPoint{Name: "cpu", Time: 0 * Second, Value: 39}},
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: 49}},
		{&influxql.FloatPoint{Name: "cpu", Time: 4 * Second, Value: 52}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_CumulativeSum_Duplicate_Integer(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0 * Second, Value: 20},
			{Name: "cpu", Time: 0 * Second, Value: 19},
			{Name: "cpu", Time: 4 * Second, Value: 10},
			{Name: "cpu", Time: 4 * Second, Value: 3},
		}}, nil
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT cumulative_sum(value) FROM cpu WHERE time >= '1970-01-01T00:00:00Z' AND time < '1970-01-01T00:00:16Z'`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 20}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 0 * Second, Value: 39}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 49}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 4 * Second, Value: 52}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_HoltWinters_GroupBy_Agg(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return influxql.NewCallIterator(&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 10 * Second, Value: 4},
			{Name: "cpu", Time: 11 * Second, Value: 6},

			{Name: "cpu", Time: 12 * Second, Value: 9},
			{Name: "cpu", Time: 13 * Second, Value: 11},

			{Name: "cpu", Time: 14 * Second, Value: 5},
			{Name: "cpu", Time: 15 * Second, Value: 7},

			{Name: "cpu", Time: 16 * Second, Value: 10},
			{Name: "cpu", Time: 17 * Second, Value: 12},

			{Name: "cpu", Time: 18 * Second, Value: 6},
			{Name: "cpu", Time: 19 * Second, Value: 8},
		}}, opt)
	}

	// Execute selection.
	itrs, err := influxql.Select(MustParseSelectStatement(`SELECT holt_winters(mean(value), 2, 2) FROM cpu WHERE time >= '1970-01-01T00:00:10Z' AND time < '1970-01-01T00:00:20Z' GROUP BY time(2s)`), &ic, nil)
	if err != nil {
		t.Fatal(err)
	} else if a, err := Iterators(itrs).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Name: "cpu", Time: 20 * Second, Value: 11.960623419918432}},
		{&influxql.FloatPoint{Name: "cpu", Time: 22 * Second, Value: 7.953140268154609}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestSelect_UnsupportedCall(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{}, nil
	}

	_, err := influxql.Select(MustParseSelectStatement(`SELECT foobar(value) FROM cpu`), &ic, nil)
	if err == nil || err.Error() != "unsupported call: foobar" {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestSelect_InvalidQueries(t *testing.T) {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		return &FloatIterator{}, nil
	}

	tests := []struct {
		q   string
		err string
	}{
		{
			q:   `SELECT foobar(value) FROM cpu`,
			err: `unsupported call: foobar`,
		},
		{
			q:   `SELECT 'value' FROM cpu`,
			err: `invalid expression type: *influxql.StringLiteral`,
		},
		{
			q:   `SELECT 'value', value FROM cpu`,
			err: `invalid expression type: *influxql.StringLiteral`,
		},
	}

	for i, tt := range tests {
		itrs, err := influxql.Select(MustParseSelectStatement(tt.q), &ic, nil)
		if err == nil || err.Error() != tt.err {
			t.Errorf("%d. expected error '%s', got '%s'", i, tt.err, err)
		}
		influxql.Iterators(itrs).Close()
	}
}

func BenchmarkSelect_Raw_1K(b *testing.B)   { benchmarkSelectRaw(b, 1000) }
func BenchmarkSelect_Raw_100K(b *testing.B) { benchmarkSelectRaw(b, 1000000) }

func benchmarkSelectRaw(b *testing.B, pointN int) {
	benchmarkSelect(b, MustParseSelectStatement(`SELECT fval FROM cpu`), NewRawBenchmarkIteratorCreator(pointN))
}

func benchmarkSelect(b *testing.B, stmt *influxql.SelectStatement, ic influxql.IteratorCreator) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		itrs, err := influxql.Select(stmt, ic, nil)
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterators(itrs)
	}
}

// NewRawBenchmarkIteratorCreator returns a new mock iterator creator with generated fields.
func NewRawBenchmarkIteratorCreator(pointN int) *IteratorCreator {
	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		if opt.Expr != nil {
			panic("unexpected expression")
		}

		p := influxql.FloatPoint{
			Name: "cpu",
			Aux:  make([]interface{}, len(opt.Aux)),
		}

		for i := range opt.Aux {
			switch opt.Aux[i].Val {
			case "fval":
				p.Aux[i] = float64(100)
			default:
				panic("unknown iterator expr: " + opt.Expr.String())
			}
		}

		return &FloatPointGenerator{N: pointN, Fn: func(i int) *influxql.FloatPoint {
			p.Time = int64(time.Duration(i) * (10 * time.Second))
			return &p
		}}, nil
	}
	return &ic
}

func benchmarkSelectDedupe(b *testing.B, seriesN, pointsPerSeries int) {
	stmt := MustParseSelectStatement(`SELECT sval::string FROM cpu`)
	stmt.Dedupe = true

	var ic IteratorCreator
	ic.CreateIteratorFn = func(opt influxql.IteratorOptions) (influxql.Iterator, error) {
		if opt.Expr != nil {
			panic("unexpected expression")
		}

		p := influxql.FloatPoint{
			Name: "tags",
			Aux:  []interface{}{nil},
		}

		return &FloatPointGenerator{N: seriesN * pointsPerSeries, Fn: func(i int) *influxql.FloatPoint {
			p.Aux[0] = fmt.Sprintf("server%d", i%seriesN)
			return &p
		}}, nil
	}

	b.ResetTimer()
	benchmarkSelect(b, stmt, &ic)
}

func BenchmarkSelect_Dedupe_1K(b *testing.B) { benchmarkSelectDedupe(b, 1000, 100) }
