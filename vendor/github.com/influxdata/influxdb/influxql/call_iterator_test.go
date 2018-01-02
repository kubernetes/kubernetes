package influxql_test

import (
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/pkg/deep"
)

// Ensure that a float iterator can be created for a count() call.
func TestCallIterator_Count_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Name: "cpu", Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "cpu", Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Name: "cpu", Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Name: "cpu", Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "mem", Time: 23, Value: 10, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`count("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 3, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 5, Value: 1, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "mem", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that an integer iterator can be created for a count() call.
func TestCallIterator_Count_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Name: "cpu", Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "cpu", Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Name: "cpu", Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Name: "cpu", Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "mem", Time: 23, Value: 10, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`count("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 3, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 5, Value: 1, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "mem", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a string iterator can be created for a count() call.
func TestCallIterator_Count_String(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&StringIterator{Points: []influxql.StringPoint{
			{Name: "cpu", Time: 0, Value: "d", Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: "c", Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "cpu", Time: 2, Value: "b", Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: "b", Tags: ParseTags("region=us-west,host=hostA")},

			{Name: "cpu", Time: 5, Value: "e", Tags: ParseTags("region=us-east,host=hostA")},

			{Name: "cpu", Time: 23, Value: "a", Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "mem", Time: 23, Value: "b", Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`count("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 3, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 5, Value: 1, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "mem", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a count() call.
func TestCallIterator_Count_Boolean(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&BooleanIterator{Points: []influxql.BooleanPoint{
			{Name: "cpu", Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "cpu", Time: 2, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
			{Name: "cpu", Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostA")},

			{Name: "cpu", Time: 5, Value: false, Tags: ParseTags("region=us-east,host=hostA")},

			{Name: "cpu", Time: 23, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
			{Name: "mem", Time: 23, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`count("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 3, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 0, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 5, Value: 1, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "cpu", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Name: "mem", Time: 20, Value: 1, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a min() call.
func TestCallIterator_Min_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 4, Value: 12, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`min("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 1, Value: 10, Tags: ParseTags("host=hostA"), Aggregated: 4}},
		{&influxql.FloatPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a integer iterator can be created for a min() call.
func TestCallIterator_Min_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 4, Value: 12, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`min("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 1, Value: 10, Tags: ParseTags("host=hostA"), Aggregated: 4}},
		{&influxql.IntegerPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a min() call.
func TestCallIterator_Min_Boolean(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&BooleanIterator{Points: []influxql.BooleanPoint{
			{Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: false, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`min("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Time: 2, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.BooleanPoint{Time: 1, Value: false, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 5, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 23, Value: true, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a max() call.
func TestCallIterator_Max_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`max("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 0, Value: 15, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.FloatPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a integer iterator can be created for a max() call.
func TestCallIterator_Max_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`max("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 0, Value: 15, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a max() call.
func TestCallIterator_Max_Boolean(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&BooleanIterator{Points: []influxql.BooleanPoint{
			{Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: false, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`max("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Time: 0, Value: true, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.BooleanPoint{Time: 1, Value: false, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 5, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 23, Value: true, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a sum() call.
func TestCallIterator_Sum_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`sum("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 0, Value: 35, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.FloatPoint{Time: 0, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 20, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that an integer iterator can be created for a sum() call.
func TestCallIterator_Sum_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 5, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`sum("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 0, Value: 35, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Time: 0, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 5, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 20, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a first() call.
func TestCallIterator_First_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`first("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 0, Value: 15, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.FloatPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 6, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that an integer iterator can be created for a first() call.
func TestCallIterator_First_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`first("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 0, Value: 15, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 6, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a string iterator can be created for a first() call.
func TestCallIterator_First_String(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&StringIterator{Points: []influxql.StringPoint{
			{Time: 1, Value: "c", Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: "b", Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: "d", Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: "b", Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: "e", Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: "a", Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`first("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Time: 0, Value: "d", Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.StringPoint{Time: 1, Value: "c", Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.StringPoint{Time: 6, Value: "e", Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.StringPoint{Time: 23, Value: "a", Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a first() call.
func TestCallIterator_First_Boolean(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&BooleanIterator{Points: []influxql.BooleanPoint{
			{Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: false, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: false, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`first("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Time: 0, Value: true, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.BooleanPoint{Time: 1, Value: true, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 6, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 23, Value: false, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a last() call.
func TestCallIterator_Last_Float(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&FloatIterator{Points: []influxql.FloatPoint{
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`last("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 2, Value: 10, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.FloatPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 6, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.FloatPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that an integer iterator can be created for a last() call.
func TestCallIterator_Last_Integer(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&IntegerIterator{Points: []influxql.IntegerPoint{
			{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`last("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 2, Value: 10, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.IntegerPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 6, Value: 20, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.IntegerPoint{Time: 23, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a string iterator can be created for a last() call.
func TestCallIterator_Last_String(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&StringIterator{Points: []influxql.StringPoint{
			{Time: 1, Value: "c", Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: "b", Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: "d", Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: "b", Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: "e", Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: "a", Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`last("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Time: 2, Value: "b", Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.StringPoint{Time: 1, Value: "c", Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.StringPoint{Time: 6, Value: "e", Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.StringPoint{Time: 23, Value: "a", Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a last() call.
func TestCallIterator_Last_Boolean(t *testing.T) {
	itr, _ := influxql.NewCallIterator(
		&BooleanIterator{Points: []influxql.BooleanPoint{
			{Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
			{Time: 2, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
			{Time: 1, Value: false, Tags: ParseTags("region=us-west,host=hostA")},

			{Time: 6, Value: false, Tags: ParseTags("region=us-east,host=hostA")},

			{Time: 23, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
		}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`last("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Time: 2, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 3}},
		{&influxql.BooleanPoint{Time: 1, Value: true, Tags: ParseTags("host=hostB"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 6, Value: false, Tags: ParseTags("host=hostA"), Aggregated: 1}},
		{&influxql.BooleanPoint{Time: 23, Value: false, Tags: ParseTags("host=hostB"), Aggregated: 1}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a float iterator can be created for a mode() call.
func TestCallIterator_Mode_Float(t *testing.T) {
	itr, _ := influxql.NewModeIterator(&FloatIterator{Points: []influxql.FloatPoint{
		{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},
		{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 3, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 4, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 7, Value: 21, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 8, Value: 21, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 22, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 24, Value: 25, Tags: ParseTags("region=us-west,host=hostB")},
	}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`mode("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.FloatPoint{Time: 0, Value: 10, Tags: ParseTags("host=hostA"), Aggregated: 0}},
		{&influxql.FloatPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB"), Aggregated: 0}},
		{&influxql.FloatPoint{Time: 5, Value: 21, Tags: ParseTags("host=hostA"), Aggregated: 0}},
		{&influxql.FloatPoint{Time: 20, Value: 8, Tags: ParseTags("host=hostB"), Aggregated: 0}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a integer iterator can be created for a mode() call.
func TestCallIterator_Mode_Integer(t *testing.T) {
	itr, _ := influxql.NewModeIterator(&IntegerIterator{Points: []influxql.IntegerPoint{
		{Time: 0, Value: 15, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 1, Value: 11, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 1, Value: 10, Tags: ParseTags("region=us-west,host=hostA")},
		{Time: 2, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 3, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 4, Value: 10, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 6, Value: 20, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 7, Value: 21, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 8, Value: 21, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 22, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 23, Value: 8, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 24, Value: 25, Tags: ParseTags("region=us-west,host=hostB")},
	}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`mode("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.IntegerPoint{Time: 0, Value: 10, Tags: ParseTags("host=hostA")}},
		{&influxql.IntegerPoint{Time: 1, Value: 11, Tags: ParseTags("host=hostB")}},
		{&influxql.IntegerPoint{Time: 5, Value: 21, Tags: ParseTags("host=hostA")}},
		{&influxql.IntegerPoint{Time: 20, Value: 8, Tags: ParseTags("host=hostB")}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a string iterator can be created for a mode() call.
func TestCallIterator_Mode_String(t *testing.T) {
	itr, _ := influxql.NewModeIterator(&StringIterator{Points: []influxql.StringPoint{
		{Time: 0, Value: "15", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 1, Value: "11", Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 1, Value: "10", Tags: ParseTags("region=us-west,host=hostA")},
		{Time: 2, Value: "10", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 3, Value: "10", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 4, Value: "10", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 6, Value: "20", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 7, Value: "21", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 7, Value: "21", Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 22, Value: "8", Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 23, Value: "8", Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 24, Value: "25", Tags: ParseTags("region=us-west,host=hostB")},
	}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`mode("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.StringPoint{Time: 0, Value: "10", Tags: ParseTags("host=hostA")}},
		{&influxql.StringPoint{Time: 1, Value: "11", Tags: ParseTags("host=hostB")}},
		{&influxql.StringPoint{Time: 5, Value: "21", Tags: ParseTags("host=hostA")}},
		{&influxql.StringPoint{Time: 20, Value: "8", Tags: ParseTags("host=hostB")}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

// Ensure that a boolean iterator can be created for a modBooleanl.
func TestCallIterator_Mode_Boolean(t *testing.T) {
	itr, _ := influxql.NewModeIterator(&BooleanIterator{Points: []influxql.BooleanPoint{
		{Time: 0, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 1, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 1, Value: true, Tags: ParseTags("region=us-west,host=hostA")},
		{Time: 2, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 3, Value: true, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 4, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 6, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 7, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 8, Value: false, Tags: ParseTags("region=us-east,host=hostA")},
		{Time: 22, Value: false, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 23, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
		{Time: 24, Value: true, Tags: ParseTags("region=us-west,host=hostB")},
	}},
		influxql.IteratorOptions{
			Expr:       MustParseExpr(`mode("value")`),
			Dimensions: []string{"host"},
			Interval:   influxql.Interval{Duration: 5 * time.Nanosecond},
		},
	)

	if a, err := Iterators([]influxql.Iterator{itr}).ReadAll(); err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if !deep.Equal(a, [][]influxql.Point{
		{&influxql.BooleanPoint{Time: 0, Value: true, Tags: ParseTags("host=hostA")}},
		{&influxql.BooleanPoint{Time: 1, Value: false, Tags: ParseTags("host=hostB")}},
		{&influxql.BooleanPoint{Time: 5, Value: false, Tags: ParseTags("host=hostA")}},
		{&influxql.BooleanPoint{Time: 20, Value: true, Tags: ParseTags("host=hostB")}},
	}) {
		t.Fatalf("unexpected points: %s", spew.Sdump(a))
	}
}

func TestNewCallIterator_UnsupportedExprName(t *testing.T) {
	_, err := influxql.NewCallIterator(
		&FloatIterator{},
		influxql.IteratorOptions{
			Expr: MustParseExpr(`foobar("value")`),
		},
	)

	if err == nil || err.Error() != "unsupported function call: foobar" {
		t.Errorf("unexpected error: %s", err)
	}
}

func BenchmarkCountIterator_1K(b *testing.B)   { benchmarkCountIterator(b, 1000) }
func BenchmarkCountIterator_100K(b *testing.B) { benchmarkCountIterator(b, 100000) }
func BenchmarkCountIterator_1M(b *testing.B)   { benchmarkCountIterator(b, 1000000) }

func benchmarkCountIterator(b *testing.B, pointN int) {
	benchmarkCallIterator(b, influxql.IteratorOptions{
		Expr:      MustParseExpr("count(value)"),
		StartTime: influxql.MinTime,
		EndTime:   influxql.MaxTime,
	}, pointN)
}

func benchmarkCallIterator(b *testing.B, opt influxql.IteratorOptions, pointN int) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Create a lightweight point generator.
		p := influxql.FloatPoint{Name: "cpu", Value: 100}
		input := FloatPointGenerator{
			N:  pointN,
			Fn: func(i int) *influxql.FloatPoint { return &p },
		}

		// Execute call against input.
		itr, err := influxql.NewCallIterator(&input, opt)
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterator(itr)
	}
}

func BenchmarkSampleIterator_1k(b *testing.B)   { benchmarkSampleIterator(b, 1000) }
func BenchmarkSampleIterator_100k(b *testing.B) { benchmarkSampleIterator(b, 100000) }
func BenchmarkSampleIterator_1M(b *testing.B)   { benchmarkSampleIterator(b, 1000000) }

func benchmarkSampleIterator(b *testing.B, pointN int) {
	b.ReportAllocs()

	// Create a lightweight point generator.
	p := influxql.FloatPoint{Name: "cpu"}
	input := FloatPointGenerator{
		N: pointN,
		Fn: func(i int) *influxql.FloatPoint {
			p.Value = float64(i)
			return &p
		},
	}

	for i := 0; i < b.N; i++ {
		// Execute call against input.
		itr, err := influxql.NewSampleIterator(&input, influxql.IteratorOptions{}, 100)
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterator(itr)
	}
}

func BenchmarkDistinctIterator_1K(b *testing.B)   { benchmarkDistinctIterator(b, 1000) }
func BenchmarkDistinctIterator_100K(b *testing.B) { benchmarkDistinctIterator(b, 100000) }
func BenchmarkDistinctIterator_1M(b *testing.B)   { benchmarkDistinctIterator(b, 1000000) }

func benchmarkDistinctIterator(b *testing.B, pointN int) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Create a lightweight point generator.
		p := influxql.FloatPoint{Name: "cpu"}
		input := FloatPointGenerator{
			N: pointN,
			Fn: func(i int) *influxql.FloatPoint {
				p.Value = float64(i % 10)
				return &p
			},
		}

		// Execute call against input.
		itr, err := influxql.NewDistinctIterator(&input, influxql.IteratorOptions{})
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterator(itr)
	}
}

func BenchmarkModeIterator_1K(b *testing.B)   { benchmarkModeIterator(b, 1000) }
func BenchmarkModeIterator_100K(b *testing.B) { benchmarkModeIterator(b, 100000) }
func BenchmarkModeIterator_1M(b *testing.B)   { benchmarkModeIterator(b, 1000000) }

func benchmarkModeIterator(b *testing.B, pointN int) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Create a lightweight point generator.
		p := influxql.FloatPoint{Name: "cpu"}
		input := FloatPointGenerator{
			N: pointN,
			Fn: func(i int) *influxql.FloatPoint {
				p.Value = float64(10)
				return &p
			},
		}

		// Execute call against input.
		itr, err := influxql.NewModeIterator(&input, influxql.IteratorOptions{})
		if err != nil {
			b.Fatal(err)
		}
		influxql.DrainIterator(itr)
	}
}

type FloatPointGenerator struct {
	i  int
	N  int
	Fn func(i int) *influxql.FloatPoint
}

func (g *FloatPointGenerator) Close() error                  { return nil }
func (g *FloatPointGenerator) Stats() influxql.IteratorStats { return influxql.IteratorStats{} }

func (g *FloatPointGenerator) Next() (*influxql.FloatPoint, error) {
	if g.i == g.N {
		return nil, nil
	}
	p := g.Fn(g.i)
	g.i++
	return p, nil
}
