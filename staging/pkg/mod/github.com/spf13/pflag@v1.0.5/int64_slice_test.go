// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func setUpI64SFlagSet(isp *[]int64) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Int64SliceVar(isp, "is", []int64{}, "Command separated list!")
	return f
}

func setUpI64SFlagSetWithDefault(isp *[]int64) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Int64SliceVar(isp, "is", []int64{0, 1}, "Command separated list!")
	return f
}

func TestEmptyI64S(t *testing.T) {
	var is []int64
	f := setUpI64SFlagSet(&is)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getI64S, err := f.GetInt64Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt64Slice():", err)
	}
	if len(getI64S) != 0 {
		t.Fatalf("got is %v with len=%d but expected length=0", getI64S, len(getI64S))
	}
}

func TestI64S(t *testing.T) {
	var is []int64
	f := setUpI64SFlagSet(&is)

	vals := []string{"1", "2", "4", "3"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d", i, vals[i], v)
		}
	}
	getI64S, err := f.GetInt64Slice("is")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for i, v := range getI64S {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d from GetInt64Slice", i, vals[i], v)
		}
	}
}

func TestI64SDefault(t *testing.T) {
	var is []int64
	f := setUpI64SFlagSetWithDefault(&is)

	vals := []string{"0", "1"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getI64S, err := f.GetInt64Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt64Slice():", err)
	}
	for i, v := range getI64S {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatal("got an error from GetInt64Slice():", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetInt64Slice but got: %d", i, d, v)
		}
	}
}

func TestI64SWithDefault(t *testing.T) {
	var is []int64
	f := setUpI64SFlagSetWithDefault(&is)

	vals := []string{"1", "2"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getI64S, err := f.GetInt64Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt64Slice():", err)
	}
	for i, v := range getI64S {
		d, err := strconv.ParseInt(vals[i], 0, 64)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetInt64Slice but got: %d", i, d, v)
		}
	}
}

func TestI64SAsSliceValue(t *testing.T) {
	var i64s []int64
	f := setUpI64SFlagSet(&i64s)

	in := []string{"1", "2"}
	argfmt := "--is=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	f.VisitAll(func(f *Flag) {
		if val, ok := f.Value.(SliceValue); ok {
			_ = val.Replace([]string{"3"})
		}
	})
	if len(i64s) != 1 || i64s[0] != 3 {
		t.Fatalf("Expected ss to be overwritten with '3.1', but got: %v", i64s)
	}
}

func TestI64SCalledTwice(t *testing.T) {
	var is []int64
	f := setUpI64SFlagSet(&is)

	in := []string{"1,2", "3"}
	expected := []int64{1, 2, 3}
	argfmt := "--is=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		if expected[i] != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, expected[i], v)
		}
	}
}
