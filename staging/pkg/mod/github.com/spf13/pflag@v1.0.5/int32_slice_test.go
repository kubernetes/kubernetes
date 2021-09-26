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

func setUpI32SFlagSet(isp *[]int32) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Int32SliceVar(isp, "is", []int32{}, "Command separated list!")
	return f
}

func setUpI32SFlagSetWithDefault(isp *[]int32) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Int32SliceVar(isp, "is", []int32{0, 1}, "Command separated list!")
	return f
}

func TestEmptyI32S(t *testing.T) {
	var is []int32
	f := setUpI32SFlagSet(&is)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getI32S, err := f.GetInt32Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt32Slice():", err)
	}
	if len(getI32S) != 0 {
		t.Fatalf("got is %v with len=%d but expected length=0", getI32S, len(getI32S))
	}
}

func TestI32S(t *testing.T) {
	var is []int32
	f := setUpI32SFlagSet(&is)

	vals := []string{"1", "2", "4", "3"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d", i, vals[i], v)
		}
	}
	getI32S, err := f.GetInt32Slice("is")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for i, v := range getI32S {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d from GetInt32Slice", i, vals[i], v)
		}
	}
}

func TestI32SDefault(t *testing.T) {
	var is []int32
	f := setUpI32SFlagSetWithDefault(&is)

	vals := []string{"0", "1"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getI32S, err := f.GetInt32Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt32Slice():", err)
	}
	for i, v := range getI32S {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatal("got an error from GetInt32Slice():", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetInt32Slice but got: %d", i, d, v)
		}
	}
}

func TestI32SWithDefault(t *testing.T) {
	var is []int32
	f := setUpI32SFlagSetWithDefault(&is)

	vals := []string{"1", "2"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getI32S, err := f.GetInt32Slice("is")
	if err != nil {
		t.Fatal("got an error from GetInt32Slice():", err)
	}
	for i, v := range getI32S {
		d64, err := strconv.ParseInt(vals[i], 0, 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		d := int32(d64)
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetInt32Slice but got: %d", i, d, v)
		}
	}
}

func TestI32SAsSliceValue(t *testing.T) {
	var i32s []int32
	f := setUpI32SFlagSet(&i32s)

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
	if len(i32s) != 1 || i32s[0] != 3 {
		t.Fatalf("Expected ss to be overwritten with '3.1', but got: %v", i32s)
	}
}

func TestI32SCalledTwice(t *testing.T) {
	var is []int32
	f := setUpI32SFlagSet(&is)

	in := []string{"1,2", "3"}
	expected := []int32{1, 2, 3}
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
