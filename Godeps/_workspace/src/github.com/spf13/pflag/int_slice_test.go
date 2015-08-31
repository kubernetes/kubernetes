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

func setUpISFlagSet(isp *[]int) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.IntSliceVar(isp, "is", []int{}, "Command separated list!")
	return f
}

func setUpISFlagSetWithDefault(isp *[]int) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.IntSliceVar(isp, "is", []int{0, 1}, "Command separated list!")
	return f
}

func TestEmptyIS(t *testing.T) {
	var is []int
	f := setUpISFlagSet(&is)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getIS, err := f.GetIntSlice("is")
	if err != nil {
		t.Fatal("got an error from GetIntSlice():", err)
	}
	if len(getIS) != 0 {
		t.Fatalf("got is %v with len=%d but expected length=0", getIS, len(getIS))
	}
}

func TestIS(t *testing.T) {
	var is []int
	f := setUpISFlagSet(&is)

	vals := []string{"1", "2", "4", "3"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d", i, vals[i], v)
		}
	}
	getIS, err := f.GetIntSlice("is")
	for i, v := range getIS {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %s but got: %d from GetIntSlice", i, vals[i], v)
		}
	}
}

func TestISDefault(t *testing.T) {
	var is []int
	f := setUpISFlagSetWithDefault(&is)

	vals := []string{"0", "1"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getIS, err := f.GetIntSlice("is")
	if err != nil {
		t.Fatal("got an error from GetIntSlice():", err)
	}
	for i, v := range getIS {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatal("got an error from GetIntSlice():", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetIntSlice but got: %d", i, d, v)
		}
	}
}

func TestISWithDefault(t *testing.T) {
	var is []int
	f := setUpISFlagSetWithDefault(&is)

	vals := []string{"1", "2"}
	arg := fmt.Sprintf("--is=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range is {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d but got: %d", i, d, v)
		}
	}

	getIS, err := f.GetIntSlice("is")
	if err != nil {
		t.Fatal("got an error from GetIntSlice():", err)
	}
	for i, v := range getIS {
		d, err := strconv.Atoi(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if d != v {
			t.Fatalf("expected is[%d] to be %d from GetIntSlice but got: %d", i, d, v)
		}
	}
}

func TestISCalledTwice(t *testing.T) {
	var is []int
	f := setUpISFlagSet(&is)

	in := []string{"1,2", "3"}
	expected := []int{1, 2, 3}
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
