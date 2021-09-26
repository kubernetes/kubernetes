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

func setUpF32SFlagSet(f32sp *[]float32) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Float32SliceVar(f32sp, "f32s", []float32{}, "Command separated list!")
	return f
}

func setUpF32SFlagSetWithDefault(f32sp *[]float32) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.Float32SliceVar(f32sp, "f32s", []float32{0.0, 1.0}, "Command separated list!")
	return f
}

func TestEmptyF32S(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSet(&f32s)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getF32S, err := f.GetFloat32Slice("f32s")
	if err != nil {
		t.Fatal("got an error from GetFloat32Slice():", err)
	}
	if len(getF32S) != 0 {
		t.Fatalf("got f32s %v with len=%d but expected length=0", getF32S, len(getF32S))
	}
}

func TestF32S(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSet(&f32s)

	vals := []string{"1.0", "2.0", "4.0", "3.0"}
	arg := fmt.Sprintf("--f32s=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range f32s {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %s but got: %f", i, vals[i], v)
		}
	}
	getF32S, err := f.GetFloat32Slice("f32s")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for i, v := range getF32S {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %s but got: %f from GetFloat32Slice", i, vals[i], v)
		}
	}
}

func TestF32SDefault(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSetWithDefault(&f32s)

	vals := []string{"0.0", "1.0"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range f32s {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %f but got: %f", i, d, v)
		}
	}

	getF32S, err := f.GetFloat32Slice("f32s")
	if err != nil {
		t.Fatal("got an error from GetFloat32Slice():", err)
	}
	for i, v := range getF32S {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatal("got an error from GetFloat32Slice():", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %f from GetFloat32Slice but got: %f", i, d, v)
		}
	}
}

func TestF32SWithDefault(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSetWithDefault(&f32s)

	vals := []string{"1.0", "2.0"}
	arg := fmt.Sprintf("--f32s=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range f32s {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %f but got: %f", i, d, v)
		}
	}

	getF32S, err := f.GetFloat32Slice("f32s")
	if err != nil {
		t.Fatal("got an error from GetFloat32Slice():", err)
	}
	for i, v := range getF32S {
		d64, err := strconv.ParseFloat(vals[i], 32)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}

		d := float32(d64)
		if d != v {
			t.Fatalf("expected f32s[%d] to be %f from GetFloat32Slice but got: %f", i, d, v)
		}
	}
}

func TestF32SAsSliceValue(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSet(&f32s)

	in := []string{"1.0", "2.0"}
	argfmt := "--f32s=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	f.VisitAll(func(f *Flag) {
		if val, ok := f.Value.(SliceValue); ok {
			_ = val.Replace([]string{"3.1"})
		}
	})
	if len(f32s) != 1 || f32s[0] != 3.1 {
		t.Fatalf("Expected ss to be overwritten with '3.1', but got: %v", f32s)
	}
}

func TestF32SCalledTwice(t *testing.T) {
	var f32s []float32
	f := setUpF32SFlagSet(&f32s)

	in := []string{"1.0,2.0", "3.0"}
	expected := []float32{1.0, 2.0, 3.0}
	argfmt := "--f32s=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range f32s {
		if expected[i] != v {
			t.Fatalf("expected f32s[%d] to be %f but got: %f", i, expected[i], v)
		}
	}
}
