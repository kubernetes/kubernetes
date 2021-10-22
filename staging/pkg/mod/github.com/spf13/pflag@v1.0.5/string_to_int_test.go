// Copyright 2009 The Go Authors. All rights reserved.
// Use of ths2i source code s2i governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"bytes"
	"fmt"
	"strconv"
	"testing"
)

func setUpS2IFlagSet(s2ip *map[string]int) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.StringToIntVar(s2ip, "s2i", map[string]int{}, "Command separated ls2it!")
	return f
}

func setUpS2IFlagSetWithDefault(s2ip *map[string]int) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.StringToIntVar(s2ip, "s2i", map[string]int{"a": 1, "b": 2}, "Command separated ls2it!")
	return f
}

func createS2IFlag(vals map[string]int) string {
	var buf bytes.Buffer
	i := 0
	for k, v := range vals {
		if i > 0 {
			buf.WriteRune(',')
		}
		buf.WriteString(k)
		buf.WriteRune('=')
		buf.WriteString(strconv.Itoa(v))
		i++
	}
	return buf.String()
}

func TestEmptyS2I(t *testing.T) {
	var s2i map[string]int
	f := setUpS2IFlagSet(&s2i)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getS2I, err := f.GetStringToInt("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt():", err)
	}
	if len(getS2I) != 0 {
		t.Fatalf("got s2i %v with len=%d but expected length=0", getS2I, len(getS2I))
	}
}

func TestS2I(t *testing.T) {
	var s2i map[string]int
	f := setUpS2IFlagSet(&s2i)

	vals := map[string]int{"a": 1, "b": 2, "d": 4, "c": 3}
	arg := fmt.Sprintf("--s2i=%s", createS2IFlag(vals))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}
	getS2I, err := f.GetStringToInt("s2i")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d from GetStringToInt", k, vals[k], v)
		}
	}
}

func TestS2IDefault(t *testing.T) {
	var s2i map[string]int
	f := setUpS2IFlagSetWithDefault(&s2i)

	vals := map[string]int{"a": 1, "b": 2}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}

	getS2I, err := f.GetStringToInt("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt():", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d from GetStringToInt but got: %d", k, vals[k], v)
		}
	}
}

func TestS2IWithDefault(t *testing.T) {
	var s2i map[string]int
	f := setUpS2IFlagSetWithDefault(&s2i)

	vals := map[string]int{"a": 1, "b": 2}
	arg := fmt.Sprintf("--s2i=%s", createS2IFlag(vals))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}

	getS2I, err := f.GetStringToInt("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt():", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d from GetStringToInt but got: %d", k, vals[k], v)
		}
	}
}

func TestS2ICalledTwice(t *testing.T) {
	var s2i map[string]int
	f := setUpS2IFlagSet(&s2i)

	in := []string{"a=1,b=2", "b=3"}
	expected := map[string]int{"a": 1, "b": 3}
	argfmt := "--s2i=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range s2i {
		if expected[i] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", i, expected[i], v)
		}
	}
}
