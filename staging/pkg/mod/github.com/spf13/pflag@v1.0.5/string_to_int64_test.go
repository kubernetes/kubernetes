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

func setUpS2I64FlagSet(s2ip *map[string]int64) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.StringToInt64Var(s2ip, "s2i", map[string]int64{}, "Command separated ls2it!")
	return f
}

func setUpS2I64FlagSetWithDefault(s2ip *map[string]int64) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.StringToInt64Var(s2ip, "s2i", map[string]int64{"a": 1, "b": 2}, "Command separated ls2it!")
	return f
}

func createS2I64Flag(vals map[string]int64) string {
	var buf bytes.Buffer
	i := 0
	for k, v := range vals {
		if i > 0 {
			buf.WriteRune(',')
		}
		buf.WriteString(k)
		buf.WriteRune('=')
		buf.WriteString(strconv.FormatInt(v, 10))
		i++
	}
	return buf.String()
}

func TestEmptyS2I64(t *testing.T) {
	var s2i map[string]int64
	f := setUpS2I64FlagSet(&s2i)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getS2I, err := f.GetStringToInt64("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt64():", err)
	}
	if len(getS2I) != 0 {
		t.Fatalf("got s2i %v with len=%d but expected length=0", getS2I, len(getS2I))
	}
}

func TestS2I64(t *testing.T) {
	var s2i map[string]int64
	f := setUpS2I64FlagSet(&s2i)

	vals := map[string]int64{"a": 1, "b": 2, "d": 4, "c": 3}
	arg := fmt.Sprintf("--s2i=%s", createS2I64Flag(vals))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}
	getS2I, err := f.GetStringToInt64("s2i")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d from GetStringToInt64", k, vals[k], v)
		}
	}
}

func TestS2I64Default(t *testing.T) {
	var s2i map[string]int64
	f := setUpS2I64FlagSetWithDefault(&s2i)

	vals := map[string]int64{"a": 1, "b": 2}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}

	getS2I, err := f.GetStringToInt64("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt64():", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d from GetStringToInt64 but got: %d", k, vals[k], v)
		}
	}
}

func TestS2I64WithDefault(t *testing.T) {
	var s2i map[string]int64
	f := setUpS2I64FlagSetWithDefault(&s2i)

	vals := map[string]int64{"a": 1, "b": 2}
	arg := fmt.Sprintf("--s2i=%s", createS2I64Flag(vals))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for k, v := range s2i {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d but got: %d", k, vals[k], v)
		}
	}

	getS2I, err := f.GetStringToInt64("s2i")
	if err != nil {
		t.Fatal("got an error from GetStringToInt64():", err)
	}
	for k, v := range getS2I {
		if vals[k] != v {
			t.Fatalf("expected s2i[%s] to be %d from GetStringToInt64 but got: %d", k, vals[k], v)
		}
	}
}

func TestS2I64CalledTwice(t *testing.T) {
	var s2i map[string]int64
	f := setUpS2I64FlagSet(&s2i)

	in := []string{"a=1,b=2", "b=3"}
	expected := map[string]int64{"a": 1, "b": 3}
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
