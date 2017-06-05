package pflag

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func setUpBSFlagSet(bsp *[]bool) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.BoolSliceVar(bsp, "bs", []bool{}, "Command separated list!")
	return f
}

func setUpBSFlagSetWithDefault(bsp *[]bool) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.BoolSliceVar(bsp, "bs", []bool{false, true}, "Command separated list!")
	return f
}

func TestEmptyBS(t *testing.T) {
	var bs []bool
	f := setUpBSFlagSet(&bs)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getBS, err := f.GetBoolSlice("bs")
	if err != nil {
		t.Fatal("got an error from GetBoolSlice():", err)
	}
	if len(getBS) != 0 {
		t.Fatalf("got bs %v with len=%d but expected length=0", getBS, len(getBS))
	}
}

func TestBS(t *testing.T) {
	var bs []bool
	f := setUpBSFlagSet(&bs)

	vals := []string{"1", "F", "TRUE", "0"}
	arg := fmt.Sprintf("--bs=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range bs {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if b != v {
			t.Fatalf("expected is[%d] to be %s but got: %t", i, vals[i], v)
		}
	}
	getBS, err := f.GetBoolSlice("bs")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for i, v := range getBS {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if b != v {
			t.Fatalf("expected bs[%d] to be %s but got: %t from GetBoolSlice", i, vals[i], v)
		}
	}
}

func TestBSDefault(t *testing.T) {
	var bs []bool
	f := setUpBSFlagSetWithDefault(&bs)

	vals := []string{"false", "T"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range bs {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if b != v {
			t.Fatalf("expected bs[%d] to be %t from GetBoolSlice but got: %t", i, b, v)
		}
	}

	getBS, err := f.GetBoolSlice("bs")
	if err != nil {
		t.Fatal("got an error from GetBoolSlice():", err)
	}
	for i, v := range getBS {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatal("got an error from GetBoolSlice():", err)
		}
		if b != v {
			t.Fatalf("expected bs[%d] to be %t from GetBoolSlice but got: %t", i, b, v)
		}
	}
}

func TestBSWithDefault(t *testing.T) {
	var bs []bool
	f := setUpBSFlagSetWithDefault(&bs)

	vals := []string{"FALSE", "1"}
	arg := fmt.Sprintf("--bs=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range bs {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if b != v {
			t.Fatalf("expected bs[%d] to be %t but got: %t", i, b, v)
		}
	}

	getBS, err := f.GetBoolSlice("bs")
	if err != nil {
		t.Fatal("got an error from GetBoolSlice():", err)
	}
	for i, v := range getBS {
		b, err := strconv.ParseBool(vals[i])
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if b != v {
			t.Fatalf("expected bs[%d] to be %t from GetBoolSlice but got: %t", i, b, v)
		}
	}
}

func TestBSCalledTwice(t *testing.T) {
	var bs []bool
	f := setUpBSFlagSet(&bs)

	in := []string{"T,F", "T"}
	expected := []bool{true, false, true}
	argfmt := "--bs=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range bs {
		if expected[i] != v {
			t.Fatalf("expected bs[%d] to be %t but got %t", i, expected[i], v)
		}
	}
}

func TestBSBadQuoting(t *testing.T) {

	tests := []struct {
		Want    []bool
		FlagArg []string
	}{
		{
			Want:    []bool{true, false, true},
			FlagArg: []string{"1", "0", "true"},
		},
		{
			Want:    []bool{true, false},
			FlagArg: []string{"True", "F"},
		},
		{
			Want:    []bool{true, false},
			FlagArg: []string{"T", "0"},
		},
		{
			Want:    []bool{true, false},
			FlagArg: []string{"1", "0"},
		},
		{
			Want:    []bool{true, false, false},
			FlagArg: []string{"true,false", "false"},
		},
		{
			Want:    []bool{true, false, false, true, false, true, false},
			FlagArg: []string{`"true,false,false,1,0,     T"`, " false "},
		},
		{
			Want:    []bool{false, false, true, false, true, false, true},
			FlagArg: []string{`"0, False,  T,false  , true,F"`, "true"},
		},
	}

	for i, test := range tests {

		var bs []bool
		f := setUpBSFlagSet(&bs)

		if err := f.Parse([]string{fmt.Sprintf("--bs=%s", strings.Join(test.FlagArg, ","))}); err != nil {
			t.Fatalf("flag parsing failed with error: %s\nparsing:\t%#v\nwant:\t\t%#v",
				err, test.FlagArg, test.Want[i])
		}

		for j, b := range bs {
			if b != test.Want[j] {
				t.Fatalf("bad value parsed for test %d on bool %d:\nwant:\t%t\ngot:\t%t", i, j, test.Want[j], b)
			}
		}
	}
}
