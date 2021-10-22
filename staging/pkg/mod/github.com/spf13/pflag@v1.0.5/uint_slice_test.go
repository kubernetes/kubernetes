package pflag

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func setUpUISFlagSet(uisp *[]uint) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.UintSliceVar(uisp, "uis", []uint{}, "Command separated list!")
	return f
}

func setUpUISFlagSetWithDefault(uisp *[]uint) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	f.UintSliceVar(uisp, "uis", []uint{0, 1}, "Command separated list!")
	return f
}

func TestEmptyUIS(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSet(&uis)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}

	getUIS, err := f.GetUintSlice("uis")
	if err != nil {
		t.Fatal("got an error from GetUintSlice():", err)
	}
	if len(getUIS) != 0 {
		t.Fatalf("got is %v with len=%d but expected length=0", getUIS, len(getUIS))
	}
}

func TestUIS(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSet(&uis)

	vals := []string{"1", "2", "4", "3"}
	arg := fmt.Sprintf("--uis=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range uis {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if uint(u) != v {
			t.Fatalf("expected uis[%d] to be %s but got %d", i, vals[i], v)
		}
	}
	getUIS, err := f.GetUintSlice("uis")
	if err != nil {
		t.Fatalf("got error: %v", err)
	}
	for i, v := range getUIS {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if uint(u) != v {
			t.Fatalf("expected uis[%d] to be %s but got: %d from GetUintSlice", i, vals[i], v)
		}
	}
}

func TestUISDefault(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSetWithDefault(&uis)

	vals := []string{"0", "1"}

	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range uis {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if uint(u) != v {
			t.Fatalf("expect uis[%d] to be %d but got: %d", i, u, v)
		}
	}

	getUIS, err := f.GetUintSlice("uis")
	if err != nil {
		t.Fatal("got an error from GetUintSlice():", err)
	}
	for i, v := range getUIS {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatal("got an error from GetIntSlice():", err)
		}
		if uint(u) != v {
			t.Fatalf("expected uis[%d] to be %d from GetUintSlice but got: %d", i, u, v)
		}
	}
}

func TestUISWithDefault(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSetWithDefault(&uis)

	vals := []string{"1", "2"}
	arg := fmt.Sprintf("--uis=%s", strings.Join(vals, ","))
	err := f.Parse([]string{arg})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range uis {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if uint(u) != v {
			t.Fatalf("expected uis[%d] to be %d from GetUintSlice but got: %d", i, u, v)
		}
	}

	getUIS, err := f.GetUintSlice("uis")
	if err != nil {
		t.Fatal("got an error from GetUintSlice():", err)
	}
	for i, v := range getUIS {
		u, err := strconv.ParseUint(vals[i], 10, 0)
		if err != nil {
			t.Fatalf("got error: %v", err)
		}
		if uint(u) != v {
			t.Fatalf("expected uis[%d] to be %d from GetUintSlice but got: %d", i, u, v)
		}
	}
}

func TestUISAsSliceValue(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSet(&uis)

	in := []string{"1", "2"}
	argfmt := "--uis=%s"
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
	if len(uis) != 1 || uis[0] != 3 {
		t.Fatalf("Expected ss to be overwritten with '3.1', but got: %v", uis)
	}
}

func TestUISCalledTwice(t *testing.T) {
	var uis []uint
	f := setUpUISFlagSet(&uis)

	in := []string{"1,2", "3"}
	expected := []int{1, 2, 3}
	argfmt := "--uis=%s"
	arg1 := fmt.Sprintf(argfmt, in[0])
	arg2 := fmt.Sprintf(argfmt, in[1])
	err := f.Parse([]string{arg1, arg2})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	for i, v := range uis {
		if uint(expected[i]) != v {
			t.Fatalf("expected uis[%d] to be %d but got: %d", i, expected[i], v)
		}
	}
}
