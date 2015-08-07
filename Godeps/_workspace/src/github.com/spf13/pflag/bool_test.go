// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"bytes"
	"fmt"
	"strconv"
	"testing"
)

// This value can be a boolean ("true", "false") or "maybe"
type triStateValue int

const (
	triStateFalse triStateValue = 0
	triStateTrue  triStateValue = 1
	triStateMaybe triStateValue = 2
)

const strTriStateMaybe = "maybe"

func (v *triStateValue) IsBoolFlag() bool {
	return true
}

func (v *triStateValue) Get() interface{} {
	return triStateValue(*v)
}

func (v *triStateValue) Set(s string) error {
	if s == strTriStateMaybe {
		*v = triStateMaybe
		return nil
	}
	boolVal, err := strconv.ParseBool(s)
	if boolVal {
		*v = triStateTrue
	} else {
		*v = triStateFalse
	}
	return err
}

func (v *triStateValue) String() string {
	if *v == triStateMaybe {
		return strTriStateMaybe
	}
	return fmt.Sprintf("%v", bool(*v == triStateTrue))
}

// The type of the flag as requred by the pflag.Value interface
func (v *triStateValue) Type() string {
	return "version"
}

func setUpFlagSet(tristate *triStateValue) *FlagSet {
	f := NewFlagSet("test", ContinueOnError)
	*tristate = triStateFalse
	flag := f.VarPF(tristate, "tristate", "t", "tristate value (true, maybe or false)")
	flag.NoOptDefVal = "true"
	return f
}

func TestExplicitTrue(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{"--tristate=true"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateTrue {
		t.Fatal("expected", triStateTrue, "(triStateTrue) but got", tristate, "instead")
	}
}

func TestImplicitTrue(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{"--tristate"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateTrue {
		t.Fatal("expected", triStateTrue, "(triStateTrue) but got", tristate, "instead")
	}
}

func TestShortFlag(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{"-t"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateTrue {
		t.Fatal("expected", triStateTrue, "(triStateTrue) but got", tristate, "instead")
	}
}

func TestShortFlagExtraArgument(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	// The"maybe"turns into an arg, since short boolean options will only do true/false
	err := f.Parse([]string{"-t", "maybe"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateTrue {
		t.Fatal("expected", triStateTrue, "(triStateTrue) but got", tristate, "instead")
	}
	args := f.Args()
	if len(args) != 1 || args[0] != "maybe" {
		t.Fatal("expected an extra 'maybe' argument to stick around")
	}
}

func TestExplicitMaybe(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{"--tristate=maybe"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateMaybe {
		t.Fatal("expected", triStateMaybe, "(triStateMaybe) but got", tristate, "instead")
	}
}

func TestExplicitFalse(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{"--tristate=false"})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateFalse {
		t.Fatal("expected", triStateFalse, "(triStateFalse) but got", tristate, "instead")
	}
}

func TestImplicitFalse(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	err := f.Parse([]string{})
	if err != nil {
		t.Fatal("expected no error; got", err)
	}
	if tristate != triStateFalse {
		t.Fatal("expected", triStateFalse, "(triStateFalse) but got", tristate, "instead")
	}
}

func TestInvalidValue(t *testing.T) {
	var tristate triStateValue
	f := setUpFlagSet(&tristate)
	var buf bytes.Buffer
	f.SetOutput(&buf)
	err := f.Parse([]string{"--tristate=invalid"})
	if err == nil {
		t.Fatal("expected an error but did not get any, tristate has value", tristate)
	}
}

func TestBoolP(t *testing.T) {
	b := BoolP("bool", "b", false, "bool value in CommandLine")
	c := BoolP("c", "c", false, "other bool value")
	args := []string{"--bool"}
	if err := CommandLine.Parse(args); err != nil {
		t.Error("expected no error, got ", err)
	}
	if *b != true {
		t.Errorf("expected b=true got b=%s", b)
	}
	if *c != false {
		t.Errorf("expect c=false got c=%s", c)
	}
}
