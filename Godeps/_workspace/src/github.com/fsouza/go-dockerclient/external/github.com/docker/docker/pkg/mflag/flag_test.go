// Copyright 2014-2015 The Docker & Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mflag

import (
	"bytes"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// ResetForTesting clears all flag state and sets the usage function as directed.
// After calling ResetForTesting, parse errors in flag handling will not
// exit the program.
func ResetForTesting(usage func()) {
	CommandLine = NewFlagSet(os.Args[0], ContinueOnError)
	Usage = usage
}
func boolString(s string) string {
	if s == "0" {
		return "false"
	}
	return "true"
}

func TestEverything(t *testing.T) {
	ResetForTesting(nil)
	Bool([]string{"test_bool"}, false, "bool value")
	Int([]string{"test_int"}, 0, "int value")
	Int64([]string{"test_int64"}, 0, "int64 value")
	Uint([]string{"test_uint"}, 0, "uint value")
	Uint64([]string{"test_uint64"}, 0, "uint64 value")
	String([]string{"test_string"}, "0", "string value")
	Float64([]string{"test_float64"}, 0, "float64 value")
	Duration([]string{"test_duration"}, 0, "time.Duration value")

	m := make(map[string]*Flag)
	desired := "0"
	visitor := func(f *Flag) {
		for _, name := range f.Names {
			if len(name) > 5 && name[0:5] == "test_" {
				m[name] = f
				ok := false
				switch {
				case f.Value.String() == desired:
					ok = true
				case name == "test_bool" && f.Value.String() == boolString(desired):
					ok = true
				case name == "test_duration" && f.Value.String() == desired+"s":
					ok = true
				}
				if !ok {
					t.Error("Visit: bad value", f.Value.String(), "for", name)
				}
			}
		}
	}
	VisitAll(visitor)
	if len(m) != 8 {
		t.Error("VisitAll misses some flags")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	m = make(map[string]*Flag)
	Visit(visitor)
	if len(m) != 0 {
		t.Errorf("Visit sees unset flags")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	// Now set all flags
	Set("test_bool", "true")
	Set("test_int", "1")
	Set("test_int64", "1")
	Set("test_uint", "1")
	Set("test_uint64", "1")
	Set("test_string", "1")
	Set("test_float64", "1")
	Set("test_duration", "1s")
	desired = "1"
	Visit(visitor)
	if len(m) != 8 {
		t.Error("Visit fails after set")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	// Now test they're visited in sort order.
	var flagNames []string
	Visit(func(f *Flag) {
		for _, name := range f.Names {
			flagNames = append(flagNames, name)
		}
	})
	if !sort.StringsAreSorted(flagNames) {
		t.Errorf("flag names not sorted: %v", flagNames)
	}
}

func TestGet(t *testing.T) {
	ResetForTesting(nil)
	Bool([]string{"test_bool"}, true, "bool value")
	Int([]string{"test_int"}, 1, "int value")
	Int64([]string{"test_int64"}, 2, "int64 value")
	Uint([]string{"test_uint"}, 3, "uint value")
	Uint64([]string{"test_uint64"}, 4, "uint64 value")
	String([]string{"test_string"}, "5", "string value")
	Float64([]string{"test_float64"}, 6, "float64 value")
	Duration([]string{"test_duration"}, 7, "time.Duration value")

	visitor := func(f *Flag) {
		for _, name := range f.Names {
			if len(name) > 5 && name[0:5] == "test_" {
				g, ok := f.Value.(Getter)
				if !ok {
					t.Errorf("Visit: value does not satisfy Getter: %T", f.Value)
					return
				}
				switch name {
				case "test_bool":
					ok = g.Get() == true
				case "test_int":
					ok = g.Get() == int(1)
				case "test_int64":
					ok = g.Get() == int64(2)
				case "test_uint":
					ok = g.Get() == uint(3)
				case "test_uint64":
					ok = g.Get() == uint64(4)
				case "test_string":
					ok = g.Get() == "5"
				case "test_float64":
					ok = g.Get() == float64(6)
				case "test_duration":
					ok = g.Get() == time.Duration(7)
				}
				if !ok {
					t.Errorf("Visit: bad value %T(%v) for %s", g.Get(), g.Get(), name)
				}
			}
		}
	}
	VisitAll(visitor)
}

func testParse(f *FlagSet, t *testing.T) {
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	boolFlag := f.Bool([]string{"bool"}, false, "bool value")
	bool2Flag := f.Bool([]string{"bool2"}, false, "bool2 value")
	f.Bool([]string{"bool3"}, false, "bool3 value")
	bool4Flag := f.Bool([]string{"bool4"}, false, "bool4 value")
	intFlag := f.Int([]string{"-int"}, 0, "int value")
	int64Flag := f.Int64([]string{"-int64"}, 0, "int64 value")
	uintFlag := f.Uint([]string{"uint"}, 0, "uint value")
	uint64Flag := f.Uint64([]string{"-uint64"}, 0, "uint64 value")
	stringFlag := f.String([]string{"string"}, "0", "string value")
	f.String([]string{"string2"}, "0", "string2 value")
	singleQuoteFlag := f.String([]string{"squote"}, "", "single quoted value")
	doubleQuoteFlag := f.String([]string{"dquote"}, "", "double quoted value")
	mixedQuoteFlag := f.String([]string{"mquote"}, "", "mixed quoted value")
	mixed2QuoteFlag := f.String([]string{"mquote2"}, "", "mixed2 quoted value")
	nestedQuoteFlag := f.String([]string{"nquote"}, "", "nested quoted value")
	nested2QuoteFlag := f.String([]string{"nquote2"}, "", "nested2 quoted value")
	float64Flag := f.Float64([]string{"float64"}, 0, "float64 value")
	durationFlag := f.Duration([]string{"duration"}, 5*time.Second, "time.Duration value")
	extra := "one-extra-argument"
	args := []string{
		"-bool",
		"-bool2=true",
		"-bool4=false",
		"--int", "22",
		"--int64", "0x23",
		"-uint", "24",
		"--uint64", "25",
		"-string", "hello",
		"-squote='single'",
		`-dquote="double"`,
		`-mquote='mixed"`,
		`-mquote2="mixed2'`,
		`-nquote="'single nested'"`,
		`-nquote2='"double nested"'`,
		"-float64", "2718e28",
		"-duration", "2m",
		extra,
	}
	if err := f.Parse(args); err != nil {
		t.Fatal(err)
	}
	if !f.Parsed() {
		t.Error("f.Parse() = false after Parse")
	}
	if *boolFlag != true {
		t.Error("bool flag should be true, is ", *boolFlag)
	}
	if *bool2Flag != true {
		t.Error("bool2 flag should be true, is ", *bool2Flag)
	}
	if !f.IsSet("bool2") {
		t.Error("bool2 should be marked as set")
	}
	if f.IsSet("bool3") {
		t.Error("bool3 should not be marked as set")
	}
	if !f.IsSet("bool4") {
		t.Error("bool4 should be marked as set")
	}
	if *bool4Flag != false {
		t.Error("bool4 flag should be false, is ", *bool4Flag)
	}
	if *intFlag != 22 {
		t.Error("int flag should be 22, is ", *intFlag)
	}
	if *int64Flag != 0x23 {
		t.Error("int64 flag should be 0x23, is ", *int64Flag)
	}
	if *uintFlag != 24 {
		t.Error("uint flag should be 24, is ", *uintFlag)
	}
	if *uint64Flag != 25 {
		t.Error("uint64 flag should be 25, is ", *uint64Flag)
	}
	if *stringFlag != "hello" {
		t.Error("string flag should be `hello`, is ", *stringFlag)
	}
	if !f.IsSet("string") {
		t.Error("string flag should be marked as set")
	}
	if f.IsSet("string2") {
		t.Error("string2 flag should not be marked as set")
	}
	if *singleQuoteFlag != "single" {
		t.Error("single quote string flag should be `single`, is ", *singleQuoteFlag)
	}
	if *doubleQuoteFlag != "double" {
		t.Error("double quote string flag should be `double`, is ", *doubleQuoteFlag)
	}
	if *mixedQuoteFlag != `'mixed"` {
		t.Error("mixed quote string flag should be `'mixed\"`, is ", *mixedQuoteFlag)
	}
	if *mixed2QuoteFlag != `"mixed2'` {
		t.Error("mixed2 quote string flag should be `\"mixed2'`, is ", *mixed2QuoteFlag)
	}
	if *nestedQuoteFlag != "'single nested'" {
		t.Error("nested quote string flag should be `'single nested'`, is ", *nestedQuoteFlag)
	}
	if *nested2QuoteFlag != `"double nested"` {
		t.Error("double quote string flag should be `\"double nested\"`, is ", *nested2QuoteFlag)
	}
	if *float64Flag != 2718e28 {
		t.Error("float64 flag should be 2718e28, is ", *float64Flag)
	}
	if *durationFlag != 2*time.Minute {
		t.Error("duration flag should be 2m, is ", *durationFlag)
	}
	if len(f.Args()) != 1 {
		t.Error("expected one argument, got", len(f.Args()))
	} else if f.Args()[0] != extra {
		t.Errorf("expected argument %q got %q", extra, f.Args()[0])
	}
}

func testPanic(f *FlagSet, t *testing.T) {
	f.Int([]string{"-int"}, 0, "int value")
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	args := []string{
		"-int", "21",
	}
	f.Parse(args)
}

func TestParsePanic(t *testing.T) {
	ResetForTesting(func() {})
	testPanic(CommandLine, t)
}

func TestParse(t *testing.T) {
	ResetForTesting(func() { t.Error("bad parse") })
	testParse(CommandLine, t)
}

func TestFlagSetParse(t *testing.T) {
	testParse(NewFlagSet("test", ContinueOnError), t)
}

// Declare a user-defined flag type.
type flagVar []string

func (f *flagVar) String() string {
	return fmt.Sprint([]string(*f))
}

func (f *flagVar) Set(value string) error {
	*f = append(*f, value)
	return nil
}

func TestUserDefined(t *testing.T) {
	var flags FlagSet
	flags.Init("test", ContinueOnError)
	var v flagVar
	flags.Var(&v, []string{"v"}, "usage")
	if err := flags.Parse([]string{"-v", "1", "-v", "2", "-v=3"}); err != nil {
		t.Error(err)
	}
	if len(v) != 3 {
		t.Fatal("expected 3 args; got ", len(v))
	}
	expect := "[1 2 3]"
	if v.String() != expect {
		t.Errorf("expected value %q got %q", expect, v.String())
	}
}

// Declare a user-defined boolean flag type.
type boolFlagVar struct {
	count int
}

func (b *boolFlagVar) String() string {
	return fmt.Sprintf("%d", b.count)
}

func (b *boolFlagVar) Set(value string) error {
	if value == "true" {
		b.count++
	}
	return nil
}

func (b *boolFlagVar) IsBoolFlag() bool {
	return b.count < 4
}

func TestUserDefinedBool(t *testing.T) {
	var flags FlagSet
	flags.Init("test", ContinueOnError)
	var b boolFlagVar
	var err error
	flags.Var(&b, []string{"b"}, "usage")
	if err = flags.Parse([]string{"-b", "-b", "-b", "-b=true", "-b=false", "-b", "barg", "-b"}); err != nil {
		if b.count < 4 {
			t.Error(err)
		}
	}

	if b.count != 4 {
		t.Errorf("want: %d; got: %d", 4, b.count)
	}

	if err == nil {
		t.Error("expected error; got none")
	}
}

func TestSetOutput(t *testing.T) {
	var flags FlagSet
	var buf bytes.Buffer
	flags.SetOutput(&buf)
	flags.Init("test", ContinueOnError)
	flags.Parse([]string{"-unknown"})
	if out := buf.String(); !strings.Contains(out, "-unknown") {
		t.Logf("expected output mentioning unknown; got %q", out)
	}
}

// This tests that one can reset the flags. This still works but not well, and is
// superseded by FlagSet.
func TestChangingArgs(t *testing.T) {
	ResetForTesting(func() { t.Fatal("bad parse") })
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"cmd", "-before", "subcmd", "-after", "args"}
	before := Bool([]string{"before"}, false, "")
	if err := CommandLine.Parse(os.Args[1:]); err != nil {
		t.Fatal(err)
	}
	cmd := Arg(0)
	os.Args = Args()
	after := Bool([]string{"after"}, false, "")
	Parse()
	args := Args()

	if !*before || cmd != "subcmd" || !*after || len(args) != 1 || args[0] != "args" {
		t.Fatalf("expected true subcmd true [args] got %v %v %v %v", *before, cmd, *after, args)
	}
}

// Test that -help invokes the usage message and returns ErrHelp.
func TestHelp(t *testing.T) {
	var helpCalled = false
	fs := NewFlagSet("help test", ContinueOnError)
	fs.Usage = func() { helpCalled = true }
	var flag bool
	fs.BoolVar(&flag, []string{"flag"}, false, "regular flag")
	// Regular flag invocation should work
	err := fs.Parse([]string{"-flag=true"})
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}
	if !flag {
		t.Error("flag was not set by -flag")
	}
	if helpCalled {
		t.Error("help called for regular flag")
		helpCalled = false // reset for next test
	}
	// Help flag should work as expected.
	err = fs.Parse([]string{"-help"})
	if err == nil {
		t.Fatal("error expected")
	}
	if err != ErrHelp {
		t.Fatal("expected ErrHelp; got ", err)
	}
	if !helpCalled {
		t.Fatal("help was not called")
	}
	// If we define a help flag, that should override.
	var help bool
	fs.BoolVar(&help, []string{"help"}, false, "help flag")
	helpCalled = false
	err = fs.Parse([]string{"-help"})
	if err != nil {
		t.Fatal("expected no error for defined -help; got ", err)
	}
	if helpCalled {
		t.Fatal("help was called; should not have been for defined help flag")
	}
}

// Test the flag count functions.
func TestFlagCounts(t *testing.T) {
	fs := NewFlagSet("help test", ContinueOnError)
	var flag bool
	fs.BoolVar(&flag, []string{"flag1"}, false, "regular flag")
	fs.BoolVar(&flag, []string{"#deprecated1"}, false, "regular flag")
	fs.BoolVar(&flag, []string{"f", "flag2"}, false, "regular flag")
	fs.BoolVar(&flag, []string{"#d", "#deprecated2"}, false, "regular flag")
	fs.BoolVar(&flag, []string{"flag3"}, false, "regular flag")
	fs.BoolVar(&flag, []string{"g", "#flag4", "-flag4"}, false, "regular flag")

	if fs.FlagCount() != 6 {
		t.Fatal("FlagCount wrong. ", fs.FlagCount())
	}
	if fs.FlagCountUndeprecated() != 4 {
		t.Fatal("FlagCountUndeprecated wrong. ", fs.FlagCountUndeprecated())
	}
	if fs.NFlag() != 0 {
		t.Fatal("NFlag wrong. ", fs.NFlag())
	}
	err := fs.Parse([]string{"-fd", "-g", "-flag4"})
	if err != nil {
		t.Fatal("expected no error for defined -help; got ", err)
	}
	if fs.NFlag() != 4 {
		t.Fatal("NFlag wrong. ", fs.NFlag())
	}
}

// Show up bug in sortFlags
func TestSortFlags(t *testing.T) {
	fs := NewFlagSet("help TestSortFlags", ContinueOnError)

	var err error

	var b bool
	fs.BoolVar(&b, []string{"b", "-banana"}, false, "usage")

	err = fs.Parse([]string{"--banana=true"})
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}

	count := 0

	fs.VisitAll(func(flag *Flag) {
		count++
		if flag == nil {
			t.Fatal("VisitAll should not return a nil flag")
		}
	})
	flagcount := fs.FlagCount()
	if flagcount != count {
		t.Fatalf("FlagCount (%d) != number (%d) of elements visited", flagcount, count)
	}
	// Make sure its idempotent
	if flagcount != fs.FlagCount() {
		t.Fatalf("FlagCount (%d) != fs.FlagCount() (%d) of elements visited", flagcount, fs.FlagCount())
	}

	count = 0
	fs.Visit(func(flag *Flag) {
		count++
		if flag == nil {
			t.Fatal("Visit should not return a nil flag")
		}
	})
	nflag := fs.NFlag()
	if nflag != count {
		t.Fatalf("NFlag (%d) != number (%d) of elements visited", nflag, count)
	}
	if nflag != fs.NFlag() {
		t.Fatalf("NFlag (%d) != fs.NFlag() (%d) of elements visited", nflag, fs.NFlag())
	}
}
