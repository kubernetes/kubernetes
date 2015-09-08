// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

var (
	testBool                     = Bool("test_bool", false, "bool value")
	testInt                      = Int("test_int", 0, "int value")
	testInt64                    = Int64("test_int64", 0, "int64 value")
	testUint                     = Uint("test_uint", 0, "uint value")
	testUint64                   = Uint64("test_uint64", 0, "uint64 value")
	testString                   = String("test_string", "0", "string value")
	testFloat                    = Float64("test_float64", 0, "float64 value")
	testDuration                 = Duration("test_duration", 0, "time.Duration value")
	testOptionalInt              = Int("test_optional_int", 0, "optional int value")
	normalizeFlagNameInvocations = 0
)

func boolString(s string) string {
	if s == "0" {
		return "false"
	}
	return "true"
}

func TestEverything(t *testing.T) {
	m := make(map[string]*Flag)
	desired := "0"
	visitor := func(f *Flag) {
		if len(f.Name) > 5 && f.Name[0:5] == "test_" {
			m[f.Name] = f
			ok := false
			switch {
			case f.Value.String() == desired:
				ok = true
			case f.Name == "test_bool" && f.Value.String() == boolString(desired):
				ok = true
			case f.Name == "test_duration" && f.Value.String() == desired+"s":
				ok = true
			}
			if !ok {
				t.Error("Visit: bad value", f.Value.String(), "for", f.Name)
			}
		}
	}
	VisitAll(visitor)
	if len(m) != 9 {
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
	Set("test_optional_int", "1")
	desired = "1"
	Visit(visitor)
	if len(m) != 9 {
		t.Error("Visit fails after set")
		for k, v := range m {
			t.Log(k, *v)
		}
	}
	// Now test they're visited in sort order.
	var flagNames []string
	Visit(func(f *Flag) { flagNames = append(flagNames, f.Name) })
	if !sort.StringsAreSorted(flagNames) {
		t.Errorf("flag names not sorted: %v", flagNames)
	}
}

func TestUsage(t *testing.T) {
	called := false
	ResetForTesting(func() { called = true })
	if GetCommandLine().Parse([]string{"--x"}) == nil {
		t.Error("parse did not fail for unknown flag")
	}
	if !called {
		t.Error("did not call Usage for unknown flag")
	}
}

func TestAddFlagSet(t *testing.T) {
	oldSet := NewFlagSet("old", ContinueOnError)
	newSet := NewFlagSet("new", ContinueOnError)

	oldSet.String("flag1", "flag1", "flag1")
	oldSet.String("flag2", "flag2", "flag2")

	newSet.String("flag2", "flag2", "flag2")
	newSet.String("flag3", "flag3", "flag3")

	oldSet.AddFlagSet(newSet)

	if len(oldSet.formal) != 3 {
		t.Errorf("Unexpected result adding a FlagSet to a FlagSet %v", oldSet)
	}
}

func TestAnnotation(t *testing.T) {
	f := NewFlagSet("shorthand", ContinueOnError)

	if err := f.SetAnnotation("missing-flag", "key", nil); err == nil {
		t.Errorf("Expected error setting annotation on non-existent flag")
	}

	f.StringP("stringa", "a", "", "string value")
	if err := f.SetAnnotation("stringa", "key", nil); err != nil {
		t.Errorf("Unexpected error setting new nil annotation: %v", err)
	}
	if annotation := f.Lookup("stringa").Annotations["key"]; annotation != nil {
		t.Errorf("Unexpected annotation: %v", annotation)
	}

	f.StringP("stringb", "b", "", "string2 value")
	if err := f.SetAnnotation("stringb", "key", []string{"value1"}); err != nil {
		t.Errorf("Unexpected error setting new annotation: %v", err)
	}
	if annotation := f.Lookup("stringb").Annotations["key"]; !reflect.DeepEqual(annotation, []string{"value1"}) {
		t.Errorf("Unexpected annotation: %v", annotation)
	}

	if err := f.SetAnnotation("stringb", "key", []string{"value2"}); err != nil {
		t.Errorf("Unexpected error updating annotation: %v", err)
	}
	if annotation := f.Lookup("stringb").Annotations["key"]; !reflect.DeepEqual(annotation, []string{"value2"}) {
		t.Errorf("Unexpected annotation: %v", annotation)
	}
}

func testParse(f *FlagSet, t *testing.T) {
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	boolFlag := f.Bool("bool", false, "bool value")
	bool2Flag := f.Bool("bool2", false, "bool2 value")
	bool3Flag := f.Bool("bool3", false, "bool3 value")
	intFlag := f.Int("int", 0, "int value")
	int8Flag := f.Int8("int8", 0, "int value")
	int32Flag := f.Int32("int32", 0, "int value")
	int64Flag := f.Int64("int64", 0, "int64 value")
	uintFlag := f.Uint("uint", 0, "uint value")
	uint8Flag := f.Uint8("uint8", 0, "uint value")
	uint16Flag := f.Uint16("uint16", 0, "uint value")
	uint32Flag := f.Uint32("uint32", 0, "uint value")
	uint64Flag := f.Uint64("uint64", 0, "uint64 value")
	stringFlag := f.String("string", "0", "string value")
	float32Flag := f.Float32("float32", 0, "float32 value")
	float64Flag := f.Float64("float64", 0, "float64 value")
	ipFlag := f.IP("ip", net.ParseIP("127.0.0.1"), "ip value")
	maskFlag := f.IPMask("mask", ParseIPv4Mask("0.0.0.0"), "mask value")
	durationFlag := f.Duration("duration", 5*time.Second, "time.Duration value")
	optionalIntNoValueFlag := f.Int("optional-int-no-value", 0, "int value")
	f.Lookup("optional-int-no-value").NoOptDefVal = "9"
	optionalIntWithValueFlag := f.Int("optional-int-with-value", 0, "int value")
	f.Lookup("optional-int-no-value").NoOptDefVal = "9"
	extra := "one-extra-argument"
	args := []string{
		"--bool",
		"--bool2=true",
		"--bool3=false",
		"--int=22",
		"--int8=-8",
		"--int32=-32",
		"--int64=0x23",
		"--uint", "24",
		"--uint8=8",
		"--uint16=16",
		"--uint32=32",
		"--uint64=25",
		"--string=hello",
		"--float32=-172e12",
		"--float64=2718e28",
		"--ip=10.11.12.13",
		"--mask=255.255.255.0",
		"--duration=2m",
		"--optional-int-no-value",
		"--optional-int-with-value=42",
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
	if v, err := f.GetBool("bool"); err != nil || v != *boolFlag {
		t.Error("GetBool does not work.")
	}
	if *bool2Flag != true {
		t.Error("bool2 flag should be true, is ", *bool2Flag)
	}
	if *bool3Flag != false {
		t.Error("bool3 flag should be false, is ", *bool2Flag)
	}
	if *intFlag != 22 {
		t.Error("int flag should be 22, is ", *intFlag)
	}
	if v, err := f.GetInt("int"); err != nil || v != *intFlag {
		t.Error("GetInt does not work.")
	}
	if *int8Flag != -8 {
		t.Error("int8 flag should be 0x23, is ", *int8Flag)
	}
	if v, err := f.GetInt8("int8"); err != nil || v != *int8Flag {
		t.Error("GetInt8 does not work.")
	}
	if *int32Flag != -32 {
		t.Error("int32 flag should be 0x23, is ", *int32Flag)
	}
	if v, err := f.GetInt32("int32"); err != nil || v != *int32Flag {
		t.Error("GetInt32 does not work.")
	}
	if *int64Flag != 0x23 {
		t.Error("int64 flag should be 0x23, is ", *int64Flag)
	}
	if v, err := f.GetInt64("int64"); err != nil || v != *int64Flag {
		t.Error("GetInt64 does not work.")
	}
	if *uintFlag != 24 {
		t.Error("uint flag should be 24, is ", *uintFlag)
	}
	if v, err := f.GetUint("uint"); err != nil || v != *uintFlag {
		t.Error("GetUint does not work.")
	}
	if *uint8Flag != 8 {
		t.Error("uint8 flag should be 8, is ", *uint8Flag)
	}
	if v, err := f.GetUint8("uint8"); err != nil || v != *uint8Flag {
		t.Error("GetUint8 does not work.")
	}
	if *uint16Flag != 16 {
		t.Error("uint16 flag should be 16, is ", *uint16Flag)
	}
	if v, err := f.GetUint16("uint16"); err != nil || v != *uint16Flag {
		t.Error("GetUint16 does not work.")
	}
	if *uint32Flag != 32 {
		t.Error("uint32 flag should be 32, is ", *uint32Flag)
	}
	if v, err := f.GetUint32("uint32"); err != nil || v != *uint32Flag {
		t.Error("GetUint32 does not work.")
	}
	if *uint64Flag != 25 {
		t.Error("uint64 flag should be 25, is ", *uint64Flag)
	}
	if v, err := f.GetUint64("uint64"); err != nil || v != *uint64Flag {
		t.Error("GetUint64 does not work.")
	}
	if *stringFlag != "hello" {
		t.Error("string flag should be `hello`, is ", *stringFlag)
	}
	if v, err := f.GetString("string"); err != nil || v != *stringFlag {
		t.Error("GetString does not work.")
	}
	if *float32Flag != -172e12 {
		t.Error("float32 flag should be -172e12, is ", *float32Flag)
	}
	if v, err := f.GetFloat32("float32"); err != nil || v != *float32Flag {
		t.Errorf("GetFloat32 returned %v but float32Flag was %v", v, *float32Flag)
	}
	if *float64Flag != 2718e28 {
		t.Error("float64 flag should be 2718e28, is ", *float64Flag)
	}
	if v, err := f.GetFloat64("float64"); err != nil || v != *float64Flag {
		t.Errorf("GetFloat64 returned %v but float64Flag was %v", v, *float64Flag)
	}
	if !(*ipFlag).Equal(net.ParseIP("10.11.12.13")) {
		t.Error("ip flag should be 10.11.12.13, is ", *ipFlag)
	}
	if v, err := f.GetIP("ip"); err != nil || !v.Equal(*ipFlag) {
		t.Errorf("GetIP returned %v but ipFlag was %v", v, *ipFlag)
	}
	if (*maskFlag).String() != ParseIPv4Mask("255.255.255.0").String() {
		t.Error("mask flag should be 255.255.255.0, is ", (*maskFlag).String())
	}
	if v, err := f.GetIPv4Mask("mask"); err != nil || v.String() != (*maskFlag).String() {
		t.Errorf("GetIP returned %v maskFlag was %v error was %v", v, *maskFlag, err)
	}
	if *durationFlag != 2*time.Minute {
		t.Error("duration flag should be 2m, is ", *durationFlag)
	}
	if v, err := f.GetDuration("duration"); err != nil || v != *durationFlag {
		t.Error("GetDuration does not work.")
	}
	if _, err := f.GetInt("duration"); err == nil {
		t.Error("GetInt parsed a time.Duration?!?!")
	}
	if *optionalIntNoValueFlag != 9 {
		t.Error("optional int flag should be the default value, is ", *optionalIntNoValueFlag)
	}
	if *optionalIntWithValueFlag != 42 {
		t.Error("optional int flag should be 42, is ", *optionalIntWithValueFlag)
	}
	if len(f.Args()) != 1 {
		t.Error("expected one argument, got", len(f.Args()))
	} else if f.Args()[0] != extra {
		t.Errorf("expected argument %q got %q", extra, f.Args()[0])
	}
}

func TestShorthand(t *testing.T) {
	f := NewFlagSet("shorthand", ContinueOnError)
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	boolaFlag := f.BoolP("boola", "a", false, "bool value")
	boolbFlag := f.BoolP("boolb", "b", false, "bool2 value")
	boolcFlag := f.BoolP("boolc", "c", false, "bool3 value")
	booldFlag := f.BoolP("boold", "d", false, "bool4 value")
	stringaFlag := f.StringP("stringa", "s", "0", "string value")
	stringzFlag := f.StringP("stringz", "z", "0", "string value")
	extra := "interspersed-argument"
	notaflag := "--i-look-like-a-flag"
	args := []string{
		"-ab",
		extra,
		"-cs",
		"hello",
		"-z=something",
		"-d=true",
		"--",
		notaflag,
	}
	f.SetOutput(ioutil.Discard)
	if err := f.Parse(args); err != nil {
		t.Error("expected no error, got ", err)
	}
	if !f.Parsed() {
		t.Error("f.Parse() = false after Parse")
	}
	if *boolaFlag != true {
		t.Error("boola flag should be true, is ", *boolaFlag)
	}
	if *boolbFlag != true {
		t.Error("boolb flag should be true, is ", *boolbFlag)
	}
	if *boolcFlag != true {
		t.Error("boolc flag should be true, is ", *boolcFlag)
	}
	if *booldFlag != true {
		t.Error("boold flag should be true, is ", *booldFlag)
	}
	if *stringaFlag != "hello" {
		t.Error("stringa flag should be `hello`, is ", *stringaFlag)
	}
	if *stringzFlag != "something" {
		t.Error("stringz flag should be `something`, is ", *stringzFlag)
	}
	if len(f.Args()) != 2 {
		t.Error("expected one argument, got", len(f.Args()))
	} else if f.Args()[0] != extra {
		t.Errorf("expected argument %q got %q", extra, f.Args()[0])
	} else if f.Args()[1] != notaflag {
		t.Errorf("expected argument %q got %q", notaflag, f.Args()[1])
	}
	if f.ArgsLenAtDash() != 1 {
		t.Errorf("expected argsLenAtDash %d got %d", f.ArgsLenAtDash(), 1)
	}
}

func TestParse(t *testing.T) {
	ResetForTesting(func() { t.Error("bad parse") })
	testParse(GetCommandLine(), t)
}

func TestFlagSetParse(t *testing.T) {
	testParse(NewFlagSet("test", ContinueOnError), t)
}

func TestChangedHelper(t *testing.T) {
	f := NewFlagSet("changedtest", ContinueOnError)
	_ = f.Bool("changed", false, "changed bool")
	_ = f.Bool("settrue", true, "true to true")
	_ = f.Bool("setfalse", false, "false to false")
	_ = f.Bool("unchanged", false, "unchanged bool")

	args := []string{"--changed", "--settrue", "--setfalse=false"}
	if err := f.Parse(args); err != nil {
		t.Error("f.Parse() = false after Parse")
	}
	if !f.Changed("changed") {
		t.Errorf("--changed wasn't changed!")
	}
	if !f.Changed("settrue") {
		t.Errorf("--settrue wasn't changed!")
	}
	if !f.Changed("setfalse") {
		t.Errorf("--setfalse wasn't changed!")
	}
	if f.Changed("unchanged") {
		t.Errorf("--unchanged was changed!")
	}
	if f.Changed("invalid") {
		t.Errorf("--invalid was changed!")
	}
	if f.ArgsLenAtDash() != -1 {
		t.Errorf("Expected argsLenAtDash: %d but got %d", -1, f.ArgsLenAtDash())
	}
}

func replaceSeparators(name string, from []string, to string) string {
	result := name
	for _, sep := range from {
		result = strings.Replace(result, sep, to, -1)
	}
	// Type convert to indicate normalization has been done.
	return result
}

func wordSepNormalizeFunc(f *FlagSet, name string) NormalizedName {
	seps := []string{"-", "_"}
	name = replaceSeparators(name, seps, ".")
	normalizeFlagNameInvocations++

	return NormalizedName(name)
}

func testWordSepNormalizedNames(args []string, t *testing.T) {
	f := NewFlagSet("normalized", ContinueOnError)
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	withDashFlag := f.Bool("with-dash-flag", false, "bool value")
	// Set this after some flags have been added and before others.
	f.SetNormalizeFunc(wordSepNormalizeFunc)
	withUnderFlag := f.Bool("with_under_flag", false, "bool value")
	withBothFlag := f.Bool("with-both_flag", false, "bool value")
	if err := f.Parse(args); err != nil {
		t.Fatal(err)
	}
	if !f.Parsed() {
		t.Error("f.Parse() = false after Parse")
	}
	if *withDashFlag != true {
		t.Error("withDashFlag flag should be true, is ", *withDashFlag)
	}
	if *withUnderFlag != true {
		t.Error("withUnderFlag flag should be true, is ", *withUnderFlag)
	}
	if *withBothFlag != true {
		t.Error("withBothFlag flag should be true, is ", *withBothFlag)
	}
}

func TestWordSepNormalizedNames(t *testing.T) {
	args := []string{
		"--with-dash-flag",
		"--with-under-flag",
		"--with-both-flag",
	}
	testWordSepNormalizedNames(args, t)

	args = []string{
		"--with_dash_flag",
		"--with_under_flag",
		"--with_both_flag",
	}
	testWordSepNormalizedNames(args, t)

	args = []string{
		"--with-dash_flag",
		"--with-under_flag",
		"--with-both_flag",
	}
	testWordSepNormalizedNames(args, t)
}

func aliasAndWordSepFlagNames(f *FlagSet, name string) NormalizedName {
	seps := []string{"-", "_"}

	oldName := replaceSeparators("old-valid_flag", seps, ".")
	newName := replaceSeparators("valid-flag", seps, ".")

	name = replaceSeparators(name, seps, ".")
	switch name {
	case oldName:
		name = newName
		break
	}

	return NormalizedName(name)
}

func TestCustomNormalizedNames(t *testing.T) {
	f := NewFlagSet("normalized", ContinueOnError)
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}

	validFlag := f.Bool("valid-flag", false, "bool value")
	f.SetNormalizeFunc(aliasAndWordSepFlagNames)
	someOtherFlag := f.Bool("some-other-flag", false, "bool value")

	args := []string{"--old_valid_flag", "--some-other_flag"}
	if err := f.Parse(args); err != nil {
		t.Fatal(err)
	}

	if *validFlag != true {
		t.Errorf("validFlag is %v even though we set the alias --old_valid_falg", *validFlag)
	}
	if *someOtherFlag != true {
		t.Error("someOtherFlag should be true, is ", *someOtherFlag)
	}
}

// Every flag we add, the name (displayed also in usage) should normalized
func TestNormalizationFuncShouldChangeFlagName(t *testing.T) {
	// Test normalization after addition
	f := NewFlagSet("normalized", ContinueOnError)

	f.Bool("valid_flag", false, "bool value")
	if f.Lookup("valid_flag").Name != "valid_flag" {
		t.Error("The new flag should have the name 'valid_flag' instead of ", f.Lookup("valid_flag").Name)
	}

	f.SetNormalizeFunc(wordSepNormalizeFunc)
	if f.Lookup("valid_flag").Name != "valid.flag" {
		t.Error("The new flag should have the name 'valid.flag' instead of ", f.Lookup("valid_flag").Name)
	}

	// Test normalization before addition
	f = NewFlagSet("normalized", ContinueOnError)
	f.SetNormalizeFunc(wordSepNormalizeFunc)

	f.Bool("valid_flag", false, "bool value")
	if f.Lookup("valid_flag").Name != "valid.flag" {
		t.Error("The new flag should have the name 'valid.flag' instead of ", f.Lookup("valid_flag").Name)
	}
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

func (f *flagVar) Type() string {
	return "flagVar"
}

func TestUserDefined(t *testing.T) {
	var flags FlagSet
	flags.Init("test", ContinueOnError)
	var v flagVar
	flags.VarP(&v, "v", "v", "usage")
	if err := flags.Parse([]string{"--v=1", "-v2", "-v", "3"}); err != nil {
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

func TestSetOutput(t *testing.T) {
	var flags FlagSet
	var buf bytes.Buffer
	flags.SetOutput(&buf)
	flags.Init("test", ContinueOnError)
	flags.Parse([]string{"--unknown"})
	if out := buf.String(); !strings.Contains(out, "--unknown") {
		t.Logf("expected output mentioning unknown; got %q", out)
	}
}

// This tests that one can reset the flags. This still works but not well, and is
// superseded by FlagSet.
func TestChangingArgs(t *testing.T) {
	ResetForTesting(func() { t.Fatal("bad parse") })
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"cmd", "--before", "subcmd"}
	before := Bool("before", false, "")
	if err := GetCommandLine().Parse(os.Args[1:]); err != nil {
		t.Fatal(err)
	}
	cmd := Arg(0)
	os.Args = []string{"subcmd", "--after", "args"}
	after := Bool("after", false, "")
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
	fs.BoolVar(&flag, "flag", false, "regular flag")
	// Regular flag invocation should work
	err := fs.Parse([]string{"--flag=true"})
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}
	if !flag {
		t.Error("flag was not set by --flag")
	}
	if helpCalled {
		t.Error("help called for regular flag")
		helpCalled = false // reset for next test
	}
	// Help flag should work as expected.
	err = fs.Parse([]string{"--help"})
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
	fs.BoolVar(&help, "help", false, "help flag")
	helpCalled = false
	err = fs.Parse([]string{"--help"})
	if err != nil {
		t.Fatal("expected no error for defined --help; got ", err)
	}
	if helpCalled {
		t.Fatal("help was called; should not have been for defined help flag")
	}
}

func TestNoInterspersed(t *testing.T) {
	f := NewFlagSet("test", ContinueOnError)
	f.SetInterspersed(false)
	f.Bool("true", true, "always true")
	f.Bool("false", false, "always false")
	err := f.Parse([]string{"--true", "break", "--false"})
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}
	args := f.Args()
	if len(args) != 2 || args[0] != "break" || args[1] != "--false" {
		t.Fatal("expected interspersed options/non-options to fail")
	}
}

func TestTermination(t *testing.T) {
	f := NewFlagSet("termination", ContinueOnError)
	boolFlag := f.BoolP("bool", "l", false, "bool value")
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	arg1 := "ls"
	arg2 := "-l"
	args := []string{
		"--",
		arg1,
		arg2,
	}
	f.SetOutput(ioutil.Discard)
	if err := f.Parse(args); err != nil {
		t.Fatal("expected no error; got ", err)
	}
	if !f.Parsed() {
		t.Error("f.Parse() = false after Parse")
	}
	if *boolFlag {
		t.Error("expected boolFlag=false, got true")
	}
	if len(f.Args()) != 2 {
		t.Errorf("expected 2 arguments, got %d: %v", len(f.Args()), f.Args())
	}
	if f.Args()[0] != arg1 {
		t.Errorf("expected argument %q got %q", arg1, f.Args()[0])
	}
	if f.Args()[1] != arg2 {
		t.Errorf("expected argument %q got %q", arg2, f.Args()[1])
	}
	if f.ArgsLenAtDash() != 0 {
		t.Errorf("expected argsLenAtDash %d got %d", 0, f.ArgsLenAtDash())
	}
}

func TestDeprecatedFlagInDocs(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	f.Bool("badflag", true, "always true")
	f.MarkDeprecated("badflag", "use --good-flag instead")

	out := new(bytes.Buffer)
	f.SetOutput(out)
	f.PrintDefaults()

	if strings.Contains(out.String(), "badflag") {
		t.Errorf("found deprecated flag in usage!")
	}
}

func TestDeprecatedFlagShorthandInDocs(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	name := "noshorthandflag"
	f.BoolP(name, "n", true, "always true")
	f.MarkShorthandDeprecated("noshorthandflag", fmt.Sprintf("use --%s instead", name))

	out := new(bytes.Buffer)
	f.SetOutput(out)
	f.PrintDefaults()

	if strings.Contains(out.String(), "-n,") {
		t.Errorf("found deprecated flag shorthand in usage!")
	}
}

func parseReturnStderr(t *testing.T, f *FlagSet, args []string) (string, error) {
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	err := f.Parse(args)

	outC := make(chan string)
	// copy the output in a separate goroutine so printing can't block indefinitely
	go func() {
		var buf bytes.Buffer
		io.Copy(&buf, r)
		outC <- buf.String()
	}()

	w.Close()
	os.Stderr = oldStderr
	out := <-outC

	return out, err
}

func TestDeprecatedFlagUsage(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	f.Bool("badflag", true, "always true")
	usageMsg := "use --good-flag instead"
	f.MarkDeprecated("badflag", usageMsg)

	args := []string{"--badflag"}
	out, err := parseReturnStderr(t, f, args)
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}

	if !strings.Contains(out, usageMsg) {
		t.Errorf("usageMsg not printed when using a deprecated flag!")
	}
}

func TestDeprecatedFlagShorthandUsage(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	name := "noshorthandflag"
	f.BoolP(name, "n", true, "always true")
	usageMsg := fmt.Sprintf("use --%s instead", name)
	f.MarkShorthandDeprecated(name, usageMsg)

	args := []string{"-n"}
	out, err := parseReturnStderr(t, f, args)
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}

	if !strings.Contains(out, usageMsg) {
		t.Errorf("usageMsg not printed when using a deprecated flag!")
	}
}

func TestDeprecatedFlagUsageNormalized(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	f.Bool("bad-double_flag", true, "always true")
	f.SetNormalizeFunc(wordSepNormalizeFunc)
	usageMsg := "use --good-flag instead"
	f.MarkDeprecated("bad_double-flag", usageMsg)

	args := []string{"--bad_double_flag"}
	out, err := parseReturnStderr(t, f, args)
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}

	if !strings.Contains(out, usageMsg) {
		t.Errorf("usageMsg not printed when using a deprecated flag!")
	}
}

// Name normalization function should be called only once on flag addition
func TestMultipleNormalizeFlagNameInvocations(t *testing.T) {
	normalizeFlagNameInvocations = 0

	f := NewFlagSet("normalized", ContinueOnError)
	f.SetNormalizeFunc(wordSepNormalizeFunc)
	f.Bool("with_under_flag", false, "bool value")

	if normalizeFlagNameInvocations != 1 {
		t.Fatal("Expected normalizeFlagNameInvocations to be 1; got ", normalizeFlagNameInvocations)
	}
}

//
func TestHiddenFlagInUsage(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	f.Bool("secretFlag", true, "shhh")
	f.MarkHidden("secretFlag")

	out := new(bytes.Buffer)
	f.SetOutput(out)
	f.PrintDefaults()

	if strings.Contains(out.String(), "secretFlag") {
		t.Errorf("found hidden flag in usage!")
	}
}

//
func TestHiddenFlagUsage(t *testing.T) {
	f := NewFlagSet("bob", ContinueOnError)
	f.Bool("secretFlag", true, "shhh")
	f.MarkHidden("secretFlag")

	args := []string{"--secretFlag"}
	out, err := parseReturnStderr(t, f, args)
	if err != nil {
		t.Fatal("expected no error; got ", err)
	}

	if strings.Contains(out, "shhh") {
		t.Errorf("usage message printed when using a hidden flag!")
	}
}
