// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

var (
	test_bool                    = Bool("test_bool", false, "bool value")
	test_int                     = Int("test_int", 0, "int value")
	test_int64                   = Int64("test_int64", 0, "int64 value")
	test_uint                    = Uint("test_uint", 0, "uint value")
	test_uint64                  = Uint64("test_uint64", 0, "uint64 value")
	test_string                  = String("test_string", "0", "string value")
	test_float64                 = Float64("test_float64", 0, "float64 value")
	test_duration                = Duration("test_duration", 0, "time.Duration value")
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

func testParse(f *FlagSet, t *testing.T) {
	if f.Parsed() {
		t.Error("f.Parse() = true before Parse")
	}
	boolFlag := f.Bool("bool", false, "bool value")
	bool2Flag := f.Bool("bool2", false, "bool2 value")
	bool3Flag := f.Bool("bool3", false, "bool3 value")
	intFlag := f.Int("int", 0, "int value")
	int64Flag := f.Int64("int64", 0, "int64 value")
	uintFlag := f.Uint("uint", 0, "uint value")
	uint64Flag := f.Uint64("uint64", 0, "uint64 value")
	stringFlag := f.String("string", "0", "string value")
	float64Flag := f.Float64("float64", 0, "float64 value")
	durationFlag := f.Duration("duration", 5*time.Second, "time.Duration value")
	extra := "one-extra-argument"
	args := []string{
		"--bool",
		"--bool2=true",
		"--bool3=false",
		"--int=22",
		"--int64=0x23",
		"--uint=24",
		"--uint64=25",
		"--string=hello",
		"--float64=2718e28",
		"--duration=2m",
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
	if *bool3Flag != false {
		t.Error("bool3 flag should be false, is ", *bool2Flag)
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
}

func TestParse(t *testing.T) {
	ResetForTesting(func() { t.Error("bad parse") })
	testParse(GetCommandLine(), t)
}

func TestFlagSetParse(t *testing.T) {
	testParse(NewFlagSet("test", ContinueOnError), t)
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
