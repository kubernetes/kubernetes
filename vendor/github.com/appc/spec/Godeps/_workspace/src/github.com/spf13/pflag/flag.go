// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	pflag is a drop-in replacement for Go's flag package, implementing
	POSIX/GNU-style --flags.

	pflag is compatible with the GNU extensions to the POSIX recommendations
	for command-line options. See
	http://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html

	Usage:

	pflag is a drop-in replacement of Go's native flag package. If you import
	pflag under the name "flag" then all code should continue to function
	with no changes.

		import flag "github.com/ogier/pflag"

	There is one exception to this: if you directly instantiate the Flag struct
	there is one more field "Shorthand" that you will need to set.
	Most code never instantiates this struct directly, and instead uses
	functions such as String(), BoolVar(), and Var(), and is therefore
	unaffected.

	Define flags using flag.String(), Bool(), Int(), etc.

	This declares an integer flag, -flagname, stored in the pointer ip, with type *int.
		var ip = flag.Int("flagname", 1234, "help message for flagname")
	If you like, you can bind the flag to a variable using the Var() functions.
		var flagvar int
		func init() {
			flag.IntVar(&flagvar, "flagname", 1234, "help message for flagname")
		}
	Or you can create custom flags that satisfy the Value interface (with
	pointer receivers) and couple them to flag parsing by
		flag.Var(&flagVal, "name", "help message for flagname")
	For such flags, the default value is just the initial value of the variable.

	After all flags are defined, call
		flag.Parse()
	to parse the command line into the defined flags.

	Flags may then be used directly. If you're using the flags themselves,
	they are all pointers; if you bind to variables, they're values.
		fmt.Println("ip has value ", *ip)
		fmt.Println("flagvar has value ", flagvar)

	After parsing, the arguments after the flag are available as the
	slice flag.Args() or individually as flag.Arg(i).
	The arguments are indexed from 0 through flag.NArg()-1.

	The pflag package also defines some new functions that are not in flag,
	that give one-letter shorthands for flags. You can use these by appending
	'P' to the name of any function that defines a flag.
		var ip = flag.IntP("flagname", "f", 1234, "help message")
		var flagvar bool
		func init() {
			flag.BoolVarP("boolname", "b", true, "help message")
		}
		flag.VarP(&flagVar, "varname", "v", 1234, "help message")
	Shorthand letters can be used with single dashes on the command line.
	Boolean shorthand flags can be combined with other shorthand flags.

	Command line flag syntax:
		--flag    // boolean flags only
		--flag=x

	Unlike the flag package, a single dash before an option means something
	different than a double dash. Single dashes signify a series of shorthand
	letters for flags. All but the last shorthand letter must be boolean flags.
		// boolean flags
		-f
		-abc
		// non-boolean flags
		-n 1234
		-Ifile
		// mixed
		-abcs "hello"
		-abcn1234

	Flag parsing stops after the terminator "--". Unlike the flag package,
	flags can be interspersed with arguments anywhere on the command line
	before this terminator.

	Integer flags accept 1234, 0664, 0x1234 and may be negative.
	Boolean flags (in their long form) accept 1, 0, t, f, true, false,
	TRUE, FALSE, True, False.
	Duration flags accept any input valid for time.ParseDuration.

	The default set of command-line flags is controlled by
	top-level functions.  The FlagSet type allows one to define
	independent sets of flags, such as to implement subcommands
	in a command-line interface. The methods of FlagSet are
	analogous to the top-level functions for the command-line
	flag set.
*/
package pflag

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ErrHelp is the error returned if the flag -help is invoked but no such flag is defined.
var ErrHelp = errors.New("pflag: help requested")

// -- bool Value
type boolValue bool

func newBoolValue(val bool, p *bool) *boolValue {
	*p = val
	return (*boolValue)(p)
}

func (b *boolValue) Set(s string) error {
	v, err := strconv.ParseBool(s)
	*b = boolValue(v)
	return err
}

func (b *boolValue) String() string { return fmt.Sprintf("%v", *b) }

// -- int Value
type intValue int

func newIntValue(val int, p *int) *intValue {
	*p = val
	return (*intValue)(p)
}

func (i *intValue) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	*i = intValue(v)
	return err
}

func (i *intValue) String() string { return fmt.Sprintf("%v", *i) }

// -- int64 Value
type int64Value int64

func newInt64Value(val int64, p *int64) *int64Value {
	*p = val
	return (*int64Value)(p)
}

func (i *int64Value) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	*i = int64Value(v)
	return err
}

func (i *int64Value) String() string { return fmt.Sprintf("%v", *i) }

// -- uint Value
type uintValue uint

func newUintValue(val uint, p *uint) *uintValue {
	*p = val
	return (*uintValue)(p)
}

func (i *uintValue) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 64)
	*i = uintValue(v)
	return err
}

func (i *uintValue) String() string { return fmt.Sprintf("%v", *i) }

// -- uint64 Value
type uint64Value uint64

func newUint64Value(val uint64, p *uint64) *uint64Value {
	*p = val
	return (*uint64Value)(p)
}

func (i *uint64Value) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 64)
	*i = uint64Value(v)
	return err
}

func (i *uint64Value) String() string { return fmt.Sprintf("%v", *i) }

// -- string Value
type stringValue string

func newStringValue(val string, p *string) *stringValue {
	*p = val
	return (*stringValue)(p)
}

func (s *stringValue) Set(val string) error {
	*s = stringValue(val)
	return nil
}

func (s *stringValue) String() string { return fmt.Sprintf("%s", *s) }

// -- float64 Value
type float64Value float64

func newFloat64Value(val float64, p *float64) *float64Value {
	*p = val
	return (*float64Value)(p)
}

func (f *float64Value) Set(s string) error {
	v, err := strconv.ParseFloat(s, 64)
	*f = float64Value(v)
	return err
}

func (f *float64Value) String() string { return fmt.Sprintf("%v", *f) }

// -- time.Duration Value
type durationValue time.Duration

func newDurationValue(val time.Duration, p *time.Duration) *durationValue {
	*p = val
	return (*durationValue)(p)
}

func (d *durationValue) Set(s string) error {
	v, err := time.ParseDuration(s)
	*d = durationValue(v)
	return err
}

func (d *durationValue) String() string { return (*time.Duration)(d).String() }

// Value is the interface to the dynamic value stored in a flag.
// (The default value is represented as a string.)
type Value interface {
	String() string
	Set(string) error
}

// ErrorHandling defines how to handle flag parsing errors.
type ErrorHandling int

const (
	ContinueOnError ErrorHandling = iota
	ExitOnError
	PanicOnError
)

// A FlagSet represents a set of defined flags.
type FlagSet struct {
	// Usage is the function called when an error occurs while parsing flags.
	// The field is a function (not a method) that may be changed to point to
	// a custom error handler.
	Usage func()

	name          string
	parsed        bool
	actual        map[string]*Flag
	formal        map[string]*Flag
	shorthands    map[byte]*Flag
	args          []string // arguments after flags
	exitOnError   bool     // does the program exit if there's an error?
	errorHandling ErrorHandling
	output        io.Writer // nil means stderr; use out() accessor
	interspersed  bool      // allow interspersed option/non-option args
}

// A Flag represents the state of a flag.
type Flag struct {
	Name      string // name as it appears on command line
	Shorthand string // one-letter abbreviated flag
	Usage     string // help message
	Value     Value  // value as set
	DefValue  string // default value (as text); for usage message
	Changed   bool   // If the user set the value (or if left to default)
}

// sortFlags returns the flags as a slice in lexicographical sorted order.
func sortFlags(flags map[string]*Flag) []*Flag {
	list := make(sort.StringSlice, len(flags))
	i := 0
	for _, f := range flags {
		list[i] = f.Name
		i++
	}
	list.Sort()
	result := make([]*Flag, len(list))
	for i, name := range list {
		result[i] = flags[name]
	}
	return result
}

func (f *FlagSet) out() io.Writer {
	if f.output == nil {
		return os.Stderr
	}
	return f.output
}

// SetOutput sets the destination for usage and error messages.
// If output is nil, os.Stderr is used.
func (f *FlagSet) SetOutput(output io.Writer) {
	f.output = output
}

// VisitAll visits the flags in lexicographical order, calling fn for each.
// It visits all flags, even those not set.
func (f *FlagSet) VisitAll(fn func(*Flag)) {
	for _, flag := range sortFlags(f.formal) {
		fn(flag)
	}
}

func (f *FlagSet) HasFlags() bool {
	return len(f.formal) > 0
}

// VisitAll visits the command-line flags in lexicographical order, calling
// fn for each.  It visits all flags, even those not set.
func VisitAll(fn func(*Flag)) {
	commandLine.VisitAll(fn)
}

// Visit visits the flags in lexicographical order, calling fn for each.
// It visits only those flags that have been set.
func (f *FlagSet) Visit(fn func(*Flag)) {
	for _, flag := range sortFlags(f.actual) {
		fn(flag)
	}
}

// Visit visits the command-line flags in lexicographical order, calling fn
// for each.  It visits only those flags that have been set.
func Visit(fn func(*Flag)) {
	commandLine.Visit(fn)
}

// Lookup returns the Flag structure of the named flag, returning nil if none exists.
func (f *FlagSet) Lookup(name string) *Flag {
	return f.formal[name]
}

// Lookup returns the Flag structure of the named command-line flag,
// returning nil if none exists.
func Lookup(name string) *Flag {
	return commandLine.formal[name]
}

// Set sets the value of the named flag.
func (f *FlagSet) Set(name, value string) error {
	flag, ok := f.formal[name]
	if !ok {
		return fmt.Errorf("no such flag -%v", name)
	}
	err := flag.Value.Set(value)
	if err != nil {
		return err
	}
	if f.actual == nil {
		f.actual = make(map[string]*Flag)
	}
	f.actual[name] = flag
	f.Lookup(name).Changed = true
	return nil
}

// Set sets the value of the named command-line flag.
func Set(name, value string) error {
	return commandLine.Set(name, value)
}

// PrintDefaults prints, to standard error unless configured
// otherwise, the default values of all defined flags in the set.
func (f *FlagSet) PrintDefaults() {
	f.VisitAll(func(flag *Flag) {
		format := "--%s=%s: %s\n"
		if _, ok := flag.Value.(*stringValue); ok {
			// put quotes on the value
			format = "--%s=%q: %s\n"
		}
		if len(flag.Shorthand) > 0 {
			format = "  -%s, " + format
		} else {
			format = "   %s   " + format
		}
		fmt.Fprintf(f.out(), format, flag.Shorthand, flag.Name, flag.DefValue, flag.Usage)
	})
}

func (f *FlagSet) FlagUsages() string {
	x := new(bytes.Buffer)

	f.VisitAll(func(flag *Flag) {
		format := "--%s=%s: %s\n"
		if _, ok := flag.Value.(*stringValue); ok {
			// put quotes on the value
			format = "--%s=%q: %s\n"
		}
		if len(flag.Shorthand) > 0 {
			format = "  -%s, " + format
		} else {
			format = "   %s   " + format
		}
		fmt.Fprintf(x, format, flag.Shorthand, flag.Name, flag.DefValue, flag.Usage)
	})

	return x.String()
}

// PrintDefaults prints to standard error the default values of all defined command-line flags.
func PrintDefaults() {
	commandLine.PrintDefaults()
}

// defaultUsage is the default function to print a usage message.
func defaultUsage(f *FlagSet) {
	fmt.Fprintf(f.out(), "Usage of %s:\n", f.name)
	f.PrintDefaults()
}

// NOTE: Usage is not just defaultUsage(commandLine)
// because it serves (via godoc flag Usage) as the example
// for how to write your own usage function.

// Usage prints to standard error a usage message documenting all defined command-line flags.
// The function is a variable that may be changed to point to a custom function.
var Usage = func() {
	fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
	PrintDefaults()
}

// NFlag returns the number of flags that have been set.
func (f *FlagSet) NFlag() int { return len(f.actual) }

// NFlag returns the number of command-line flags that have been set.
func NFlag() int { return len(commandLine.actual) }

// Arg returns the i'th argument.  Arg(0) is the first remaining argument
// after flags have been processed.
func (f *FlagSet) Arg(i int) string {
	if i < 0 || i >= len(f.args) {
		return ""
	}
	return f.args[i]
}

// Arg returns the i'th command-line argument.  Arg(0) is the first remaining argument
// after flags have been processed.
func Arg(i int) string {
	return commandLine.Arg(i)
}

// NArg is the number of arguments remaining after flags have been processed.
func (f *FlagSet) NArg() int { return len(f.args) }

// NArg is the number of arguments remaining after flags have been processed.
func NArg() int { return len(commandLine.args) }

// Args returns the non-flag arguments.
func (f *FlagSet) Args() []string { return f.args }

// Args returns the non-flag command-line arguments.
func Args() []string { return commandLine.args }

// BoolVar defines a bool flag with specified name, default value, and usage string.
// The argument p points to a bool variable in which to store the value of the flag.
func (f *FlagSet) BoolVar(p *bool, name string, value bool, usage string) {
	f.VarP(newBoolValue(value, p), name, "", usage)
}

// Like BoolVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) BoolVarP(p *bool, name, shorthand string, value bool, usage string) {
	f.VarP(newBoolValue(value, p), name, shorthand, usage)
}

// BoolVar defines a bool flag with specified name, default value, and usage string.
// The argument p points to a bool variable in which to store the value of the flag.
func BoolVar(p *bool, name string, value bool, usage string) {
	commandLine.VarP(newBoolValue(value, p), name, "", usage)
}

// Like BoolVar, but accepts a shorthand letter that can be used after a single dash.
func BoolVarP(p *bool, name, shorthand string, value bool, usage string) {
	commandLine.VarP(newBoolValue(value, p), name, shorthand, usage)
}

// Bool defines a bool flag with specified name, default value, and usage string.
// The return value is the address of a bool variable that stores the value of the flag.
func (f *FlagSet) Bool(name string, value bool, usage string) *bool {
	p := new(bool)
	f.BoolVarP(p, name, "", value, usage)
	return p
}

// Like Bool, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) BoolP(name, shorthand string, value bool, usage string) *bool {
	p := new(bool)
	f.BoolVarP(p, name, shorthand, value, usage)
	return p
}

// Bool defines a bool flag with specified name, default value, and usage string.
// The return value is the address of a bool variable that stores the value of the flag.
func Bool(name string, value bool, usage string) *bool {
	return commandLine.BoolP(name, "", value, usage)
}

// Like Bool, but accepts a shorthand letter that can be used after a single dash.
func BoolP(name, shorthand string, value bool, usage string) *bool {
	return commandLine.BoolP(name, shorthand, value, usage)
}

// IntVar defines an int flag with specified name, default value, and usage string.
// The argument p points to an int variable in which to store the value of the flag.
func (f *FlagSet) IntVar(p *int, name string, value int, usage string) {
	f.VarP(newIntValue(value, p), name, "", usage)
}

// Like IntVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IntVarP(p *int, name, shorthand string, value int, usage string) {
	f.VarP(newIntValue(value, p), name, shorthand, usage)
}

// IntVar defines an int flag with specified name, default value, and usage string.
// The argument p points to an int variable in which to store the value of the flag.
func IntVar(p *int, name string, value int, usage string) {
	commandLine.VarP(newIntValue(value, p), name, "", usage)
}

// Like IntVar, but accepts a shorthand letter that can be used after a single dash.
func IntVarP(p *int, name, shorthand string, value int, usage string) {
	commandLine.VarP(newIntValue(value, p), name, shorthand, usage)
}

// Int defines an int flag with specified name, default value, and usage string.
// The return value is the address of an int variable that stores the value of the flag.
func (f *FlagSet) Int(name string, value int, usage string) *int {
	p := new(int)
	f.IntVarP(p, name, "", value, usage)
	return p
}

// Like Int, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IntP(name, shorthand string, value int, usage string) *int {
	p := new(int)
	f.IntVarP(p, name, shorthand, value, usage)
	return p
}

// Int defines an int flag with specified name, default value, and usage string.
// The return value is the address of an int variable that stores the value of the flag.
func Int(name string, value int, usage string) *int {
	return commandLine.IntP(name, "", value, usage)
}

// Like Int, but accepts a shorthand letter that can be used after a single dash.
func IntP(name, shorthand string, value int, usage string) *int {
	return commandLine.IntP(name, shorthand, value, usage)
}

// Int64Var defines an int64 flag with specified name, default value, and usage string.
// The argument p points to an int64 variable in which to store the value of the flag.
func (f *FlagSet) Int64Var(p *int64, name string, value int64, usage string) {
	f.VarP(newInt64Value(value, p), name, "", usage)
}

// Like Int64Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Int64VarP(p *int64, name, shorthand string, value int64, usage string) {
	f.VarP(newInt64Value(value, p), name, shorthand, usage)
}

// Int64Var defines an int64 flag with specified name, default value, and usage string.
// The argument p points to an int64 variable in which to store the value of the flag.
func Int64Var(p *int64, name string, value int64, usage string) {
	commandLine.VarP(newInt64Value(value, p), name, "", usage)
}

// Like Int64Var, but accepts a shorthand letter that can be used after a single dash.
func Int64VarP(p *int64, name, shorthand string, value int64, usage string) {
	commandLine.VarP(newInt64Value(value, p), name, shorthand, usage)
}

// Int64 defines an int64 flag with specified name, default value, and usage string.
// The return value is the address of an int64 variable that stores the value of the flag.
func (f *FlagSet) Int64(name string, value int64, usage string) *int64 {
	p := new(int64)
	f.Int64VarP(p, name, "", value, usage)
	return p
}

// Like Int64, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Int64P(name, shorthand string, value int64, usage string) *int64 {
	p := new(int64)
	f.Int64VarP(p, name, shorthand, value, usage)
	return p
}

// Int64 defines an int64 flag with specified name, default value, and usage string.
// The return value is the address of an int64 variable that stores the value of the flag.
func Int64(name string, value int64, usage string) *int64 {
	return commandLine.Int64P(name, "", value, usage)
}

// Like Int64, but accepts a shorthand letter that can be used after a single dash.
func Int64P(name, shorthand string, value int64, usage string) *int64 {
	return commandLine.Int64P(name, shorthand, value, usage)
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint variable in which to store the value of the flag.
func (f *FlagSet) UintVar(p *uint, name string, value uint, usage string) {
	f.VarP(newUintValue(value, p), name, "", usage)
}

// Like UintVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintVarP(p *uint, name, shorthand string, value uint, usage string) {
	f.VarP(newUintValue(value, p), name, shorthand, usage)
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint  variable in which to store the value of the flag.
func UintVar(p *uint, name string, value uint, usage string) {
	commandLine.VarP(newUintValue(value, p), name, "", usage)
}

// Like UintVar, but accepts a shorthand letter that can be used after a single dash.
func UintVarP(p *uint, name, shorthand string, value uint, usage string) {
	commandLine.VarP(newUintValue(value, p), name, shorthand, usage)
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint  variable that stores the value of the flag.
func (f *FlagSet) Uint(name string, value uint, usage string) *uint {
	p := new(uint)
	f.UintVarP(p, name, "", value, usage)
	return p
}

// Like Uint, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintP(name, shorthand string, value uint, usage string) *uint {
	p := new(uint)
	f.UintVarP(p, name, shorthand, value, usage)
	return p
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint  variable that stores the value of the flag.
func Uint(name string, value uint, usage string) *uint {
	return commandLine.UintP(name, "", value, usage)
}

// Like Uint, but accepts a shorthand letter that can be used after a single dash.
func UintP(name, shorthand string, value uint, usage string) *uint {
	return commandLine.UintP(name, shorthand, value, usage)
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func (f *FlagSet) Uint64Var(p *uint64, name string, value uint64, usage string) {
	f.VarP(newUint64Value(value, p), name, "", usage)
}

// Like Uint64Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint64VarP(p *uint64, name, shorthand string, value uint64, usage string) {
	f.VarP(newUint64Value(value, p), name, shorthand, usage)
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func Uint64Var(p *uint64, name string, value uint64, usage string) {
	commandLine.VarP(newUint64Value(value, p), name, "", usage)
}

// Like Uint64Var, but accepts a shorthand letter that can be used after a single dash.
func Uint64VarP(p *uint64, name, shorthand string, value uint64, usage string) {
	commandLine.VarP(newUint64Value(value, p), name, shorthand, usage)
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func (f *FlagSet) Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64)
	f.Uint64VarP(p, name, "", value, usage)
	return p
}

// Like Uint64, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint64P(name, shorthand string, value uint64, usage string) *uint64 {
	p := new(uint64)
	f.Uint64VarP(p, name, shorthand, value, usage)
	return p
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func Uint64(name string, value uint64, usage string) *uint64 {
	return commandLine.Uint64P(name, "", value, usage)
}

// Like Uint64, but accepts a shorthand letter that can be used after a single dash.
func Uint64P(name, shorthand string, value uint64, usage string) *uint64 {
	return commandLine.Uint64P(name, shorthand, value, usage)
}

// StringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a string variable in which to store the value of the flag.
func (f *FlagSet) StringVar(p *string, name string, value string, usage string) {
	f.VarP(newStringValue(value, p), name, "", usage)
}

// Like StringVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringVarP(p *string, name, shorthand string, value string, usage string) {
	f.VarP(newStringValue(value, p), name, shorthand, usage)
}

// StringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a string variable in which to store the value of the flag.
func StringVar(p *string, name string, value string, usage string) {
	commandLine.VarP(newStringValue(value, p), name, "", usage)
}

// Like StringVar, but accepts a shorthand letter that can be used after a single dash.
func StringVarP(p *string, name, shorthand string, value string, usage string) {
	commandLine.VarP(newStringValue(value, p), name, shorthand, usage)
}

// String defines a string flag with specified name, default value, and usage string.
// The return value is the address of a string variable that stores the value of the flag.
func (f *FlagSet) String(name string, value string, usage string) *string {
	p := new(string)
	f.StringVarP(p, name, "", value, usage)
	return p
}

// Like String, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringP(name, shorthand string, value string, usage string) *string {
	p := new(string)
	f.StringVarP(p, name, shorthand, value, usage)
	return p
}

// String defines a string flag with specified name, default value, and usage string.
// The return value is the address of a string variable that stores the value of the flag.
func String(name string, value string, usage string) *string {
	return commandLine.StringP(name, "", value, usage)
}

// Like String, but accepts a shorthand letter that can be used after a single dash.
func StringP(name, shorthand string, value string, usage string) *string {
	return commandLine.StringP(name, shorthand, value, usage)
}

// Float64Var defines a float64 flag with specified name, default value, and usage string.
// The argument p points to a float64 variable in which to store the value of the flag.
func (f *FlagSet) Float64Var(p *float64, name string, value float64, usage string) {
	f.VarP(newFloat64Value(value, p), name, "", usage)
}

// Like Float64Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Float64VarP(p *float64, name, shorthand string, value float64, usage string) {
	f.VarP(newFloat64Value(value, p), name, shorthand, usage)
}

// Float64Var defines a float64 flag with specified name, default value, and usage string.
// The argument p points to a float64 variable in which to store the value of the flag.
func Float64Var(p *float64, name string, value float64, usage string) {
	commandLine.VarP(newFloat64Value(value, p), name, "", usage)
}

// Like Float64Var, but accepts a shorthand letter that can be used after a single dash.
func Float64VarP(p *float64, name, shorthand string, value float64, usage string) {
	commandLine.VarP(newFloat64Value(value, p), name, shorthand, usage)
}

// Float64 defines a float64 flag with specified name, default value, and usage string.
// The return value is the address of a float64 variable that stores the value of the flag.
func (f *FlagSet) Float64(name string, value float64, usage string) *float64 {
	p := new(float64)
	f.Float64VarP(p, name, "", value, usage)
	return p
}

// Like Float64, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Float64P(name, shorthand string, value float64, usage string) *float64 {
	p := new(float64)
	f.Float64VarP(p, name, shorthand, value, usage)
	return p
}

// Float64 defines a float64 flag with specified name, default value, and usage string.
// The return value is the address of a float64 variable that stores the value of the flag.
func Float64(name string, value float64, usage string) *float64 {
	return commandLine.Float64P(name, "", value, usage)
}

// Like Float64, but accepts a shorthand letter that can be used after a single dash.
func Float64P(name, shorthand string, value float64, usage string) *float64 {
	return commandLine.Float64P(name, shorthand, value, usage)
}

// DurationVar defines a time.Duration flag with specified name, default value, and usage string.
// The argument p points to a time.Duration variable in which to store the value of the flag.
func (f *FlagSet) DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	f.VarP(newDurationValue(value, p), name, "", usage)
}

// Like DurationVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationVarP(p *time.Duration, name, shorthand string, value time.Duration, usage string) {
	f.VarP(newDurationValue(value, p), name, shorthand, usage)
}

// DurationVar defines a time.Duration flag with specified name, default value, and usage string.
// The argument p points to a time.Duration variable in which to store the value of the flag.
func DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	commandLine.VarP(newDurationValue(value, p), name, "", usage)
}

// Like DurationVar, but accepts a shorthand letter that can be used after a single dash.
func DurationVarP(p *time.Duration, name, shorthand string, value time.Duration, usage string) {
	commandLine.VarP(newDurationValue(value, p), name, shorthand, usage)
}

// Duration defines a time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a time.Duration variable that stores the value of the flag.
func (f *FlagSet) Duration(name string, value time.Duration, usage string) *time.Duration {
	p := new(time.Duration)
	f.DurationVarP(p, name, "", value, usage)
	return p
}

// Like Duration, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationP(name, shorthand string, value time.Duration, usage string) *time.Duration {
	p := new(time.Duration)
	f.DurationVarP(p, name, shorthand, value, usage)
	return p
}

// Duration defines a time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a time.Duration variable that stores the value of the flag.
func Duration(name string, value time.Duration, usage string) *time.Duration {
	return commandLine.DurationP(name, "", value, usage)
}

// Like Duration, but accepts a shorthand letter that can be used after a single dash.
func DurationP(name, shorthand string, value time.Duration, usage string) *time.Duration {
	return commandLine.DurationP(name, shorthand, value, usage)
}

// Var defines a flag with the specified name and usage string. The type and
// value of the flag are represented by the first argument, of type Value, which
// typically holds a user-defined implementation of Value. For instance, the
// caller could create a flag that turns a comma-separated string into a slice
// of strings by giving the slice the methods of Value; in particular, Set would
// decompose the comma-separated string into the slice.
func (f *FlagSet) Var(value Value, name string, usage string) {
	f.VarP(value, name, "", usage)
}

// Like Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) VarP(value Value, name, shorthand, usage string) {
	// Remember the default value as a string; it won't change.
	flag := &Flag{name, shorthand, usage, value, value.String(), false}
	f.AddFlag(flag)
}

func (f *FlagSet) AddFlag(flag *Flag) {
	_, alreadythere := f.formal[flag.Name]
	if alreadythere {
		msg := fmt.Sprintf("%s flag redefined: %s", f.name, flag.Name)
		fmt.Fprintln(f.out(), msg)
		panic(msg) // Happens only if flags are declared with identical names
	}
	if f.formal == nil {
		f.formal = make(map[string]*Flag)
	}
	f.formal[flag.Name] = flag

	if len(flag.Shorthand) == 0 {
		return
	}
	if len(flag.Shorthand) > 1 {
		fmt.Fprintf(f.out(), "%s shorthand more than ASCII character: %s\n", f.name, flag.Shorthand)
		panic("shorthand is more than one character")
	}
	if f.shorthands == nil {
		f.shorthands = make(map[byte]*Flag)
	}
	c := flag.Shorthand[0]
	old, alreadythere := f.shorthands[c]
	if alreadythere {
		fmt.Fprintf(f.out(), "%s shorthand reused: %q for %s and %s\n", f.name, c, flag.Name, old.Name)
		panic("shorthand redefinition")
	}
	f.shorthands[c] = flag
}

// Var defines a flag with the specified name and usage string. The type and
// value of the flag are represented by the first argument, of type Value, which
// typically holds a user-defined implementation of Value. For instance, the
// caller could create a flag that turns a comma-separated string into a slice
// of strings by giving the slice the methods of Value; in particular, Set would
// decompose the comma-separated string into the slice.
func Var(value Value, name string, usage string) {
	commandLine.VarP(value, name, "", usage)
}

// Like Var, but accepts a shorthand letter that can be used after a single dash.
func VarP(value Value, name, shorthand, usage string) {
	commandLine.VarP(value, name, shorthand, usage)
}

// failf prints to standard error a formatted error and usage message and
// returns the error.
func (f *FlagSet) failf(format string, a ...interface{}) error {
	err := fmt.Errorf(format, a...)
	fmt.Fprintln(f.out(), err)
	f.usage()
	return err
}

// usage calls the Usage method for the flag set, or the usage function if
// the flag set is commandLine.
func (f *FlagSet) usage() {
	if f == commandLine {
		Usage()
	} else if f.Usage == nil {
		defaultUsage(f)
	} else {
		f.Usage()
	}
}

func (f *FlagSet) setFlag(flag *Flag, value string, origArg string) error {
	if err := flag.Value.Set(value); err != nil {
		return f.failf("invalid argument %q for %s: %v", value, origArg, err)
	}
	// mark as visited for Visit()
	if f.actual == nil {
		f.actual = make(map[string]*Flag)
	}
	f.actual[flag.Name] = flag
	flag.Changed = true
	return nil
}

func (f *FlagSet) parseLongArg(s string, args []string) (a []string, err error) {
	a = args
	if len(s) == 2 { // "--" terminates the flags
		f.args = append(f.args, args...)
		return
	}
	name := s[2:]
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		err = f.failf("bad flag syntax: %s", s)
		return
	}
	split := strings.SplitN(name, "=", 2)
	name = split[0]
	m := f.formal
	flag, alreadythere := m[name] // BUG
	if !alreadythere {
		if name == "help" { // special case for nice help message.
			f.usage()
			return args, ErrHelp
		}
		err = f.failf("unknown flag: --%s", name)
		return
	}
	if len(split) == 1 {
		if _, ok := flag.Value.(*boolValue); !ok {
			err = f.failf("flag needs an argument: %s", s)
			return
		}
		f.setFlag(flag, "true", s)
	} else {
		if e := f.setFlag(flag, split[1], s); e != nil {
			err = e
			return
		}
	}
	return args, nil
}

func (f *FlagSet) parseShortArg(s string, args []string) (a []string, err error) {
	a = args
	shorthands := s[1:]

	for i := 0; i < len(shorthands); i++ {
		c := shorthands[i]
		flag, alreadythere := f.shorthands[c]
		if !alreadythere {
			if c == 'h' { // special case for nice help message.
				f.usage()
				err = ErrHelp
				return
			}
			//TODO continue on error
			err = f.failf("unknown shorthand flag: %q in -%s", c, shorthands)
			if len(args) == 0 {
				return
			}
		}
		if alreadythere {
			if _, ok := flag.Value.(*boolValue); ok {
				f.setFlag(flag, "true", s)
				continue
			}
			if i < len(shorthands)-1 {
				if e := f.setFlag(flag, shorthands[i+1:], s); e != nil {
					err = e
					return
				}
				break
			}
			if len(args) == 0 {
				err = f.failf("flag needs an argument: %q in -%s", c, shorthands)
				return
			}
			if e := f.setFlag(flag, args[0], s); e != nil {
				err = e
				return
			}
		}
		a = args[1:]
		break // should be unnecessary
	}

	return
}

func (f *FlagSet) parseArgs(args []string) (err error) {
	for len(args) > 0 {
		s := args[0]
		args = args[1:]
		if len(s) == 0 || s[0] != '-' || len(s) == 1 {
			if !f.interspersed {
				f.args = append(f.args, s)
				f.args = append(f.args, args...)
				return nil
			}
			f.args = append(f.args, s)
			continue
		}

		if s[1] == '-' {
			args, err = f.parseLongArg(s, args)
		} else {
			args, err = f.parseShortArg(s, args)
		}
	}
	return
}

// Parse parses flag definitions from the argument list, which should not
// include the command name.  Must be called after all flags in the FlagSet
// are defined and before flags are accessed by the program.
// The return value will be ErrHelp if -help was set but not defined.
func (f *FlagSet) Parse(arguments []string) error {
	f.parsed = true
	f.args = make([]string, 0, len(arguments))
	err := f.parseArgs(arguments)
	if err != nil {
		switch f.errorHandling {
		case ContinueOnError:
			return err
		case ExitOnError:
			os.Exit(2)
		case PanicOnError:
			panic(err)
		}
	}
	return nil
}

// Parsed reports whether f.Parse has been called.
func (f *FlagSet) Parsed() bool {
	return f.parsed
}

// Parse parses the command-line flags from os.Args[1:].  Must be called
// after all flags are defined and before flags are accessed by the program.
func Parse() {
	// Ignore errors; commandLine is set for ExitOnError.
	commandLine.Parse(os.Args[1:])
}

// Whether to support interspersed option/non-option arguments.
func SetInterspersed(interspersed bool) {
	commandLine.SetInterspersed(interspersed)
}

// Parsed returns true if the command-line flags have been parsed.
func Parsed() bool {
	return commandLine.Parsed()
}

// The default set of command-line flags, parsed from os.Args.
var commandLine = NewFlagSet(os.Args[0], ExitOnError)

// NewFlagSet returns a new, empty flag set with the specified name and
// error handling property.
func NewFlagSet(name string, errorHandling ErrorHandling) *FlagSet {
	f := &FlagSet{
		name:          name,
		errorHandling: errorHandling,
		interspersed:  true,
	}
	return f
}

// Whether to support interspersed option/non-option arguments.
func (f *FlagSet) SetInterspersed(interspersed bool) {
	f.interspersed = interspersed
}

// Init sets the name and error handling property for a flag set.
// By default, the zero FlagSet uses an empty name and the
// ContinueOnError error handling policy.
func (f *FlagSet) Init(name string, errorHandling ErrorHandling) {
	f.name = name
	f.errorHandling = errorHandling
}
