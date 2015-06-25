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
	"strings"
)

// ErrHelp is the error returned if the flag -help is invoked but no such flag is defined.
var ErrHelp = errors.New("pflag: help requested")

// ErrorHandling defines how to handle flag parsing errors.
type ErrorHandling int

const (
	ContinueOnError ErrorHandling = iota
	ExitOnError
	PanicOnError
)

// NormalizedName is a flag name that has been normalized according to rules
// for the FlagSet (e.g. making '-' and '_' equivalent).
type NormalizedName string

// A FlagSet represents a set of defined flags.
type FlagSet struct {
	// Usage is the function called when an error occurs while parsing flags.
	// The field is a function (not a method) that may be changed to point to
	// a custom error handler.
	Usage func()

	name              string
	parsed            bool
	actual            map[NormalizedName]*Flag
	formal            map[NormalizedName]*Flag
	shorthands        map[byte]*Flag
	args              []string // arguments after flags
	exitOnError       bool     // does the program exit if there's an error?
	errorHandling     ErrorHandling
	output            io.Writer // nil means stderr; use out() accessor
	interspersed      bool      // allow interspersed option/non-option args
	normalizeNameFunc func(f *FlagSet, name string) NormalizedName
}

// A Flag represents the state of a flag.
type Flag struct {
	Name        string              // name as it appears on command line
	Shorthand   string              // one-letter abbreviated flag
	Usage       string              // help message
	Value       Value               // value as set
	DefValue    string              // default value (as text); for usage message
	Changed     bool                // If the user set the value (or if left to default)
	Deprecated  string              // If this flag is deprecated, this string is the new or now thing to use
	Annotations map[string][]string // used by cobra.Command  bash autocomple code
}

// Value is the interface to the dynamic value stored in a flag.
// (The default value is represented as a string.)
type Value interface {
	String() string
	Set(string) error
	Type() string
}

// sortFlags returns the flags as a slice in lexicographical sorted order.
func sortFlags(flags map[NormalizedName]*Flag) []*Flag {
	list := make(sort.StringSlice, len(flags))
	i := 0
	for k := range flags {
		list[i] = string(k)
		i++
	}
	list.Sort()
	result := make([]*Flag, len(list))
	for i, name := range list {
		result[i] = flags[NormalizedName(name)]
	}
	return result
}

func (f *FlagSet) SetNormalizeFunc(n func(f *FlagSet, name string) NormalizedName) {
	f.normalizeNameFunc = n
	for k, v := range f.formal {
		delete(f.formal, k)
		nname := f.normalizeFlagName(string(k))
		f.formal[nname] = v
		v.Name = string(nname)
	}
}

func (f *FlagSet) GetNormalizeFunc() func(f *FlagSet, name string) NormalizedName {
	if f.normalizeNameFunc != nil {
		return f.normalizeNameFunc
	}
	return func(f *FlagSet, name string) NormalizedName { return NormalizedName(name) }
}

func (f *FlagSet) normalizeFlagName(name string) NormalizedName {
	n := f.GetNormalizeFunc()
	return n(f, name)
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
	CommandLine.VisitAll(fn)
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
	CommandLine.Visit(fn)
}

// Lookup returns the Flag structure of the named flag, returning nil if none exists.
func (f *FlagSet) Lookup(name string) *Flag {
	return f.lookup(f.normalizeFlagName(name))
}

// lookup returns the Flag structure of the named flag, returning nil if none exists.
func (f *FlagSet) lookup(name NormalizedName) *Flag {
	return f.formal[name]
}

// Mark a flag deprecated in your program
func (f *FlagSet) MarkDeprecated(name string, usageMessage string) error {
	flag := f.Lookup(name)
	if flag == nil {
		return fmt.Errorf("flag %q does not exist", name)
	}
	flag.Deprecated = usageMessage
	return nil
}

// Lookup returns the Flag structure of the named command-line flag,
// returning nil if none exists.
func Lookup(name string) *Flag {
	return CommandLine.Lookup(name)
}

// Set sets the value of the named flag.
func (f *FlagSet) Set(name, value string) error {
	normalName := f.normalizeFlagName(name)
	flag, ok := f.formal[normalName]
	if !ok {
		return fmt.Errorf("no such flag -%v", name)
	}
	err := flag.Value.Set(value)
	if err != nil {
		return err
	}
	if f.actual == nil {
		f.actual = make(map[NormalizedName]*Flag)
	}
	f.actual[normalName] = flag
	flag.Changed = true
	if len(flag.Deprecated) > 0 {
		fmt.Fprintf(os.Stderr, "Flag --%s has been deprecated, %s\n", flag.Name, flag.Deprecated)
	}
	return nil
}

// Set sets the value of the named command-line flag.
func Set(name, value string) error {
	return CommandLine.Set(name, value)
}

// PrintDefaults prints, to standard error unless configured
// otherwise, the default values of all defined flags in the set.
func (f *FlagSet) PrintDefaults() {
	f.VisitAll(func(flag *Flag) {
		if len(flag.Deprecated) > 0 {
			return
		}
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
		if len(flag.Deprecated) > 0 {
			return
		}
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
	CommandLine.PrintDefaults()
}

// defaultUsage is the default function to print a usage message.
func defaultUsage(f *FlagSet) {
	fmt.Fprintf(f.out(), "Usage of %s:\n", f.name)
	f.PrintDefaults()
}

// NOTE: Usage is not just defaultUsage(CommandLine)
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
func NFlag() int { return len(CommandLine.actual) }

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
	return CommandLine.Arg(i)
}

// NArg is the number of arguments remaining after flags have been processed.
func (f *FlagSet) NArg() int { return len(f.args) }

// NArg is the number of arguments remaining after flags have been processed.
func NArg() int { return len(CommandLine.args) }

// Args returns the non-flag arguments.
func (f *FlagSet) Args() []string { return f.args }

// Args returns the non-flag command-line arguments.
func Args() []string { return CommandLine.args }

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
	flag := &Flag{
		Name:      name,
		Shorthand: shorthand,
		Usage:     usage,
		Value:     value,
		DefValue:  value.String(),
	}
	f.AddFlag(flag)
}

func (f *FlagSet) AddFlag(flag *Flag) {
	// Call normalizeFlagName function only once
	var normalizedFlagName NormalizedName = f.normalizeFlagName(flag.Name)

	_, alreadythere := f.formal[normalizedFlagName]
	if alreadythere {
		msg := fmt.Sprintf("%s flag redefined: %s", f.name, flag.Name)
		fmt.Fprintln(f.out(), msg)
		panic(msg) // Happens only if flags are declared with identical names
	}
	if f.formal == nil {
		f.formal = make(map[NormalizedName]*Flag)
	}

	flag.Name = string(normalizedFlagName)
	f.formal[normalizedFlagName] = flag

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
		fmt.Fprintf(f.out(), "%s shorthand reused: %q for %s already used for %s\n", f.name, c, flag.Name, old.Name)
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
	CommandLine.VarP(value, name, "", usage)
}

// Like Var, but accepts a shorthand letter that can be used after a single dash.
func VarP(value Value, name, shorthand, usage string) {
	CommandLine.VarP(value, name, shorthand, usage)
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
// the flag set is CommandLine.
func (f *FlagSet) usage() {
	if f == CommandLine {
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
		f.actual = make(map[NormalizedName]*Flag)
	}
	f.actual[f.normalizeFlagName(flag.Name)] = flag
	flag.Changed = true
	if len(flag.Deprecated) > 0 {
		fmt.Fprintf(os.Stderr, "Flag --%s has been deprecated, %s\n", flag.Name, flag.Deprecated)
	}
	return nil
}

func (f *FlagSet) parseLongArg(s string, args []string) (a []string, err error) {
	a = args
	name := s[2:]
	if len(name) == 0 || name[0] == '-' || name[0] == '=' {
		err = f.failf("bad flag syntax: %s", s)
		return
	}
	split := strings.SplitN(name, "=", 2)
	name = split[0]
	flag, alreadythere := f.formal[f.normalizeFlagName(name)]
	if !alreadythere {
		if name == "help" { // special case for nice help message.
			f.usage()
			return a, ErrHelp
		}
		err = f.failf("unknown flag: --%s", name)
		return
	}
	var value string
	if len(split) == 1 {
		if bv, ok := flag.Value.(boolFlag); !ok || !bv.IsBoolFlag() {
			err = f.failf("flag needs an argument: %s", s)
			return
		}
		value = "true"
	} else {
		value = split[1]
	}
	err = f.setFlag(flag, value, s)
	return
}

func (f *FlagSet) parseSingleShortArg(shorthands string, args []string) (outShorts string, outArgs []string, err error) {
	outArgs = args
	outShorts = shorthands[1:]
	c := shorthands[0]

	flag, alreadythere := f.shorthands[c]
	if !alreadythere {
		if c == 'h' { // special case for nice help message.
			f.usage()
			err = ErrHelp
			return
		}
		//TODO continue on error
		err = f.failf("unknown shorthand flag: %q in -%s", c, shorthands)
		return
	}
	var value string
	if len(shorthands) > 2 && shorthands[1] == '=' {
		value = shorthands[2:]
		outShorts = ""
	} else if bv, ok := flag.Value.(boolFlag); ok && bv.IsBoolFlag() {
		value = "true"
	} else if len(shorthands) > 1 {
		value = shorthands[1:]
		outShorts = ""
	} else if len(args) > 0 {
		value = args[0]
		outArgs = args[1:]
	} else {
		err = f.failf("flag needs an argument: %q in -%s", c, shorthands)
		return
	}
	err = f.setFlag(flag, value, shorthands)
	return
}

func (f *FlagSet) parseShortArg(s string, args []string) (a []string, err error) {
	a = args
	shorthands := s[1:]

	for len(shorthands) > 0 {
		shorthands, a, err = f.parseSingleShortArg(shorthands, args)
		if err != nil {
			return
		}
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
			if len(s) == 2 { // "--" terminates the flags
				f.args = append(f.args, args...)
				break
			}
			args, err = f.parseLongArg(s, args)
		} else {
			args, err = f.parseShortArg(s, args)
		}
		if err != nil {
			return
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
	// Ignore errors; CommandLine is set for ExitOnError.
	CommandLine.Parse(os.Args[1:])
}

// Whether to support interspersed option/non-option arguments.
func SetInterspersed(interspersed bool) {
	CommandLine.SetInterspersed(interspersed)
}

// Parsed returns true if the command-line flags have been parsed.
func Parsed() bool {
	return CommandLine.Parsed()
}

// The default set of command-line flags, parsed from os.Args.
var CommandLine = NewFlagSet(os.Args[0], ExitOnError)

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
