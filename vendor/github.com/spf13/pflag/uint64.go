package pflag

import (
	"fmt"
	"strconv"
)

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

func (i *uint64Value) Type() string {
	return "uint64"
}

func (i *uint64Value) String() string { return fmt.Sprintf("%v", *i) }

func uint64Conv(sval string) (interface{}, error) {
	v, err := strconv.ParseUint(sval, 0, 64)
	if err != nil {
		return 0, err
	}
	return uint64(v), nil
}

// GetUint64 return the uint64 value of a flag with the given name
func (f *FlagSet) GetUint64(name string) (uint64, error) {
	val, err := f.getFlagType(name, "uint64", uint64Conv)
	if err != nil {
		return 0, err
	}
	return val.(uint64), nil
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func (f *FlagSet) Uint64Var(p *uint64, name string, value uint64, usage string) {
	f.VarP(newUint64Value(value, p), name, "", usage)
}

// Uint64VarP is like Uint64Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint64VarP(p *uint64, name, shorthand string, value uint64, usage string) {
	f.VarP(newUint64Value(value, p), name, shorthand, usage)
}

// Uint64Var defines a uint64 flag with specified name, default value, and usage string.
// The argument p points to a uint64 variable in which to store the value of the flag.
func Uint64Var(p *uint64, name string, value uint64, usage string) {
	CommandLine.VarP(newUint64Value(value, p), name, "", usage)
}

// Uint64VarP is like Uint64Var, but accepts a shorthand letter that can be used after a single dash.
func Uint64VarP(p *uint64, name, shorthand string, value uint64, usage string) {
	CommandLine.VarP(newUint64Value(value, p), name, shorthand, usage)
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func (f *FlagSet) Uint64(name string, value uint64, usage string) *uint64 {
	p := new(uint64)
	f.Uint64VarP(p, name, "", value, usage)
	return p
}

// Uint64P is like Uint64, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint64P(name, shorthand string, value uint64, usage string) *uint64 {
	p := new(uint64)
	f.Uint64VarP(p, name, shorthand, value, usage)
	return p
}

// Uint64 defines a uint64 flag with specified name, default value, and usage string.
// The return value is the address of a uint64 variable that stores the value of the flag.
func Uint64(name string, value uint64, usage string) *uint64 {
	return CommandLine.Uint64P(name, "", value, usage)
}

// Uint64P is like Uint64, but accepts a shorthand letter that can be used after a single dash.
func Uint64P(name, shorthand string, value uint64, usage string) *uint64 {
	return CommandLine.Uint64P(name, shorthand, value, usage)
}
