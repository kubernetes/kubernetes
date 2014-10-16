package pflag

import (
	"fmt"
	"strconv"
)

// -- uint8 Value
type uint8Value uint8

func newUint8Value(val uint8, p *uint8) *uint8Value {
	*p = val
	return (*uint8Value)(p)
}

func (i *uint8Value) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 8)
	*i = uint8Value(v)
	return err
}

func (i *uint8Value) Type() string {
	return "uint8"
}

func (i *uint8Value) String() string { return fmt.Sprintf("%v", *i) }

// Uint8Var defines a uint8 flag with specified name, default value, and usage string.
// The argument p points to a uint8 variable in which to store the value of the flag.
func (f *FlagSet) Uint8Var(p *uint8, name string, value uint8, usage string) {
	f.VarP(newUint8Value(value, p), name, "", usage)
}

// Like Uint8Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint8VarP(p *uint8, name, shorthand string, value uint8, usage string) {
	f.VarP(newUint8Value(value, p), name, shorthand, usage)
}

// Uint8Var defines a uint8 flag with specified name, default value, and usage string.
// The argument p points to a uint8 variable in which to store the value of the flag.
func Uint8Var(p *uint8, name string, value uint8, usage string) {
	CommandLine.VarP(newUint8Value(value, p), name, "", usage)
}

// Like Uint8Var, but accepts a shorthand letter that can be used after a single dash.
func Uint8VarP(p *uint8, name, shorthand string, value uint8, usage string) {
	CommandLine.VarP(newUint8Value(value, p), name, shorthand, usage)
}

// Uint8 defines a uint8 flag with specified name, default value, and usage string.
// The return value is the address of a uint8 variable that stores the value of the flag.
func (f *FlagSet) Uint8(name string, value uint8, usage string) *uint8 {
	p := new(uint8)
	f.Uint8VarP(p, name, "", value, usage)
	return p
}

// Like Uint8, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint8P(name, shorthand string, value uint8, usage string) *uint8 {
	p := new(uint8)
	f.Uint8VarP(p, name, shorthand, value, usage)
	return p
}

// Uint8 defines a uint8 flag with specified name, default value, and usage string.
// The return value is the address of a uint8 variable that stores the value of the flag.
func Uint8(name string, value uint8, usage string) *uint8 {
	return CommandLine.Uint8P(name, "", value, usage)
}

// Like Uint8, but accepts a shorthand letter that can be used after a single dash.
func Uint8P(name, shorthand string, value uint8, usage string) *uint8 {
	return CommandLine.Uint8P(name, shorthand, value, usage)
}
