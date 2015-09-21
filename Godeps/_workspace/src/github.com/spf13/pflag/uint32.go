package pflag

import (
	"fmt"
	"strconv"
)

// -- uint16 value
type uint32Value uint32

func newUint32Value(val uint32, p *uint32) *uint32Value {
	*p = val
	return (*uint32Value)(p)
}
func (i *uint32Value) String() string { return fmt.Sprintf("%d", *i) }
func (i *uint32Value) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 32)
	*i = uint32Value(v)
	return err
}
func (i *uint32Value) Get() interface{} {
	return uint32(*i)
}

func (i *uint32Value) Type() string {
	return "uint32"
}

// Uint32Var defines a uint32 flag with specified name, default value, and usage string.
// The argument p points to a uint32 variable in which to store the value of the flag.
func (f *FlagSet) Uint32Var(p *uint32, name string, value uint32, usage string) {
	f.VarP(newUint32Value(value, p), name, "", usage)
}

// Like Uint32Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint32VarP(p *uint32, name, shorthand string, value uint32, usage string) {
	f.VarP(newUint32Value(value, p), name, shorthand, usage)
}

// Uint32Var defines a uint32 flag with specified name, default value, and usage string.
// The argument p points to a uint32  variable in which to store the value of the flag.
func Uint32Var(p *uint32, name string, value uint32, usage string) {
	CommandLine.VarP(newUint32Value(value, p), name, "", usage)
}

// Like Uint32Var, but accepts a shorthand letter that can be used after a single dash.
func Uint32VarP(p *uint32, name, shorthand string, value uint32, usage string) {
	CommandLine.VarP(newUint32Value(value, p), name, shorthand, usage)
}

// Uint32 defines a uint32 flag with specified name, default value, and usage string.
// The return value is the address of a uint32  variable that stores the value of the flag.
func (f *FlagSet) Uint32(name string, value uint32, usage string) *uint32 {
	p := new(uint32)
	f.Uint32VarP(p, name, "", value, usage)
	return p
}

// Like Uint32, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint32P(name, shorthand string, value uint32, usage string) *uint32 {
	p := new(uint32)
	f.Uint32VarP(p, name, shorthand, value, usage)
	return p
}

// Uint32 defines a uint32 flag with specified name, default value, and usage string.
// The return value is the address of a uint32  variable that stores the value of the flag.
func Uint32(name string, value uint32, usage string) *uint32 {
	return CommandLine.Uint32P(name, "", value, usage)
}

// Like Uint32, but accepts a shorthand letter that can be used after a single dash.
func Uint32P(name, shorthand string, value uint32, usage string) *uint32 {
	return CommandLine.Uint32P(name, shorthand, value, usage)
}
