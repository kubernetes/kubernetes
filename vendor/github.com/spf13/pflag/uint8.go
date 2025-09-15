package pflag

import "strconv"

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

func (i *uint8Value) String() string { return strconv.FormatUint(uint64(*i), 10) }

func uint8Conv(sval string) (interface{}, error) {
	v, err := strconv.ParseUint(sval, 0, 8)
	if err != nil {
		return 0, err
	}
	return uint8(v), nil
}

// GetUint8 return the uint8 value of a flag with the given name
func (f *FlagSet) GetUint8(name string) (uint8, error) {
	val, err := f.getFlagType(name, "uint8", uint8Conv)
	if err != nil {
		return 0, err
	}
	return val.(uint8), nil
}

// Uint8Var defines a uint8 flag with specified name, default value, and usage string.
// The argument p points to a uint8 variable in which to store the value of the flag.
func (f *FlagSet) Uint8Var(p *uint8, name string, value uint8, usage string) {
	f.VarP(newUint8Value(value, p), name, "", usage)
}

// Uint8VarP is like Uint8Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Uint8VarP(p *uint8, name, shorthand string, value uint8, usage string) {
	f.VarP(newUint8Value(value, p), name, shorthand, usage)
}

// Uint8Var defines a uint8 flag with specified name, default value, and usage string.
// The argument p points to a uint8 variable in which to store the value of the flag.
func Uint8Var(p *uint8, name string, value uint8, usage string) {
	CommandLine.VarP(newUint8Value(value, p), name, "", usage)
}

// Uint8VarP is like Uint8Var, but accepts a shorthand letter that can be used after a single dash.
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

// Uint8P is like Uint8, but accepts a shorthand letter that can be used after a single dash.
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

// Uint8P is like Uint8, but accepts a shorthand letter that can be used after a single dash.
func Uint8P(name, shorthand string, value uint8, usage string) *uint8 {
	return CommandLine.Uint8P(name, shorthand, value, usage)
}
