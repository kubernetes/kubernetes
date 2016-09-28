package pflag

import "strconv"

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

func (i *uintValue) Type() string {
	return "uint"
}

func (i *uintValue) String() string { return strconv.FormatUint(uint64(*i), 10) }

func uintConv(sval string) (interface{}, error) {
	v, err := strconv.ParseUint(sval, 0, 0)
	if err != nil {
		return 0, err
	}
	return uint(v), nil
}

// GetUint return the uint value of a flag with the given name
func (f *FlagSet) GetUint(name string) (uint, error) {
	val, err := f.getFlagType(name, "uint", uintConv)
	if err != nil {
		return 0, err
	}
	return val.(uint), nil
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint variable in which to store the value of the flag.
func (f *FlagSet) UintVar(p *uint, name string, value uint, usage string) {
	f.VarP(newUintValue(value, p), name, "", usage)
}

// UintVarP is like UintVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintVarP(p *uint, name, shorthand string, value uint, usage string) {
	f.VarP(newUintValue(value, p), name, shorthand, usage)
}

// UintVar defines a uint flag with specified name, default value, and usage string.
// The argument p points to a uint  variable in which to store the value of the flag.
func UintVar(p *uint, name string, value uint, usage string) {
	CommandLine.VarP(newUintValue(value, p), name, "", usage)
}

// UintVarP is like UintVar, but accepts a shorthand letter that can be used after a single dash.
func UintVarP(p *uint, name, shorthand string, value uint, usage string) {
	CommandLine.VarP(newUintValue(value, p), name, shorthand, usage)
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint  variable that stores the value of the flag.
func (f *FlagSet) Uint(name string, value uint, usage string) *uint {
	p := new(uint)
	f.UintVarP(p, name, "", value, usage)
	return p
}

// UintP is like Uint, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintP(name, shorthand string, value uint, usage string) *uint {
	p := new(uint)
	f.UintVarP(p, name, shorthand, value, usage)
	return p
}

// Uint defines a uint flag with specified name, default value, and usage string.
// The return value is the address of a uint  variable that stores the value of the flag.
func Uint(name string, value uint, usage string) *uint {
	return CommandLine.UintP(name, "", value, usage)
}

// UintP is like Uint, but accepts a shorthand letter that can be used after a single dash.
func UintP(name, shorthand string, value uint, usage string) *uint {
	return CommandLine.UintP(name, shorthand, value, usage)
}
