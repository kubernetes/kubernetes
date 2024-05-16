package pflag

import "strconv"

// -- int16 Value
type int16Value int16

func newInt16Value(val int16, p *int16) *int16Value {
	*p = val
	return (*int16Value)(p)
}

func (i *int16Value) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 16)
	*i = int16Value(v)
	return err
}

func (i *int16Value) Type() string {
	return "int16"
}

func (i *int16Value) String() string { return strconv.FormatInt(int64(*i), 10) }

func int16Conv(sval string) (interface{}, error) {
	v, err := strconv.ParseInt(sval, 0, 16)
	if err != nil {
		return 0, err
	}
	return int16(v), nil
}

// GetInt16 returns the int16 value of a flag with the given name
func (f *FlagSet) GetInt16(name string) (int16, error) {
	val, err := f.getFlagType(name, "int16", int16Conv)
	if err != nil {
		return 0, err
	}
	return val.(int16), nil
}

// Int16Var defines an int16 flag with specified name, default value, and usage string.
// The argument p points to an int16 variable in which to store the value of the flag.
func (f *FlagSet) Int16Var(p *int16, name string, value int16, usage string) {
	f.VarP(newInt16Value(value, p), name, "", usage)
}

// Int16VarP is like Int16Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Int16VarP(p *int16, name, shorthand string, value int16, usage string) {
	f.VarP(newInt16Value(value, p), name, shorthand, usage)
}

// Int16Var defines an int16 flag with specified name, default value, and usage string.
// The argument p points to an int16 variable in which to store the value of the flag.
func Int16Var(p *int16, name string, value int16, usage string) {
	CommandLine.VarP(newInt16Value(value, p), name, "", usage)
}

// Int16VarP is like Int16Var, but accepts a shorthand letter that can be used after a single dash.
func Int16VarP(p *int16, name, shorthand string, value int16, usage string) {
	CommandLine.VarP(newInt16Value(value, p), name, shorthand, usage)
}

// Int16 defines an int16 flag with specified name, default value, and usage string.
// The return value is the address of an int16 variable that stores the value of the flag.
func (f *FlagSet) Int16(name string, value int16, usage string) *int16 {
	p := new(int16)
	f.Int16VarP(p, name, "", value, usage)
	return p
}

// Int16P is like Int16, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Int16P(name, shorthand string, value int16, usage string) *int16 {
	p := new(int16)
	f.Int16VarP(p, name, shorthand, value, usage)
	return p
}

// Int16 defines an int16 flag with specified name, default value, and usage string.
// The return value is the address of an int16 variable that stores the value of the flag.
func Int16(name string, value int16, usage string) *int16 {
	return CommandLine.Int16P(name, "", value, usage)
}

// Int16P is like Int16, but accepts a shorthand letter that can be used after a single dash.
func Int16P(name, shorthand string, value int16, usage string) *int16 {
	return CommandLine.Int16P(name, shorthand, value, usage)
}
