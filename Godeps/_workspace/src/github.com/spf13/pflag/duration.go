package pflag

import (
	"time"
)

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

func (d *durationValue) Type() string {
	return "duration"
}

func (d *durationValue) String() string { return (*time.Duration)(d).String() }

func durationConv(sval string) (interface{}, error) {
	return time.ParseDuration(sval)
}

// GetDuration return the duration value of a flag with the given name
func (f *FlagSet) GetDuration(name string) (time.Duration, error) {
	val, err := f.getFlagType(name, "duration", durationConv)
	if err != nil {
		return 0, err
	}
	return val.(time.Duration), nil
}

// DurationVar defines a time.Duration flag with specified name, default value, and usage string.
// The argument p points to a time.Duration variable in which to store the value of the flag.
func (f *FlagSet) DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	f.VarP(newDurationValue(value, p), name, "", usage)
}

// DurationVarP is like DurationVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationVarP(p *time.Duration, name, shorthand string, value time.Duration, usage string) {
	f.VarP(newDurationValue(value, p), name, shorthand, usage)
}

// DurationVar defines a time.Duration flag with specified name, default value, and usage string.
// The argument p points to a time.Duration variable in which to store the value of the flag.
func DurationVar(p *time.Duration, name string, value time.Duration, usage string) {
	CommandLine.VarP(newDurationValue(value, p), name, "", usage)
}

// DurationVarP is like DurationVar, but accepts a shorthand letter that can be used after a single dash.
func DurationVarP(p *time.Duration, name, shorthand string, value time.Duration, usage string) {
	CommandLine.VarP(newDurationValue(value, p), name, shorthand, usage)
}

// Duration defines a time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a time.Duration variable that stores the value of the flag.
func (f *FlagSet) Duration(name string, value time.Duration, usage string) *time.Duration {
	p := new(time.Duration)
	f.DurationVarP(p, name, "", value, usage)
	return p
}

// DurationP is like Duration, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationP(name, shorthand string, value time.Duration, usage string) *time.Duration {
	p := new(time.Duration)
	f.DurationVarP(p, name, shorthand, value, usage)
	return p
}

// Duration defines a time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a time.Duration variable that stores the value of the flag.
func Duration(name string, value time.Duration, usage string) *time.Duration {
	return CommandLine.DurationP(name, "", value, usage)
}

// DurationP is like Duration, but accepts a shorthand letter that can be used after a single dash.
func DurationP(name, shorthand string, value time.Duration, usage string) *time.Duration {
	return CommandLine.DurationP(name, shorthand, value, usage)
}
