package pflag

import (
	"fmt"
	"strings"
	"time"
)

// TimeValue adapts time.Time for use as a flag.
type timeValue struct {
	*time.Time
	formats []string
}

func newTimeValue(val time.Time, p *time.Time, formats []string) *timeValue {
	*p = val
	return &timeValue{
		Time:    p,
		formats: formats,
	}
}

// Set time.Time value from string based on accepted formats.
func (d *timeValue) Set(s string) error {
	s = strings.TrimSpace(s)
	for _, f := range d.formats {
		v, err := time.Parse(f, s)
		if err != nil {
			continue
		}
		*d.Time = v
		return nil
	}

	formatsString := ""
	for i, f := range d.formats {
		if i > 0 {
			formatsString += ", "
		}
		formatsString += fmt.Sprintf("`%s`", f)
	}

	return fmt.Errorf("invalid time format `%s` must be one of: %s", s, formatsString)
}

// Type name for time.Time flags.
func (d *timeValue) Type() string {
	return "time"
}

func (d *timeValue) String() string {
	if d.Time.IsZero() {
		return ""
	} else {
		return d.Time.Format(time.RFC3339Nano)
	}
}

// GetTime return the time value of a flag with the given name
func (f *FlagSet) GetTime(name string) (time.Time, error) {
	flag := f.Lookup(name)
	if flag == nil {
		err := fmt.Errorf("flag accessed but not defined: %s", name)
		return time.Time{}, err
	}

	if flag.Value.Type() != "time" {
		err := fmt.Errorf("trying to get %s value of flag of type %s", "time", flag.Value.Type())
		return time.Time{}, err
	}

	val, ok := flag.Value.(*timeValue)
	if !ok {
		return time.Time{}, fmt.Errorf("value %s is not a time", flag.Value)
	}

	return *val.Time, nil
}

// TimeVar defines a time.Time flag with specified name, default value, and usage string.
// The argument p points to a time.Time variable in which to store the value of the flag.
func (f *FlagSet) TimeVar(p *time.Time, name string, value time.Time, formats []string, usage string) {
	f.TimeVarP(p, name, "", value, formats, usage)
}

// TimeVarP is like TimeVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) TimeVarP(p *time.Time, name, shorthand string, value time.Time, formats []string, usage string) {
	f.VarP(newTimeValue(value, p, formats), name, shorthand, usage)
}

// TimeVar defines a time.Time flag with specified name, default value, and usage string.
// The argument p points to a time.Time variable in which to store the value of the flag.
func TimeVar(p *time.Time, name string, value time.Time, formats []string, usage string) {
	CommandLine.TimeVarP(p, name, "", value, formats, usage)
}

// TimeVarP is like TimeVar, but accepts a shorthand letter that can be used after a single dash.
func TimeVarP(p *time.Time, name, shorthand string, value time.Time, formats []string, usage string) {
	CommandLine.VarP(newTimeValue(value, p, formats), name, shorthand, usage)
}

// Time defines a time.Time flag with specified name, default value, and usage string.
// The return value is the address of a time.Time variable that stores the value of the flag.
func (f *FlagSet) Time(name string, value time.Time, formats []string, usage string) *time.Time {
	return f.TimeP(name, "", value, formats, usage)
}

// TimeP is like Time, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) TimeP(name, shorthand string, value time.Time, formats []string, usage string) *time.Time {
	p := new(time.Time)
	f.TimeVarP(p, name, shorthand, value, formats, usage)
	return p
}

// Time defines a time.Time flag with specified name, default value, and usage string.
// The return value is the address of a time.Time variable that stores the value of the flag.
func Time(name string, value time.Time, formats []string, usage string) *time.Time {
	return CommandLine.TimeP(name, "", value, formats, usage)
}

// TimeP is like Time, but accepts a shorthand letter that can be used after a single dash.
func TimeP(name, shorthand string, value time.Time, formats []string, usage string) *time.Time {
	return CommandLine.TimeP(name, shorthand, value, formats, usage)
}
