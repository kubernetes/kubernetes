package pflag

import (
	"fmt"
	"strings"
	"time"
)

// -- durationSlice Value
type durationSliceValue struct {
	value   *[]time.Duration
	changed bool
}

func newDurationSliceValue(val []time.Duration, p *[]time.Duration) *durationSliceValue {
	dsv := new(durationSliceValue)
	dsv.value = p
	*dsv.value = val
	return dsv
}

func (s *durationSliceValue) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make([]time.Duration, len(ss))
	for i, d := range ss {
		var err error
		out[i], err = time.ParseDuration(d)
		if err != nil {
			return err
		}

	}
	if !s.changed {
		*s.value = out
	} else {
		*s.value = append(*s.value, out...)
	}
	s.changed = true
	return nil
}

func (s *durationSliceValue) Type() string {
	return "durationSlice"
}

func (s *durationSliceValue) String() string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = fmt.Sprintf("%s", d)
	}
	return "[" + strings.Join(out, ",") + "]"
}

func (s *durationSliceValue) fromString(val string) (time.Duration, error) {
	return time.ParseDuration(val)
}

func (s *durationSliceValue) toString(val time.Duration) string {
	return fmt.Sprintf("%s", val)
}

func (s *durationSliceValue) Append(val string) error {
	i, err := s.fromString(val)
	if err != nil {
		return err
	}
	*s.value = append(*s.value, i)
	return nil
}

func (s *durationSliceValue) Replace(val []string) error {
	out := make([]time.Duration, len(val))
	for i, d := range val {
		var err error
		out[i], err = s.fromString(d)
		if err != nil {
			return err
		}
	}
	*s.value = out
	return nil
}

func (s *durationSliceValue) GetSlice() []string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = s.toString(d)
	}
	return out
}

func durationSliceConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// Empty string would cause a slice with one (empty) entry
	if len(val) == 0 {
		return []time.Duration{}, nil
	}
	ss := strings.Split(val, ",")
	out := make([]time.Duration, len(ss))
	for i, d := range ss {
		var err error
		out[i], err = time.ParseDuration(d)
		if err != nil {
			return nil, err
		}

	}
	return out, nil
}

// GetDurationSlice returns the []time.Duration value of a flag with the given name
func (f *FlagSet) GetDurationSlice(name string) ([]time.Duration, error) {
	val, err := f.getFlagType(name, "durationSlice", durationSliceConv)
	if err != nil {
		return []time.Duration{}, err
	}
	return val.([]time.Duration), nil
}

// DurationSliceVar defines a durationSlice flag with specified name, default value, and usage string.
// The argument p points to a []time.Duration variable in which to store the value of the flag.
func (f *FlagSet) DurationSliceVar(p *[]time.Duration, name string, value []time.Duration, usage string) {
	f.VarP(newDurationSliceValue(value, p), name, "", usage)
}

// DurationSliceVarP is like DurationSliceVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationSliceVarP(p *[]time.Duration, name, shorthand string, value []time.Duration, usage string) {
	f.VarP(newDurationSliceValue(value, p), name, shorthand, usage)
}

// DurationSliceVar defines a duration[] flag with specified name, default value, and usage string.
// The argument p points to a duration[] variable in which to store the value of the flag.
func DurationSliceVar(p *[]time.Duration, name string, value []time.Duration, usage string) {
	CommandLine.VarP(newDurationSliceValue(value, p), name, "", usage)
}

// DurationSliceVarP is like DurationSliceVar, but accepts a shorthand letter that can be used after a single dash.
func DurationSliceVarP(p *[]time.Duration, name, shorthand string, value []time.Duration, usage string) {
	CommandLine.VarP(newDurationSliceValue(value, p), name, shorthand, usage)
}

// DurationSlice defines a []time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a []time.Duration variable that stores the value of the flag.
func (f *FlagSet) DurationSlice(name string, value []time.Duration, usage string) *[]time.Duration {
	p := []time.Duration{}
	f.DurationSliceVarP(&p, name, "", value, usage)
	return &p
}

// DurationSliceP is like DurationSlice, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) DurationSliceP(name, shorthand string, value []time.Duration, usage string) *[]time.Duration {
	p := []time.Duration{}
	f.DurationSliceVarP(&p, name, shorthand, value, usage)
	return &p
}

// DurationSlice defines a []time.Duration flag with specified name, default value, and usage string.
// The return value is the address of a []time.Duration variable that stores the value of the flag.
func DurationSlice(name string, value []time.Duration, usage string) *[]time.Duration {
	return CommandLine.DurationSliceP(name, "", value, usage)
}

// DurationSliceP is like DurationSlice, but accepts a shorthand letter that can be used after a single dash.
func DurationSliceP(name, shorthand string, value []time.Duration, usage string) *[]time.Duration {
	return CommandLine.DurationSliceP(name, shorthand, value, usage)
}
