package pflag

import (
	"fmt"
	"strconv"
	"strings"
)

// -- uintSlice Value
type uintSliceValue struct {
	value   *[]uint
	changed bool
}

func newUintSliceValue(val []uint, p *[]uint) *uintSliceValue {
	uisv := new(uintSliceValue)
	uisv.value = p
	*uisv.value = val
	return uisv
}

func (s *uintSliceValue) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make([]uint, len(ss))
	for i, d := range ss {
		u, err := strconv.ParseUint(d, 10, 0)
		if err != nil {
			return err
		}
		out[i] = uint(u)
	}
	if !s.changed {
		*s.value = out
	} else {
		*s.value = append(*s.value, out...)
	}
	s.changed = true
	return nil
}

func (s *uintSliceValue) Type() string {
	return "uintSlice"
}

func (s *uintSliceValue) String() string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(out, ",") + "]"
}

func (s *uintSliceValue) fromString(val string) (uint, error) {
	t, err := strconv.ParseUint(val, 10, 0)
	if err != nil {
		return 0, err
	}
	return uint(t), nil
}

func (s *uintSliceValue) toString(val uint) string {
	return fmt.Sprintf("%d", val)
}

func (s *uintSliceValue) Append(val string) error {
	i, err := s.fromString(val)
	if err != nil {
		return err
	}
	*s.value = append(*s.value, i)
	return nil
}

func (s *uintSliceValue) Replace(val []string) error {
	out := make([]uint, len(val))
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

func (s *uintSliceValue) GetSlice() []string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = s.toString(d)
	}
	return out
}

func uintSliceConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// Empty string would cause a slice with one (empty) entry
	if len(val) == 0 {
		return []uint{}, nil
	}
	ss := strings.Split(val, ",")
	out := make([]uint, len(ss))
	for i, d := range ss {
		u, err := strconv.ParseUint(d, 10, 0)
		if err != nil {
			return nil, err
		}
		out[i] = uint(u)
	}
	return out, nil
}

// GetUintSlice returns the []uint value of a flag with the given name.
func (f *FlagSet) GetUintSlice(name string) ([]uint, error) {
	val, err := f.getFlagType(name, "uintSlice", uintSliceConv)
	if err != nil {
		return []uint{}, err
	}
	return val.([]uint), nil
}

// UintSliceVar defines a uintSlice flag with specified name, default value, and usage string.
// The argument p points to a []uint variable in which to store the value of the flag.
func (f *FlagSet) UintSliceVar(p *[]uint, name string, value []uint, usage string) {
	f.VarP(newUintSliceValue(value, p), name, "", usage)
}

// UintSliceVarP is like UintSliceVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintSliceVarP(p *[]uint, name, shorthand string, value []uint, usage string) {
	f.VarP(newUintSliceValue(value, p), name, shorthand, usage)
}

// UintSliceVar defines a uint[] flag with specified name, default value, and usage string.
// The argument p points to a uint[] variable in which to store the value of the flag.
func UintSliceVar(p *[]uint, name string, value []uint, usage string) {
	CommandLine.VarP(newUintSliceValue(value, p), name, "", usage)
}

// UintSliceVarP is like the UintSliceVar, but accepts a shorthand letter that can be used after a single dash.
func UintSliceVarP(p *[]uint, name, shorthand string, value []uint, usage string) {
	CommandLine.VarP(newUintSliceValue(value, p), name, shorthand, usage)
}

// UintSlice defines a []uint flag with specified name, default value, and usage string.
// The return value is the address of a []uint variable that stores the value of the flag.
func (f *FlagSet) UintSlice(name string, value []uint, usage string) *[]uint {
	p := []uint{}
	f.UintSliceVarP(&p, name, "", value, usage)
	return &p
}

// UintSliceP is like UintSlice, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) UintSliceP(name, shorthand string, value []uint, usage string) *[]uint {
	p := []uint{}
	f.UintSliceVarP(&p, name, shorthand, value, usage)
	return &p
}

// UintSlice defines a []uint flag with specified name, default value, and usage string.
// The return value is the address of a []uint variable that stores the value of the flag.
func UintSlice(name string, value []uint, usage string) *[]uint {
	return CommandLine.UintSliceP(name, "", value, usage)
}

// UintSliceP is like UintSlice, but accepts a shorthand letter that can be used after a single dash.
func UintSliceP(name, shorthand string, value []uint, usage string) *[]uint {
	return CommandLine.UintSliceP(name, shorthand, value, usage)
}
