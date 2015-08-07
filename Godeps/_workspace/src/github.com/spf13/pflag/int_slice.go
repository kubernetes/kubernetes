package pflag

import (
	"fmt"
	"strconv"
	"strings"
)

// -- intSlice Value
type intSliceValue []int

func newIntSliceValue(val []int, p *[]int) *intSliceValue {
	*p = val
	return (*intSliceValue)(p)
}

func (s *intSliceValue) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make([]int, len(ss))
	for i, d := range ss {
		var err error
		out[i], err = strconv.Atoi(d)
		if err != nil {
			return err
		}

	}
	*s = append(*s, out...)
	return nil
}

func (s *intSliceValue) Type() string {
	return "intSlice"
}

func (s *intSliceValue) String() string {
	out := make([]string, len(*s))
	for i, d := range *s {
		out[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(out, ",") + "]"
}

func intSliceConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// Empty string would cause a slice with one (empty) entry
	if len(val) == 0 {
		return []int{}, nil
	}
	ss := strings.Split(val, ",")
	out := make([]int, len(ss))
	for i, d := range ss {
		var err error
		out[i], err = strconv.Atoi(d)
		if err != nil {
			return nil, err
		}

	}
	return out, nil
}

// GetIntSlice return the []int value of a flag with the given name
func (f *FlagSet) GetIntSlice(name string) ([]int, error) {
	val, err := f.getFlagType(name, "intSlice", intSliceConv)
	if err != nil {
		return []int{}, err
	}
	return val.([]int), nil
}

// IntSliceVar defines a intSlice flag with specified name, default value, and usage string.
// The argument p points to a []int variable in which to store the value of the flag.
func (f *FlagSet) IntSliceVar(p *[]int, name string, value []int, usage string) {
	f.VarP(newIntSliceValue(value, p), name, "", usage)
}

// Like IntSliceVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IntSliceVarP(p *[]int, name, shorthand string, value []int, usage string) {
	f.VarP(newIntSliceValue(value, p), name, shorthand, usage)
}

// IntSliceVar defines a int[] flag with specified name, default value, and usage string.
// The argument p points to a int[] variable in which to store the value of the flag.
func IntSliceVar(p *[]int, name string, value []int, usage string) {
	CommandLine.VarP(newIntSliceValue(value, p), name, "", usage)
}

// Like IntSliceVar, but accepts a shorthand letter that can be used after a single dash.
func IntSliceVarP(p *[]int, name, shorthand string, value []int, usage string) {
	CommandLine.VarP(newIntSliceValue(value, p), name, shorthand, usage)
}

// IntSlice defines a []int flag with specified name, default value, and usage string.
// The return value is the address of a []int variable that stores the value of the flag.
func (f *FlagSet) IntSlice(name string, value []int, usage string) *[]int {
	p := make([]int, 0)
	f.IntSliceVarP(&p, name, "", value, usage)
	return &p
}

// Like IntSlice, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) IntSliceP(name, shorthand string, value []int, usage string) *[]int {
	p := make([]int, 0)
	f.IntSliceVarP(&p, name, shorthand, value, usage)
	return &p
}

// IntSlice defines a []int flag with specified name, default value, and usage string.
// The return value is the address of a []int variable that stores the value of the flag.
func IntSlice(name string, value []int, usage string) *[]int {
	return CommandLine.IntSliceP(name, "", value, usage)
}

// Like IntSlice, but accepts a shorthand letter that can be used after a single dash.
func IntSliceP(name, shorthand string, value []int, usage string) *[]int {
	return CommandLine.IntSliceP(name, shorthand, value, usage)
}
