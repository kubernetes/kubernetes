package pflag

import (
	"fmt"
	"strconv"
	"strings"
)

// -- float32Slice Value
type float32SliceValue struct {
	value   *[]float32
	changed bool
}

func newFloat32SliceValue(val []float32, p *[]float32) *float32SliceValue {
	isv := new(float32SliceValue)
	isv.value = p
	*isv.value = val
	return isv
}

func (s *float32SliceValue) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make([]float32, len(ss))
	for i, d := range ss {
		var err error
		var temp64 float64
		temp64, err = strconv.ParseFloat(d, 32)
		if err != nil {
			return err
		}
		out[i] = float32(temp64)

	}
	if !s.changed {
		*s.value = out
	} else {
		*s.value = append(*s.value, out...)
	}
	s.changed = true
	return nil
}

func (s *float32SliceValue) Type() string {
	return "float32Slice"
}

func (s *float32SliceValue) String() string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = fmt.Sprintf("%f", d)
	}
	return "[" + strings.Join(out, ",") + "]"
}

func (s *float32SliceValue) fromString(val string) (float32, error) {
	t64, err := strconv.ParseFloat(val, 32)
	if err != nil {
		return 0, err
	}
	return float32(t64), nil
}

func (s *float32SliceValue) toString(val float32) string {
	return fmt.Sprintf("%f", val)
}

func (s *float32SliceValue) Append(val string) error {
	i, err := s.fromString(val)
	if err != nil {
		return err
	}
	*s.value = append(*s.value, i)
	return nil
}

func (s *float32SliceValue) Replace(val []string) error {
	out := make([]float32, len(val))
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

func (s *float32SliceValue) GetSlice() []string {
	out := make([]string, len(*s.value))
	for i, d := range *s.value {
		out[i] = s.toString(d)
	}
	return out
}

func float32SliceConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// Empty string would cause a slice with one (empty) entry
	if len(val) == 0 {
		return []float32{}, nil
	}
	ss := strings.Split(val, ",")
	out := make([]float32, len(ss))
	for i, d := range ss {
		var err error
		var temp64 float64
		temp64, err = strconv.ParseFloat(d, 32)
		if err != nil {
			return nil, err
		}
		out[i] = float32(temp64)

	}
	return out, nil
}

// GetFloat32Slice return the []float32 value of a flag with the given name
func (f *FlagSet) GetFloat32Slice(name string) ([]float32, error) {
	val, err := f.getFlagType(name, "float32Slice", float32SliceConv)
	if err != nil {
		return []float32{}, err
	}
	return val.([]float32), nil
}

// Float32SliceVar defines a float32Slice flag with specified name, default value, and usage string.
// The argument p points to a []float32 variable in which to store the value of the flag.
func (f *FlagSet) Float32SliceVar(p *[]float32, name string, value []float32, usage string) {
	f.VarP(newFloat32SliceValue(value, p), name, "", usage)
}

// Float32SliceVarP is like Float32SliceVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Float32SliceVarP(p *[]float32, name, shorthand string, value []float32, usage string) {
	f.VarP(newFloat32SliceValue(value, p), name, shorthand, usage)
}

// Float32SliceVar defines a float32[] flag with specified name, default value, and usage string.
// The argument p points to a float32[] variable in which to store the value of the flag.
func Float32SliceVar(p *[]float32, name string, value []float32, usage string) {
	CommandLine.VarP(newFloat32SliceValue(value, p), name, "", usage)
}

// Float32SliceVarP is like Float32SliceVar, but accepts a shorthand letter that can be used after a single dash.
func Float32SliceVarP(p *[]float32, name, shorthand string, value []float32, usage string) {
	CommandLine.VarP(newFloat32SliceValue(value, p), name, shorthand, usage)
}

// Float32Slice defines a []float32 flag with specified name, default value, and usage string.
// The return value is the address of a []float32 variable that stores the value of the flag.
func (f *FlagSet) Float32Slice(name string, value []float32, usage string) *[]float32 {
	p := []float32{}
	f.Float32SliceVarP(&p, name, "", value, usage)
	return &p
}

// Float32SliceP is like Float32Slice, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) Float32SliceP(name, shorthand string, value []float32, usage string) *[]float32 {
	p := []float32{}
	f.Float32SliceVarP(&p, name, shorthand, value, usage)
	return &p
}

// Float32Slice defines a []float32 flag with specified name, default value, and usage string.
// The return value is the address of a []float32 variable that stores the value of the flag.
func Float32Slice(name string, value []float32, usage string) *[]float32 {
	return CommandLine.Float32SliceP(name, "", value, usage)
}

// Float32SliceP is like Float32Slice, but accepts a shorthand letter that can be used after a single dash.
func Float32SliceP(name, shorthand string, value []float32, usage string) *[]float32 {
	return CommandLine.Float32SliceP(name, shorthand, value, usage)
}
