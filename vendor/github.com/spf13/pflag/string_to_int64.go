package pflag

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// -- stringToInt64 Value
type stringToInt64Value struct {
	value   *map[string]int64
	changed bool
}

func newStringToInt64Value(val map[string]int64, p *map[string]int64) *stringToInt64Value {
	ssv := new(stringToInt64Value)
	ssv.value = p
	*ssv.value = val
	return ssv
}

// Format: a=1,b=2
func (s *stringToInt64Value) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make(map[string]int64, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return fmt.Errorf("%s must be formatted as key=value", pair)
		}
		var err error
		out[kv[0]], err = strconv.ParseInt(kv[1], 10, 64)
		if err != nil {
			return err
		}
	}
	if !s.changed {
		*s.value = out
	} else {
		for k, v := range out {
			(*s.value)[k] = v
		}
	}
	s.changed = true
	return nil
}

func (s *stringToInt64Value) Type() string {
	return "stringToInt64"
}

func (s *stringToInt64Value) String() string {
	var buf bytes.Buffer
	i := 0
	for k, v := range *s.value {
		if i > 0 {
			buf.WriteRune(',')
		}
		buf.WriteString(k)
		buf.WriteRune('=')
		buf.WriteString(strconv.FormatInt(v, 10))
		i++
	}
	return "[" + buf.String() + "]"
}

func stringToInt64Conv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// An empty string would cause an empty map
	if len(val) == 0 {
		return map[string]int64{}, nil
	}
	ss := strings.Split(val, ",")
	out := make(map[string]int64, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("%s must be formatted as key=value", pair)
		}
		var err error
		out[kv[0]], err = strconv.ParseInt(kv[1], 10, 64)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// GetStringToInt64 return the map[string]int64 value of a flag with the given name
func (f *FlagSet) GetStringToInt64(name string) (map[string]int64, error) {
	val, err := f.getFlagType(name, "stringToInt64", stringToInt64Conv)
	if err != nil {
		return map[string]int64{}, err
	}
	return val.(map[string]int64), nil
}

// StringToInt64Var defines a string flag with specified name, default value, and usage string.
// The argument p point64s to a map[string]int64 variable in which to store the values of the multiple flags.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToInt64Var(p *map[string]int64, name string, value map[string]int64, usage string) {
	f.VarP(newStringToInt64Value(value, p), name, "", usage)
}

// StringToInt64VarP is like StringToInt64Var, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToInt64VarP(p *map[string]int64, name, shorthand string, value map[string]int64, usage string) {
	f.VarP(newStringToInt64Value(value, p), name, shorthand, usage)
}

// StringToInt64Var defines a string flag with specified name, default value, and usage string.
// The argument p point64s to a map[string]int64 variable in which to store the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToInt64Var(p *map[string]int64, name string, value map[string]int64, usage string) {
	CommandLine.VarP(newStringToInt64Value(value, p), name, "", usage)
}

// StringToInt64VarP is like StringToInt64Var, but accepts a shorthand letter that can be used after a single dash.
func StringToInt64VarP(p *map[string]int64, name, shorthand string, value map[string]int64, usage string) {
	CommandLine.VarP(newStringToInt64Value(value, p), name, shorthand, usage)
}

// StringToInt64 defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]int64 variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToInt64(name string, value map[string]int64, usage string) *map[string]int64 {
	p := map[string]int64{}
	f.StringToInt64VarP(&p, name, "", value, usage)
	return &p
}

// StringToInt64P is like StringToInt64, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToInt64P(name, shorthand string, value map[string]int64, usage string) *map[string]int64 {
	p := map[string]int64{}
	f.StringToInt64VarP(&p, name, shorthand, value, usage)
	return &p
}

// StringToInt64 defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]int64 variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToInt64(name string, value map[string]int64, usage string) *map[string]int64 {
	return CommandLine.StringToInt64P(name, "", value, usage)
}

// StringToInt64P is like StringToInt64, but accepts a shorthand letter that can be used after a single dash.
func StringToInt64P(name, shorthand string, value map[string]int64, usage string) *map[string]int64 {
	return CommandLine.StringToInt64P(name, shorthand, value, usage)
}
