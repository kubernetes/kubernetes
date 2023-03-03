package pflag

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// -- stringToInt Value
type stringToIntValue struct {
	value   *map[string]int
	changed bool
}

func newStringToIntValue(val map[string]int, p *map[string]int) *stringToIntValue {
	ssv := new(stringToIntValue)
	ssv.value = p
	*ssv.value = val
	return ssv
}

// Format: a=1,b=2
func (s *stringToIntValue) Set(val string) error {
	ss := strings.Split(val, ",")
	out := make(map[string]int, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return fmt.Errorf("%s must be formatted as key=value", pair)
		}
		var err error
		out[kv[0]], err = strconv.Atoi(kv[1])
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

func (s *stringToIntValue) Type() string {
	return "stringToInt"
}

func (s *stringToIntValue) String() string {
	var buf bytes.Buffer
	i := 0
	for k, v := range *s.value {
		if i > 0 {
			buf.WriteRune(',')
		}
		buf.WriteString(k)
		buf.WriteRune('=')
		buf.WriteString(strconv.Itoa(v))
		i++
	}
	return "[" + buf.String() + "]"
}

func stringToIntConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// An empty string would cause an empty map
	if len(val) == 0 {
		return map[string]int{}, nil
	}
	ss := strings.Split(val, ",")
	out := make(map[string]int, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("%s must be formatted as key=value", pair)
		}
		var err error
		out[kv[0]], err = strconv.Atoi(kv[1])
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// GetStringToInt return the map[string]int value of a flag with the given name
func (f *FlagSet) GetStringToInt(name string) (map[string]int, error) {
	val, err := f.getFlagType(name, "stringToInt", stringToIntConv)
	if err != nil {
		return map[string]int{}, err
	}
	return val.(map[string]int), nil
}

// StringToIntVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a map[string]int variable in which to store the values of the multiple flags.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToIntVar(p *map[string]int, name string, value map[string]int, usage string) {
	f.VarP(newStringToIntValue(value, p), name, "", usage)
}

// StringToIntVarP is like StringToIntVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToIntVarP(p *map[string]int, name, shorthand string, value map[string]int, usage string) {
	f.VarP(newStringToIntValue(value, p), name, shorthand, usage)
}

// StringToIntVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a map[string]int variable in which to store the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToIntVar(p *map[string]int, name string, value map[string]int, usage string) {
	CommandLine.VarP(newStringToIntValue(value, p), name, "", usage)
}

// StringToIntVarP is like StringToIntVar, but accepts a shorthand letter that can be used after a single dash.
func StringToIntVarP(p *map[string]int, name, shorthand string, value map[string]int, usage string) {
	CommandLine.VarP(newStringToIntValue(value, p), name, shorthand, usage)
}

// StringToInt defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]int variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToInt(name string, value map[string]int, usage string) *map[string]int {
	p := map[string]int{}
	f.StringToIntVarP(&p, name, "", value, usage)
	return &p
}

// StringToIntP is like StringToInt, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToIntP(name, shorthand string, value map[string]int, usage string) *map[string]int {
	p := map[string]int{}
	f.StringToIntVarP(&p, name, shorthand, value, usage)
	return &p
}

// StringToInt defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]int variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToInt(name string, value map[string]int, usage string) *map[string]int {
	return CommandLine.StringToIntP(name, "", value, usage)
}

// StringToIntP is like StringToInt, but accepts a shorthand letter that can be used after a single dash.
func StringToIntP(name, shorthand string, value map[string]int, usage string) *map[string]int {
	return CommandLine.StringToIntP(name, shorthand, value, usage)
}
