package pflag

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"strings"
)

// -- stringToString Value
type stringToStringValue struct {
	value   *map[string]string
	changed bool
}

func newStringToStringValue(val map[string]string, p *map[string]string) *stringToStringValue {
	ssv := new(stringToStringValue)
	ssv.value = p
	*ssv.value = val
	return ssv
}

// Format: a=1,b=2
func (s *stringToStringValue) Set(val string) error {
	var ss []string
	n := strings.Count(val, "=")
	switch n {
	case 0:
		return fmt.Errorf("%s must be formatted as key=value", val)
	case 1:
		ss = append(ss, strings.Trim(val, `"`))
	default:
		r := csv.NewReader(strings.NewReader(val))
		var err error
		ss, err = r.Read()
		if err != nil {
			return err
		}
	}

	out := make(map[string]string, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return fmt.Errorf("%s must be formatted as key=value", pair)
		}
		out[kv[0]] = kv[1]
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

func (s *stringToStringValue) Type() string {
	return "stringToString"
}

func (s *stringToStringValue) String() string {
	records := make([]string, 0, len(*s.value)>>1)
	for k, v := range *s.value {
		records = append(records, k+"="+v)
	}

	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	if err := w.Write(records); err != nil {
		panic(err)
	}
	w.Flush()
	return "[" + strings.TrimSpace(buf.String()) + "]"
}

func stringToStringConv(val string) (interface{}, error) {
	val = strings.Trim(val, "[]")
	// An empty string would cause an empty map
	if len(val) == 0 {
		return map[string]string{}, nil
	}
	r := csv.NewReader(strings.NewReader(val))
	ss, err := r.Read()
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(ss))
	for _, pair := range ss {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("%s must be formatted as key=value", pair)
		}
		out[kv[0]] = kv[1]
	}
	return out, nil
}

// GetStringToString return the map[string]string value of a flag with the given name
func (f *FlagSet) GetStringToString(name string) (map[string]string, error) {
	val, err := f.getFlagType(name, "stringToString", stringToStringConv)
	if err != nil {
		return map[string]string{}, err
	}
	return val.(map[string]string), nil
}

// StringToStringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a map[string]string variable in which to store the values of the multiple flags.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToStringVar(p *map[string]string, name string, value map[string]string, usage string) {
	f.VarP(newStringToStringValue(value, p), name, "", usage)
}

// StringToStringVarP is like StringToStringVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToStringVarP(p *map[string]string, name, shorthand string, value map[string]string, usage string) {
	f.VarP(newStringToStringValue(value, p), name, shorthand, usage)
}

// StringToStringVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a map[string]string variable in which to store the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToStringVar(p *map[string]string, name string, value map[string]string, usage string) {
	CommandLine.VarP(newStringToStringValue(value, p), name, "", usage)
}

// StringToStringVarP is like StringToStringVar, but accepts a shorthand letter that can be used after a single dash.
func StringToStringVarP(p *map[string]string, name, shorthand string, value map[string]string, usage string) {
	CommandLine.VarP(newStringToStringValue(value, p), name, shorthand, usage)
}

// StringToString defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]string variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func (f *FlagSet) StringToString(name string, value map[string]string, usage string) *map[string]string {
	p := map[string]string{}
	f.StringToStringVarP(&p, name, "", value, usage)
	return &p
}

// StringToStringP is like StringToString, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringToStringP(name, shorthand string, value map[string]string, usage string) *map[string]string {
	p := map[string]string{}
	f.StringToStringVarP(&p, name, shorthand, value, usage)
	return &p
}

// StringToString defines a string flag with specified name, default value, and usage string.
// The return value is the address of a map[string]string variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma
func StringToString(name string, value map[string]string, usage string) *map[string]string {
	return CommandLine.StringToStringP(name, "", value, usage)
}

// StringToStringP is like StringToString, but accepts a shorthand letter that can be used after a single dash.
func StringToStringP(name, shorthand string, value map[string]string, usage string) *map[string]string {
	return CommandLine.StringToStringP(name, shorthand, value, usage)
}
