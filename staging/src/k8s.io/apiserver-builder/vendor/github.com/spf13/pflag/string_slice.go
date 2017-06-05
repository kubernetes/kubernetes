package pflag

import (
	"bytes"
	"encoding/csv"
	"strings"
)

// -- stringSlice Value
type stringSliceValue struct {
	value   *[]string
	changed bool
}

func newStringSliceValue(val []string, p *[]string) *stringSliceValue {
	ssv := new(stringSliceValue)
	ssv.value = p
	*ssv.value = val
	return ssv
}

func readAsCSV(val string) ([]string, error) {
	if val == "" {
		return []string{}, nil
	}
	stringReader := strings.NewReader(val)
	csvReader := csv.NewReader(stringReader)
	return csvReader.Read()
}

func writeAsCSV(vals []string) (string, error) {
	b := &bytes.Buffer{}
	w := csv.NewWriter(b)
	err := w.Write(vals)
	if err != nil {
		return "", err
	}
	w.Flush()
	return strings.TrimSuffix(b.String(), "\n"), nil
}

func (s *stringSliceValue) Set(val string) error {
	v, err := readAsCSV(val)
	if err != nil {
		return err
	}
	if !s.changed {
		*s.value = v
	} else {
		*s.value = append(*s.value, v...)
	}
	s.changed = true
	return nil
}

func (s *stringSliceValue) Type() string {
	return "stringSlice"
}

func (s *stringSliceValue) String() string {
	str, _ := writeAsCSV(*s.value)
	return "[" + str + "]"
}

func stringSliceConv(sval string) (interface{}, error) {
	sval = sval[1 : len(sval)-1]
	// An empty string would cause a slice with one (empty) string
	if len(sval) == 0 {
		return []string{}, nil
	}
	return readAsCSV(sval)
}

// GetStringSlice return the []string value of a flag with the given name
func (f *FlagSet) GetStringSlice(name string) ([]string, error) {
	val, err := f.getFlagType(name, "stringSlice", stringSliceConv)
	if err != nil {
		return []string{}, err
	}
	return val.([]string), nil
}

// StringSliceVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a []string variable in which to store the value of the flag.
func (f *FlagSet) StringSliceVar(p *[]string, name string, value []string, usage string) {
	f.VarP(newStringSliceValue(value, p), name, "", usage)
}

// StringSliceVarP is like StringSliceVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringSliceVarP(p *[]string, name, shorthand string, value []string, usage string) {
	f.VarP(newStringSliceValue(value, p), name, shorthand, usage)
}

// StringSliceVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a []string variable in which to store the value of the flag.
func StringSliceVar(p *[]string, name string, value []string, usage string) {
	CommandLine.VarP(newStringSliceValue(value, p), name, "", usage)
}

// StringSliceVarP is like StringSliceVar, but accepts a shorthand letter that can be used after a single dash.
func StringSliceVarP(p *[]string, name, shorthand string, value []string, usage string) {
	CommandLine.VarP(newStringSliceValue(value, p), name, shorthand, usage)
}

// StringSlice defines a string flag with specified name, default value, and usage string.
// The return value is the address of a []string variable that stores the value of the flag.
func (f *FlagSet) StringSlice(name string, value []string, usage string) *[]string {
	p := []string{}
	f.StringSliceVarP(&p, name, "", value, usage)
	return &p
}

// StringSliceP is like StringSlice, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringSliceP(name, shorthand string, value []string, usage string) *[]string {
	p := []string{}
	f.StringSliceVarP(&p, name, shorthand, value, usage)
	return &p
}

// StringSlice defines a string flag with specified name, default value, and usage string.
// The return value is the address of a []string variable that stores the value of the flag.
func StringSlice(name string, value []string, usage string) *[]string {
	return CommandLine.StringSliceP(name, "", value, usage)
}

// StringSliceP is like StringSlice, but accepts a shorthand letter that can be used after a single dash.
func StringSliceP(name, shorthand string, value []string, usage string) *[]string {
	return CommandLine.StringSliceP(name, shorthand, value, usage)
}
