package pflag

// -- stringArray Value
type stringArrayValue struct {
	value   *[]string
	changed bool
}

func newStringArrayValue(val []string, p *[]string) *stringArrayValue {
	ssv := new(stringArrayValue)
	ssv.value = p
	*ssv.value = val
	return ssv
}

func (s *stringArrayValue) Set(val string) error {
	if !s.changed {
		*s.value = []string{val}
		s.changed = true
	} else {
		*s.value = append(*s.value, val)
	}
	return nil
}

func (s *stringArrayValue) Type() string {
	return "stringArray"
}

func (s *stringArrayValue) String() string {
	str, _ := writeAsCSV(*s.value)
	return "[" + str + "]"
}

func stringArrayConv(sval string) (interface{}, error) {
	sval = sval[1 : len(sval)-1]
	// An empty string would cause a array with one (empty) string
	if len(sval) == 0 {
		return []string{}, nil
	}
	return readAsCSV(sval)
}

// GetStringArray return the []string value of a flag with the given name
func (f *FlagSet) GetStringArray(name string) ([]string, error) {
	val, err := f.getFlagType(name, "stringArray", stringArrayConv)
	if err != nil {
		return []string{}, err
	}
	return val.([]string), nil
}

// StringArrayVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a []string variable in which to store the values of the multiple flags.
// The value of each argument will not try to be separated by comma. Use a StringSlice for that.
func (f *FlagSet) StringArrayVar(p *[]string, name string, value []string, usage string) {
	f.VarP(newStringArrayValue(value, p), name, "", usage)
}

// StringArrayVarP is like StringArrayVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringArrayVarP(p *[]string, name, shorthand string, value []string, usage string) {
	f.VarP(newStringArrayValue(value, p), name, shorthand, usage)
}

// StringArrayVar defines a string flag with specified name, default value, and usage string.
// The argument p points to a []string variable in which to store the value of the flag.
// The value of each argument will not try to be separated by comma. Use a StringSlice for that.
func StringArrayVar(p *[]string, name string, value []string, usage string) {
	CommandLine.VarP(newStringArrayValue(value, p), name, "", usage)
}

// StringArrayVarP is like StringArrayVar, but accepts a shorthand letter that can be used after a single dash.
func StringArrayVarP(p *[]string, name, shorthand string, value []string, usage string) {
	CommandLine.VarP(newStringArrayValue(value, p), name, shorthand, usage)
}

// StringArray defines a string flag with specified name, default value, and usage string.
// The return value is the address of a []string variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma. Use a StringSlice for that.
func (f *FlagSet) StringArray(name string, value []string, usage string) *[]string {
	p := []string{}
	f.StringArrayVarP(&p, name, "", value, usage)
	return &p
}

// StringArrayP is like StringArray, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) StringArrayP(name, shorthand string, value []string, usage string) *[]string {
	p := []string{}
	f.StringArrayVarP(&p, name, shorthand, value, usage)
	return &p
}

// StringArray defines a string flag with specified name, default value, and usage string.
// The return value is the address of a []string variable that stores the value of the flag.
// The value of each argument will not try to be separated by comma. Use a StringSlice for that.
func StringArray(name string, value []string, usage string) *[]string {
	return CommandLine.StringArrayP(name, "", value, usage)
}

// StringArrayP is like StringArray, but accepts a shorthand letter that can be used after a single dash.
func StringArrayP(name, shorthand string, value []string, usage string) *[]string {
	return CommandLine.StringArrayP(name, shorthand, value, usage)
}
