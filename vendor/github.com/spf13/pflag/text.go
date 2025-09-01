package pflag

import (
	"encoding"
	"fmt"
	"reflect"
)

// following is copied from go 1.23.4 flag.go
type textValue struct{ p encoding.TextUnmarshaler }

func newTextValue(val encoding.TextMarshaler, p encoding.TextUnmarshaler) textValue {
	ptrVal := reflect.ValueOf(p)
	if ptrVal.Kind() != reflect.Ptr {
		panic("variable value type must be a pointer")
	}
	defVal := reflect.ValueOf(val)
	if defVal.Kind() == reflect.Ptr {
		defVal = defVal.Elem()
	}
	if defVal.Type() != ptrVal.Type().Elem() {
		panic(fmt.Sprintf("default type does not match variable type: %v != %v", defVal.Type(), ptrVal.Type().Elem()))
	}
	ptrVal.Elem().Set(defVal)
	return textValue{p}
}

func (v textValue) Set(s string) error {
	return v.p.UnmarshalText([]byte(s))
}

func (v textValue) Get() interface{} {
	return v.p
}

func (v textValue) String() string {
	if m, ok := v.p.(encoding.TextMarshaler); ok {
		if b, err := m.MarshalText(); err == nil {
			return string(b)
		}
	}
	return ""
}

//end of copy

func (v textValue) Type() string {
	return reflect.ValueOf(v.p).Type().Name()
}

// GetText set out, which implements encoding.UnmarshalText, to the value of a flag with given name
func (f *FlagSet) GetText(name string, out encoding.TextUnmarshaler) error {
	flag := f.Lookup(name)
	if flag == nil {
		return fmt.Errorf("flag accessed but not defined: %s", name)
	}
	if flag.Value.Type() != reflect.TypeOf(out).Name() {
		return fmt.Errorf("trying to get %s value of flag of type %s", reflect.TypeOf(out).Name(), flag.Value.Type())
	}
	return out.UnmarshalText([]byte(flag.Value.String()))
}

// TextVar defines a flag with a specified name, default value, and usage string. The argument p must be a pointer to a variable that will hold the value of the flag, and p must implement encoding.TextUnmarshaler. If the flag is used, the flag value will be passed to p's UnmarshalText method. The type of the default value must be the same as the type of p.
func (f *FlagSet) TextVar(p encoding.TextUnmarshaler, name string, value encoding.TextMarshaler, usage string) {
	f.VarP(newTextValue(value, p), name, "", usage)
}

// TextVarP is like TextVar, but accepts a shorthand letter that can be used after a single dash.
func (f *FlagSet) TextVarP(p encoding.TextUnmarshaler, name, shorthand string, value encoding.TextMarshaler, usage string) {
	f.VarP(newTextValue(value, p), name, shorthand, usage)
}

// TextVar defines a flag with a specified name, default value, and usage string. The argument p must be a pointer to a variable that will hold the value of the flag, and p must implement encoding.TextUnmarshaler. If the flag is used, the flag value will be passed to p's UnmarshalText method. The type of the default value must be the same as the type of p.
func TextVar(p encoding.TextUnmarshaler, name string, value encoding.TextMarshaler, usage string) {
	CommandLine.VarP(newTextValue(value, p), name, "", usage)
}

// TextVarP is like TextVar, but accepts a shorthand letter that can be used after a single dash.
func TextVarP(p encoding.TextUnmarshaler, name, shorthand string, value encoding.TextMarshaler, usage string) {
	CommandLine.VarP(newTextValue(value, p), name, shorthand, usage)
}
