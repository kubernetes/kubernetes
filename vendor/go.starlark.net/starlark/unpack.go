package starlark

// This file defines the Unpack helper functions used by
// built-in functions to interpret their call arguments.

import (
	"fmt"
	"log"
	"reflect"
	"strings"

	"go.starlark.net/internal/spell"
)

// An Unpacker defines custom argument unpacking behavior.
// See UnpackArgs.
type Unpacker interface {
	Unpack(v Value) error
}

// UnpackArgs unpacks the positional and keyword arguments into the
// supplied parameter variables.  pairs is an alternating list of names
// and pointers to variables.
//
// If the variable is a bool, integer, string, *List, *Dict, Callable,
// Iterable, or user-defined implementation of Value,
// UnpackArgs performs the appropriate type check.
// Predeclared Go integer types uses the AsInt check.
//
// If the parameter name ends with "?", it is optional.
//
// If the parameter name ends with "??", it is optional and treats the None value
// as if the argument was absent.
//
// If a parameter is marked optional, then all following parameters are
// implicitly optional where or not they are marked.
//
// If the variable implements Unpacker, its Unpack argument
// is called with the argument value, allowing an application
// to define its own argument validation and conversion.
//
// If the variable implements Value, UnpackArgs may call
// its Type() method while constructing the error message.
//
// Examples:
//
//      var (
//          a Value
//          b = MakeInt(42)
//          c Value = starlark.None
//      )
//
//      // 1. mixed parameters, like def f(a, b=42, c=None).
//      err := UnpackArgs("f", args, kwargs, "a", &a, "b?", &b, "c?", &c)
//
//      // 2. keyword parameters only, like def f(*, a, b, c=None).
//      if len(args) > 0 {
//              return fmt.Errorf("f: unexpected positional arguments")
//      }
//      err := UnpackArgs("f", args, kwargs, "a", &a, "b?", &b, "c?", &c)
//
//      // 3. positional parameters only, like def f(a, b=42, c=None, /) in Python 3.8.
//      err := UnpackPositionalArgs("f", args, kwargs, 1, &a, &b, &c)
//
// More complex forms such as def f(a, b=42, *args, c, d=123, **kwargs)
// require additional logic, but their need in built-ins is exceedingly rare.
//
// In the examples above, the declaration of b with type Int causes UnpackArgs
// to require that b's argument value, if provided, is also an int.
// To allow arguments of any type, while retaining the default value of 42,
// declare b as a Value:
//
//	var b Value = MakeInt(42)
//
// The zero value of a variable of type Value, such as 'a' in the
// examples above, is not a valid Starlark value, so if the parameter is
// optional, the caller must explicitly handle the default case by
// interpreting nil as None or some computed default. The same is true
// for the zero values of variables of type *List, *Dict, Callable, or
// Iterable. For example:
//
//      // def myfunc(d=None, e=[], f={})
//      var (
//          d Value
//          e *List
//          f *Dict
//      )
//      err := UnpackArgs("myfunc", args, kwargs, "d?", &d, "e?", &e, "f?", &f)
//      if d == nil { d = None; }
//      if e == nil { e = new(List); }
//      if f == nil { f = new(Dict); }
//
func UnpackArgs(fnname string, args Tuple, kwargs []Tuple, pairs ...interface{}) error {
	nparams := len(pairs) / 2
	var defined intset
	defined.init(nparams)

	paramName := func(x interface{}) (name string, skipNone bool) { // (no free variables)
		name = x.(string)
		if strings.HasSuffix(name, "??") {
			name = strings.TrimSuffix(name, "??")
			skipNone = true
		} else if name[len(name)-1] == '?' {
			name = name[:len(name)-1]
		}

		return name, skipNone
	}

	// positional arguments
	if len(args) > nparams {
		return fmt.Errorf("%s: got %d arguments, want at most %d",
			fnname, len(args), nparams)
	}
	for i, arg := range args {
		defined.set(i)
		name, skipNone := paramName(pairs[2*i])
		if skipNone {
			if _, isNone := arg.(NoneType); isNone {
				continue
			}
		}
		if err := unpackOneArg(arg, pairs[2*i+1]); err != nil {
			return fmt.Errorf("%s: for parameter %s: %s", fnname, name, err)
		}
	}

	// keyword arguments
kwloop:
	for _, item := range kwargs {
		name, arg := item[0].(String), item[1]
		for i := 0; i < nparams; i++ {
			pName, skipNone := paramName(pairs[2*i])
			if pName == string(name) {
				// found it
				if defined.set(i) {
					return fmt.Errorf("%s: got multiple values for keyword argument %s",
						fnname, name)
				}

				if skipNone {
					if _, isNone := arg.(NoneType); isNone {
						continue kwloop
					}
				}

				ptr := pairs[2*i+1]
				if err := unpackOneArg(arg, ptr); err != nil {
					return fmt.Errorf("%s: for parameter %s: %s", fnname, name, err)
				}
				continue kwloop
			}
		}
		err := fmt.Errorf("%s: unexpected keyword argument %s", fnname, name)
		names := make([]string, 0, nparams)
		for i := 0; i < nparams; i += 2 {
			param, _ := paramName(pairs[i])
			names = append(names, param)
		}
		if n := spell.Nearest(string(name), names); n != "" {
			err = fmt.Errorf("%s (did you mean %s?)", err.Error(), n)
		}
		return err
	}

	// Check that all non-optional parameters are defined.
	// (We needn't check the first len(args).)
	for i := len(args); i < nparams; i++ {
		name := pairs[2*i].(string)
		if strings.HasSuffix(name, "?") {
			break // optional
		}
		if !defined.get(i) {
			return fmt.Errorf("%s: missing argument for %s", fnname, name)
		}
	}

	return nil
}

// UnpackPositionalArgs unpacks the positional arguments into
// corresponding variables.  Each element of vars is a pointer; see
// UnpackArgs for allowed types and conversions.
//
// UnpackPositionalArgs reports an error if the number of arguments is
// less than min or greater than len(vars), if kwargs is nonempty, or if
// any conversion fails.
//
// See UnpackArgs for general comments.
func UnpackPositionalArgs(fnname string, args Tuple, kwargs []Tuple, min int, vars ...interface{}) error {
	if len(kwargs) > 0 {
		return fmt.Errorf("%s: unexpected keyword arguments", fnname)
	}
	max := len(vars)
	if len(args) < min {
		var atleast string
		if min < max {
			atleast = "at least "
		}
		return fmt.Errorf("%s: got %d arguments, want %s%d", fnname, len(args), atleast, min)
	}
	if len(args) > max {
		var atmost string
		if max > min {
			atmost = "at most "
		}
		return fmt.Errorf("%s: got %d arguments, want %s%d", fnname, len(args), atmost, max)
	}
	for i, arg := range args {
		if err := unpackOneArg(arg, vars[i]); err != nil {
			return fmt.Errorf("%s: for parameter %d: %s", fnname, i+1, err)
		}
	}
	return nil
}

func unpackOneArg(v Value, ptr interface{}) error {
	// On failure, don't clobber *ptr.
	switch ptr := ptr.(type) {
	case Unpacker:
		return ptr.Unpack(v)
	case *Value:
		*ptr = v
	case *string:
		s, ok := AsString(v)
		if !ok {
			return fmt.Errorf("got %s, want string", v.Type())
		}
		*ptr = s
	case *bool:
		b, ok := v.(Bool)
		if !ok {
			return fmt.Errorf("got %s, want bool", v.Type())
		}
		*ptr = bool(b)
	case *int, *int8, *int16, *int32, *int64,
		*uint, *uint8, *uint16, *uint32, *uint64, *uintptr:
		return AsInt(v, ptr)
	case *float64:
		f, ok := v.(Float)
		if !ok {
			return fmt.Errorf("got %s, want float", v.Type())
		}
		*ptr = float64(f)
	case **List:
		list, ok := v.(*List)
		if !ok {
			return fmt.Errorf("got %s, want list", v.Type())
		}
		*ptr = list
	case **Dict:
		dict, ok := v.(*Dict)
		if !ok {
			return fmt.Errorf("got %s, want dict", v.Type())
		}
		*ptr = dict
	case *Callable:
		f, ok := v.(Callable)
		if !ok {
			return fmt.Errorf("got %s, want callable", v.Type())
		}
		*ptr = f
	case *Iterable:
		it, ok := v.(Iterable)
		if !ok {
			return fmt.Errorf("got %s, want iterable", v.Type())
		}
		*ptr = it
	default:
		// v must have type *V, where V is some subtype of starlark.Value.
		ptrv := reflect.ValueOf(ptr)
		if ptrv.Kind() != reflect.Ptr {
			log.Panicf("internal error: not a pointer: %T", ptr)
		}
		paramVar := ptrv.Elem()
		if !reflect.TypeOf(v).AssignableTo(paramVar.Type()) {
			// The value is not assignable to the variable.

			// Detect a possible bug in the Go program that called Unpack:
			// If the variable *ptr is not a subtype of Value,
			// no value of v can possibly work.
			if !paramVar.Type().AssignableTo(reflect.TypeOf(new(Value)).Elem()) {
				log.Panicf("pointer element type does not implement Value: %T", ptr)
			}

			// Report Starlark dynamic type error.
			//
			// We prefer the Starlark Value.Type name over
			// its Go reflect.Type name, but calling the
			// Value.Type method on the variable is not safe
			// in general. If the variable is an interface,
			// the call will fail. Even if the variable has
			// a concrete type, it might not be safe to call
			// Type() on a zero instance. Thus we must use
			// recover.

			// Default to Go reflect.Type name
			paramType := paramVar.Type().String()

			// Attempt to call Value.Type method.
			func() {
				defer func() { recover() }()
				paramType = paramVar.MethodByName("Type").Call(nil)[0].String()
			}()
			return fmt.Errorf("got %s, want %s", v.Type(), paramType)
		}
		paramVar.Set(reflect.ValueOf(v))
	}
	return nil
}

type intset struct {
	small uint64       // bitset, used if n < 64
	large map[int]bool //    set, used if n >= 64
}

func (is *intset) init(n int) {
	if n >= 64 {
		is.large = make(map[int]bool)
	}
}

func (is *intset) set(i int) (prev bool) {
	if is.large == nil {
		prev = is.small&(1<<uint(i)) != 0
		is.small |= 1 << uint(i)
	} else {
		prev = is.large[i]
		is.large[i] = true
	}
	return
}

func (is *intset) get(i int) bool {
	if is.large == nil {
		return is.small&(1<<uint(i)) != 0
	}
	return is.large[i]
}

func (is *intset) len() int {
	if is.large == nil {
		// Suboptimal, but used only for error reporting.
		len := 0
		for i := 0; i < 64; i++ {
			if is.small&(1<<uint(i)) != 0 {
				len++
			}
		}
		return len
	}
	return len(is.large)
}
