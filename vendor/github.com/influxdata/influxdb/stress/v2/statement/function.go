package statement

import (
	crypto "crypto/rand"
	"fmt"
	"math/rand"
)

// ################
// # Function     #
// ################

// Function is a struct that holds information for generating values in templated points
type Function struct {
	Type     string
	Fn       string
	Argument int
	Count    int
}

// NewStringer creates a new Stringer
func (f *Function) NewStringer(series int) Stringer {
	var fn Stringer
	switch f.Type {
	case "int":
		fn = NewIntFunc(f.Fn, f.Argument)
	case "float":
		fn = NewFloatFunc(f.Fn, f.Argument)
	case "str":
		fn = NewStrFunc(f.Fn, f.Argument)
	default:
		fn = func() string { return "STRINGER ERROR" }
	}

	if int(f.Count) != 0 {
		return cycle(f.Count, fn)
	}

	return nTimes(series, fn)

}

// ################
// # Stringers    #
// ################

// Stringers is a collection of Stringer
type Stringers []Stringer

// Eval returns an array of all the Stringer functions evaluated once
func (s Stringers) Eval(time func() int64) []interface{} {
	arr := make([]interface{}, len(s)+1)

	for i, st := range s {
		arr[i] = st()
	}

	arr[len(s)] = time()

	return arr
}

// Stringer is a function that returns a string
type Stringer func() string

func randStr(n int) func() string {
	return func() string {
		b := make([]byte, n/2)

		_, _ = crypto.Read(b)

		return fmt.Sprintf("%x", b)
	}
}

// NewStrFunc reates a new striger to create strings for templated writes
func NewStrFunc(fn string, arg int) Stringer {
	switch fn {
	case "rand":
		return randStr(arg)
	default:
		return func() string { return "STR ERROR" }
	}
}

func randFloat(n int) func() string {
	return func() string {
		return fmt.Sprintf("%v", rand.Intn(n))
	}
}

func incFloat(n int) func() string {
	i := n
	return func() string {
		s := fmt.Sprintf("%v", i)
		i++
		return s
	}
}

// NewFloatFunc reates a new striger to create float values for templated writes
func NewFloatFunc(fn string, arg int) Stringer {
	switch fn {
	case "rand":
		return randFloat(arg)
	case "inc":
		return incFloat(arg)
	default:
		return func() string { return "FLOAT ERROR" }
	}
}

func randInt(n int) Stringer {
	return func() string {
		return fmt.Sprintf("%vi", rand.Intn(n))
	}
}

func incInt(n int) Stringer {
	i := n
	return func() string {
		s := fmt.Sprintf("%vi", i)
		i++
		return s
	}
}

// NewIntFunc reates a new striger to create int values for templated writes
func NewIntFunc(fn string, arg int) Stringer {
	switch fn {
	case "rand":
		return randInt(arg)
	case "inc":
		return incInt(arg)
	default:
		return func() string { return "INT ERROR" }
	}
}

// nTimes will return the previous return value of a function
// n-many times before calling the function again
func nTimes(n int, fn Stringer) Stringer {
	i := 0
	t := fn()
	return func() string {
		i++
		if i > n {
			t = fn()
			i = 1
		}
		return t
	}
}

// cycle will cycle through a list of values before repeating them

func cycle(n int, fn Stringer) Stringer {
	if n == 0 {
		return fn
	}
	i := 0
	cache := make([]string, n)
	t := fn()
	cache[i] = t

	return func() string {
		i++

		if i < n {
			cache[i] = fn()
		}

		t = cache[(i-1)%n]
		return t
	}
}
