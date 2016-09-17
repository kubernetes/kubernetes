package hil

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/hashicorp/hil/ast"
	"github.com/mitchellh/reflectwalk"
)

// WalkFn is the type of function to pass to Walk. Modify fields within
// WalkData to control whether replacement happens.
type WalkFn func(*WalkData) error

// WalkData is the structure passed to the callback of the Walk function.
//
// This structure contains data passed in as well as fields that are expected
// to be written by the caller as a result. Please see the documentation for
// each field for more information.
type WalkData struct {
	// Root is the parsed root of this HIL program
	Root ast.Node

	// Location is the location within the structure where this
	// value was found. This can be used to modify behavior within
	// slices and so on.
	Location reflectwalk.Location

	// The below two values must be set by the callback to have any effect.
	//
	// Replace, if true, will replace the value in the structure with
	// ReplaceValue. It is up to the caller to make sure this is a string.
	Replace      bool
	ReplaceValue string
}

// Walk will walk an arbitrary Go structure and parse any string as an
// HIL program and call the callback cb to determine what to replace it
// with.
//
// This function is very useful for arbitrary HIL program interpolation
// across a complex configuration structure. Due to the heavy use of
// reflection in this function, it is recommend to write many unit tests
// with your typical configuration structures to hilp mitigate the risk
// of panics.
func Walk(v interface{}, cb WalkFn) error {
	walker := &interpolationWalker{F: cb}
	return reflectwalk.Walk(v, walker)
}

// interpolationWalker implements interfaces for the reflectwalk package
// (github.com/mitchellh/reflectwalk) that can be used to automatically
// execute a callback for an interpolation.
type interpolationWalker struct {
	F WalkFn

	key         []string
	lastValue   reflect.Value
	loc         reflectwalk.Location
	cs          []reflect.Value
	csKey       []reflect.Value
	csData      interface{}
	sliceIndex  int
	unknownKeys []string
}

func (w *interpolationWalker) Enter(loc reflectwalk.Location) error {
	w.loc = loc
	return nil
}

func (w *interpolationWalker) Exit(loc reflectwalk.Location) error {
	w.loc = reflectwalk.None

	switch loc {
	case reflectwalk.Map:
		w.cs = w.cs[:len(w.cs)-1]
	case reflectwalk.MapValue:
		w.key = w.key[:len(w.key)-1]
		w.csKey = w.csKey[:len(w.csKey)-1]
	case reflectwalk.Slice:
		// Split any values that need to be split
		w.splitSlice()
		w.cs = w.cs[:len(w.cs)-1]
	case reflectwalk.SliceElem:
		w.csKey = w.csKey[:len(w.csKey)-1]
	}

	return nil
}

func (w *interpolationWalker) Map(m reflect.Value) error {
	w.cs = append(w.cs, m)
	return nil
}

func (w *interpolationWalker) MapElem(m, k, v reflect.Value) error {
	w.csData = k
	w.csKey = append(w.csKey, k)
	w.key = append(w.key, k.String())
	w.lastValue = v
	return nil
}

func (w *interpolationWalker) Slice(s reflect.Value) error {
	w.cs = append(w.cs, s)
	return nil
}

func (w *interpolationWalker) SliceElem(i int, elem reflect.Value) error {
	w.csKey = append(w.csKey, reflect.ValueOf(i))
	w.sliceIndex = i
	return nil
}

func (w *interpolationWalker) Primitive(v reflect.Value) error {
	setV := v

	// We only care about strings
	if v.Kind() == reflect.Interface {
		setV = v
		v = v.Elem()
	}
	if v.Kind() != reflect.String {
		return nil
	}

	astRoot, err := Parse(v.String())
	if err != nil {
		return err
	}

	// If the AST we got is just a literal string value with the same
	// value then we ignore it. We have to check if its the same value
	// because it is possible to input a string, get out a string, and
	// have it be different. For example: "foo-$${bar}" turns into
	// "foo-${bar}"
	if n, ok := astRoot.(*ast.LiteralNode); ok {
		if s, ok := n.Value.(string); ok && s == v.String() {
			return nil
		}
	}

	if w.F == nil {
		return nil
	}

	data := WalkData{Root: astRoot, Location: w.loc}
	if err := w.F(&data); err != nil {
		return fmt.Errorf(
			"%s in:\n\n%s",
			err, v.String())
	}

	if data.Replace {
		/*
			if remove {
				w.removeCurrent()
				return nil
			}
		*/

		resultVal := reflect.ValueOf(data.ReplaceValue)
		switch w.loc {
		case reflectwalk.MapKey:
			m := w.cs[len(w.cs)-1]

			// Delete the old value
			var zero reflect.Value
			m.SetMapIndex(w.csData.(reflect.Value), zero)

			// Set the new key with the existing value
			m.SetMapIndex(resultVal, w.lastValue)

			// Set the key to be the new key
			w.csData = resultVal
		case reflectwalk.MapValue:
			// If we're in a map, then the only way to set a map value is
			// to set it directly.
			m := w.cs[len(w.cs)-1]
			mk := w.csData.(reflect.Value)
			m.SetMapIndex(mk, resultVal)
		default:
			// Otherwise, we should be addressable
			setV.Set(resultVal)
		}
	}

	return nil
}

func (w *interpolationWalker) removeCurrent() {
	// Append the key to the unknown keys
	w.unknownKeys = append(w.unknownKeys, strings.Join(w.key, "."))

	for i := 1; i <= len(w.cs); i++ {
		c := w.cs[len(w.cs)-i]
		switch c.Kind() {
		case reflect.Map:
			// Zero value so that we delete the map key
			var val reflect.Value

			// Get the key and delete it
			k := w.csData.(reflect.Value)
			c.SetMapIndex(k, val)
			return
		}
	}

	panic("No container found for removeCurrent")
}

func (w *interpolationWalker) replaceCurrent(v reflect.Value) {
	c := w.cs[len(w.cs)-2]
	switch c.Kind() {
	case reflect.Map:
		// Get the key and delete it
		k := w.csKey[len(w.csKey)-1]
		c.SetMapIndex(k, v)
	}
}

func (w *interpolationWalker) splitSlice() {
	// Get the []interface{} slice so we can do some operations on
	// it without dealing with reflection. We'll document each step
	// here to be clear.
	var s []interface{}
	raw := w.cs[len(w.cs)-1]
	switch v := raw.Interface().(type) {
	case []interface{}:
		s = v
	case []map[string]interface{}:
		return
	default:
		panic("Unknown kind: " + raw.Kind().String())
	}

	// Check if we have any elements that we need to split. If not, then
	// just return since we're done.
	split := false
	if !split {
		return
	}

	// Make a new result slice that is twice the capacity to fit our growth.
	result := make([]interface{}, 0, len(s)*2)

	// Go over each element of the original slice and start building up
	// the resulting slice by splitting where we have to.
	for _, v := range s {
		sv, ok := v.(string)
		if !ok {
			// Not a string, so just set it
			result = append(result, v)
			continue
		}

		// Not a string list, so just set it
		result = append(result, sv)
	}

	// Our slice is now done, we have to replace the slice now
	// with this new one that we have.
	w.replaceCurrent(reflect.ValueOf(result))
}
