/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modes

import (
	"encoding"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"github.com/fxamacker/cbor/v2"
)

// Returns a non-nil error if and only if the argument's type (or one of its component types, for
// composite types) implements json.Marshaler or encoding.TextMarshaler without also implementing
// cbor.Marshaler and likewise for the respective Unmarshaler interfaces.
//
// This is a temporary, graduation-blocking restriction and will be removed in favor of automatic
// transcoding between CBOR and JSON/text for these types. This restriction allows CBOR to be
// exercised for in-tree and unstructured types while mitigating the risk of mangling out-of-tree
// types in client programs.
func RejectCustomMarshalers(v interface{}) error {
	if v == nil {
		return nil
	}
	rv := reflect.ValueOf(v)
	if err := marshalerCache.getChecker(rv.Type()).check(rv, maxDepth); err != nil {
		return fmt.Errorf("unable to serialize %T: %w", v, err)
	}
	if err := unmarshalerCache.getChecker(rv.Type()).check(rv, maxDepth); err != nil {
		return fmt.Errorf("unable to serialize %T: %w", v, err)
	}
	return nil
}

// Recursion depth is limited as a basic mitigation against cyclic objects. Objects created by the
// decoder shouldn't be able to contain cycles, but practically any object can be passed to the
// encoder.
var errMaxDepthExceeded = errors.New("object depth exceeds limit (possible cycle?)")

// The JSON encoder begins detecting cycles after depth 1000. Use a generous limit here, knowing
// that it can might deeply nested acyclic objects. The limit will be removed along with the rest of
// this mechanism.
const maxDepth = 2048

var marshalerCache = checkers{
	cborInterface: reflect.TypeFor[cbor.Marshaler](),
	nonCBORInterfaces: []reflect.Type{
		reflect.TypeFor[json.Marshaler](),
		reflect.TypeFor[encoding.TextMarshaler](),
	},
}

var unmarshalerCache = checkers{
	cborInterface: reflect.TypeFor[cbor.Unmarshaler](),
	nonCBORInterfaces: []reflect.Type{
		reflect.TypeFor[json.Unmarshaler](),
		reflect.TypeFor[encoding.TextUnmarshaler](),
	},
	assumeAddressableValues: true,
}

// checker wraps a function for dynamically checking a value of a specific type for custom JSON
// behaviors not matched by a custom CBOR behavior.
type checker struct {
	// check returns a non-nil error if the given value might be marshalled to or from CBOR
	// using the default behavior for its kind, but marshalled to or from JSON using custom
	// behavior.
	check func(rv reflect.Value, depth int) error

	// safe returns true if all values of this type are safe from mismatched custom marshalers.
	safe func() bool
}

// TODO: stale
// Having a single addressable checker for comparisons lets us prune and collapse parts of the
// object traversal that are statically known to be safe. Depending on the type, it may be
// unnecessary to inspect each value of that type. For example, no value of the built-in type bool
// can implement json.Marshaler (a named type whose underlying type is bool could, but it is a
// distinct type from bool).
var noop = checker{
	safe: func() bool {
		return true
	},
	check: func(rv reflect.Value, depth int) error {
		return nil
	},
}

type checkers struct {
	m sync.Map // reflect.Type => *checker

	cborInterface     reflect.Type
	nonCBORInterfaces []reflect.Type

	assumeAddressableValues bool
}

func (cache *checkers) getChecker(rt reflect.Type) checker {
	if ptr, ok := cache.m.Load(rt); ok {
		return *ptr.(*checker)
	}

	return cache.getCheckerInternal(rt, nil)
}

// linked list node representing the path from a composite type to an element type
type path struct {
	Type   reflect.Type
	Parent *path
}

func (p path) cyclic(rt reflect.Type) bool {
	for ancestor := &p; ancestor != nil; ancestor = ancestor.Parent {
		if ancestor.Type == rt {
			return true
		}
	}
	return false
}

func (cache *checkers) getCheckerInternal(rt reflect.Type, parent *path) (c checker) {
	// Store a placeholder cache entry first to handle cyclic types.
	var wg sync.WaitGroup
	wg.Add(1)
	defer wg.Done()
	placeholder := checker{
		safe: func() bool {
			wg.Wait()
			return c.safe()
		},
		check: func(rv reflect.Value, depth int) error {
			wg.Wait()
			return c.check(rv, depth)
		},
	}
	if actual, loaded := cache.m.LoadOrStore(rt, &placeholder); loaded {
		// Someone else stored an entry for this type, use it.
		return *actual.(*checker)
	}

	// Take a nonreflective path for the unstructured container types. They're common and
	// usually nested inside one another.
	switch rt {
	case reflect.TypeFor[map[string]interface{}](), reflect.TypeFor[[]interface{}]():
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				return checkUnstructuredValue(cache, rv.Interface(), depth)
			},
		}
	}

	// It's possible that one of the relevant interfaces is implemented on a type with a pointer
	// receiver, but that a particular value of that type is not addressable. For example:
	//
	//   func (Foo) MarshalText() ([]byte, error) { ... }
	//   func (*Foo) MarshalCBOR() ([]byte, error) { ... }
	//
	// Both methods are in the method set of *Foo, but the method set of Foo contains only
	// MarshalText.
	//
	// Both the unmarshaler and marshaler checks assume that methods implementing a JSON or text
	// interface with a pointer receiver are always accessible. Only the unmarshaler check
	// assumes that CBOR methods with pointer receivers are accessible.

	if rt.Implements(cache.cborInterface) {
		return noop
	}
	for _, unsafe := range cache.nonCBORInterfaces {
		if rt.Implements(unsafe) {
			err := fmt.Errorf("%v implements %v without corresponding cbor interface", rt, unsafe)
			return checker{
				safe: func() bool {
					return false
				},
				check: func(reflect.Value, int) error {
					return err
				},
			}
		}
	}

	if cache.assumeAddressableValues && reflect.PointerTo(rt).Implements(cache.cborInterface) {
		return noop
	}
	for _, unsafe := range cache.nonCBORInterfaces {
		if reflect.PointerTo(rt).Implements(unsafe) {
			err := fmt.Errorf("%v implements %v without corresponding cbor interface", reflect.PointerTo(rt), unsafe)
			return checker{
				safe: func() bool {
					return false
				},
				check: func(reflect.Value, int) error {
					return err
				},
			}
		}
	}

	self := &path{Type: rt, Parent: parent}

	switch rt.Kind() {
	case reflect.Array:
		ce := cache.getCheckerInternal(rt.Elem(), self)
		rtlen := rt.Len()
		if rtlen == 0 || (!self.cyclic(rt.Elem()) && ce.safe()) {
			return noop
		}
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				for i := 0; i < rtlen; i++ {
					if err := ce.check(rv.Index(i), depth-1); err != nil {
						return err
					}
				}
				return nil
			},
		}

	case reflect.Interface:
		// All interface values have to be checked because their dynamic type might
		// implement one of the interesting interfaces or be composed of another type that
		// does.
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if rv.IsNil() {
					return nil
				}
				// Unpacking interfaces must count against recursion depth,
				// consider this cycle:
				// >  var i interface{}
				// >  var p *interface{} = &i
				// >  i = p
				// >  rv := reflect.ValueOf(i)
				// >  for {
				// >    rv = rv.Elem()
				// >  }
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				rv = rv.Elem()
				return cache.getChecker(rv.Type()).check(rv, depth-1)
			},
		}

	case reflect.Map:
		rtk := rt.Key()
		ck := cache.getCheckerInternal(rtk, self)
		rte := rt.Elem()
		ce := cache.getCheckerInternal(rte, self)
		if !self.cyclic(rtk) && !self.cyclic(rte) && ck.safe() && ce.safe() {
			return noop
		}
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				iter := rv.MapRange()
				rvk := reflect.New(rtk).Elem()
				rve := reflect.New(rte).Elem()
				for iter.Next() {
					rvk.SetIterKey(iter)
					if err := ck.check(rvk, depth-1); err != nil {
						return err
					}
					rve.SetIterValue(iter)
					if err := ce.check(rve, depth-1); err != nil {
						return err
					}
				}
				return nil
			},
		}

	case reflect.Pointer:
		ce := cache.getCheckerInternal(rt.Elem(), self)
		if !self.cyclic(rt.Elem()) && ce.safe() {
			return noop
		}
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if rv.IsNil() {
					return nil
				}
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				return ce.check(rv.Elem(), depth-1)
			},
		}

	case reflect.Slice:
		ce := cache.getCheckerInternal(rt.Elem(), self)
		if !self.cyclic(rt.Elem()) && ce.safe() {
			return noop
		}
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				for i := 0; i < rv.Len(); i++ {
					if err := ce.check(rv.Index(i), depth-1); err != nil {
						return err
					}
				}
				return nil
			},
		}

	case reflect.Struct:
		type field struct {
			Index   int
			Checker checker
		}
		var fields []field
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			cf := cache.getCheckerInternal(f.Type, self)
			if !self.cyclic(f.Type) && cf.safe() {
				continue
			}
			fields = append(fields, field{Index: i, Checker: cf})
		}
		if len(fields) == 0 {
			return noop
		}
		return checker{
			safe: func() bool {
				return false
			},
			check: func(rv reflect.Value, depth int) error {
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				for _, fi := range fields {
					if err := fi.Checker.check(rv.Field(fi.Index), depth-1); err != nil {
						return err
					}
				}
				return nil
			},
		}

	default:
		// Not a serializable composite type (funcs and channels are composite types but are
		// rejected by JSON and CBOR serialization).
		return noop

	}
}

func checkUnstructuredValue(cache *checkers, v interface{}, depth int) error {
	switch v := v.(type) {
	case nil, bool, int64, float64, string:
		return nil
	case []interface{}:
		if depth <= 0 {
			return errMaxDepthExceeded
		}
		for _, element := range v {
			if err := checkUnstructuredValue(cache, element, depth-1); err != nil {
				return err
			}
		}
		return nil
	case map[string]interface{}:
		if depth <= 0 {
			return errMaxDepthExceeded
		}
		for _, element := range v {
			if err := checkUnstructuredValue(cache, element, depth-1); err != nil {
				return err
			}
		}
		return nil
	default:
		// Unmarshaling an unstructured doesn't use other dynamic types, but nothing
		// prevents inserting values with arbitrary dynamic types into unstructured content,
		// as long as they can be marshalled.
		rv := reflect.ValueOf(v)
		return cache.getChecker(rv.Type()).check(rv, depth)
	}
}
