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
// cbor.Marshaler and likewise for the respective Unmarshaler interfaces. This will be removed in
// favor of automatic transcoding between CBOR and JSON/text for these types.
func CheckUnsupportedMarshalers(v interface{}) error {
	return checkInternal(v, 128)
}

// Recursion depth is limited as a basic mitigation against cyclic objects. Objects created by the
// decoder shouldn't be able to contain cycles, but practically any object can be passed to the
// encoder.
var errMaxDepthExceeded = errors.New("object depth exceeds reasonable limits (possible cycle?)")

var marshalerCache = checkers{
	safeInterface: reflect.TypeFor[cbor.Marshaler](),
	unsafeInterfaces: []reflect.Type{
		reflect.TypeFor[json.Marshaler](),
		reflect.TypeFor[encoding.TextMarshaler](),
	},
}

var unmarshalerCache = checkers{
	safeInterface: reflect.TypeFor[cbor.Unmarshaler](),
	unsafeInterfaces: []reflect.Type{
		reflect.TypeFor[json.Unmarshaler](),
		reflect.TypeFor[encoding.TextUnmarshaler](),
	},
}

func checkInternal(v interface{}, depth int) error {
	if v == nil {
		return nil
	}
	rv := reflect.ValueOf(v)
	if err := marshalerCache.getChecker(rv.Type()).check(rv, depth); err != nil {
		return err
	}
	if err := unmarshalerCache.getChecker(rv.Type()).check(rv, depth); err != nil {
		return err
	}
	return nil
}

type checker struct {
	check func(rv reflect.Value, depth int) error
}

// Having a single addressable checker for comparisons lets us prune parts of the object traversal
// that are statically known to be safe.
var noop = checker{check: func(rv reflect.Value, depth int) error {
	return nil
}}

type checkers struct {
	lock             sync.RWMutex
	m                map[reflect.Type]**checker
	safeInterface    reflect.Type
	unsafeInterfaces []reflect.Type
}

func (cache *checkers) getChecker(rt reflect.Type) checker {
	cache.lock.RLock()
	c, ok := cache.m[rt]
	cache.lock.RUnlock()
	if ok {
		return **c
	}

	cache.lock.Lock()
	defer cache.lock.Unlock()
	return *cache.getCheckerLocked(rt)
}

func (cache *checkers) getCheckerLocked(rt reflect.Type) *checker {
	if c, ok := cache.m[rt]; ok {
		// This type was cached while waiting to acquire the lock.
		return *c
	}

	// Store the cache entry now, before populating it, so that circular references between
	// types don't overflow the call stack.
	c := new(checker)
	if cache.m == nil {
		cache.m = make(map[reflect.Type]**checker)
	}
	cache.m[rt] = &c

	// Take a nonreflective path for the unstructured container types.
	switch rt {
	case reflect.TypeFor[map[string]interface{}](), reflect.TypeFor[[]interface{}]():
		var nonreflect func(v interface{}, depth int) error
		nonreflect = func(v interface{}, depth int) error {
			switch v := v.(type) {
			case nil, bool, int64, float64, string:
				return nil
			case []interface{}:
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				for _, element := range v {
					if err := nonreflect(element, depth-1); err != nil {
						return err
					}
				}
				return nil
			case map[string]interface{}:
				if depth <= 0 {
					return errMaxDepthExceeded
				}
				for _, element := range v {
					if err := nonreflect(element, depth-1); err != nil {
						return err
					}
				}
				return nil
			default:
				rv := reflect.ValueOf(v)
				return cache.getChecker(rv.Type()).check(rv, depth)
			}
		}
		c.check = func(rv reflect.Value, depth int) error {
			return nonreflect(rv.Interface(), depth)
		}
		return c
	}

	if rt.Implements(cache.safeInterface) {
		c = &noop
		return c
	} else {
		for _, risky := range cache.unsafeInterfaces {
			if rt.Implements(risky) {
				err := fmt.Errorf("%v implements %v without corresponding cbor interface", rt, risky)
				c.check = func(reflect.Value, int) error {
					return err
				}
				return c
			}
		}
	}

	// It's possible that pointer-to-type implements one of the relevant interfaces but that a
	// particular value is not addressable. Take this example:
	//
	//   func (Foo) MarshalText() ([]byte, error) { ... }
	//   func (*Foo) MarshalCBOR() ([]byte, error) { ... }
	//
	// Both methods would be usable when marshalling an addressable Foo value, but if
	// Foo is _not_ addressable then only the method with the non-pointer receiver
	// (MarshalText) would be usable.
	//
	// As a simplification, since this check is a temporary measure, it makes
	// conservative assumptions about addressability instead of checking every value: if
	// pointer-to-type implements one of the non-CBOR interfaces, pointer-to-type must
	// also implement the corresponding CBOR interface.
	if reflect.PointerTo(rt).Implements(cache.safeInterface) {
		c = &noop
		return c
	} else {
		for _, risky := range cache.unsafeInterfaces {
			if rt.Implements(risky) {
				err := fmt.Errorf("%v implements %v without corresponding cbor interface", rt, risky)
				c.check = func(reflect.Value, int) error {
					return err
				}
				return c
			}
		}
	}

	switch rt.Kind() {
	case reflect.Array:
		ce := cache.getCheckerLocked(rt.Elem())
		rtlen := rt.Len()
		if rtlen == 0 || ce == &noop {
			c = &noop
			break
		}
		c.check = func(rv reflect.Value, depth int) error {
			if depth <= 0 {
				return errMaxDepthExceeded
			}
			for i := 0; i < rtlen; i++ {
				if err := ce.check(rv.Index(i), depth-1); err != nil {
					return err
				}
			}
			return nil
		}

	case reflect.Interface:
		// All interface values have to be checked because their dynamic type might
		// implement one of the interesting interfaces or be composed of another type that
		// does.
		c.check = func(rv reflect.Value, depth int) error {
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
		}

	case reflect.Map:
		rtk := rt.Key()
		ck := cache.getCheckerLocked(rtk)
		rte := rt.Elem()
		ce := cache.getCheckerLocked(rte)
		if ck == &noop && ce == &noop {
			c = &noop
			break
		}
		c.check = func(rv reflect.Value, depth int) error {
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
		}

	case reflect.Pointer:
		ce := cache.getCheckerLocked(rt.Elem())
		if ce == &noop {
			c = &noop
			break
		}
		c.check = func(rv reflect.Value, depth int) error {
			if rv.IsNil() {
				return nil
			}
			if depth <= 0 {
				return errMaxDepthExceeded
			}
			return ce.check(rv.Elem(), depth-1)
		}

	case reflect.Slice:
		ce := cache.getCheckerLocked(rt.Elem())
		if ce == &noop {
			c = &noop
			break
		}
		c.check = func(rv reflect.Value, depth int) error {
			if depth <= 0 {
				return errMaxDepthExceeded
			}
			for i := 0; i < rv.Len(); i++ {
				if err := ce.check(rv.Index(i), depth-1); err != nil {
					return err
				}
			}
			return nil
		}

	case reflect.Struct:
		type field struct {
			Index   int
			Checker *checker
		}
		var fields []field
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			cf := cache.getCheckerLocked(f.Type)
			if cf == &noop {
				continue
			}
			fields = append(fields, field{Index: i, Checker: cf})
		}
		if len(fields) == 0 {
			c = &noop
			break
		}
		c.check = func(rv reflect.Value, depth int) error {
			if depth <= 0 {
				return errMaxDepthExceeded
			}
			for _, fi := range fields {
				if err := fi.Checker.check(rv.Field(fi.Index), depth-1); err != nil {
					return err
				}
			}
			return nil
		}

	default:
		c = &noop

	}

	return c
}
