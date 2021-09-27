// +build go1.12

package fmtsort

import "reflect"

const brokenNaNs = false

func mapElems(mapValue reflect.Value) ([]reflect.Value, []reflect.Value) {
	// Note: this code is arranged to not panic even in the presence
	// of a concurrent map update. The runtime is responsible for
	// yelling loudly if that happens. See issue 33275.
	n := mapValue.Len()
	key := make([]reflect.Value, 0, n)
	value := make([]reflect.Value, 0, n)
	iter := mapValue.MapRange()
	for iter.Next() {
		key = append(key, iter.Key())
		value = append(value, iter.Value())
	}
	return key, value
}
