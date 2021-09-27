// +build !go1.12

package fmtsort

import "reflect"

const brokenNaNs = true

func mapElems(mapValue reflect.Value) ([]reflect.Value, []reflect.Value) {
	key := mapValue.MapKeys()
	value := make([]reflect.Value, 0, len(key))
	for _, k := range key {
		v := mapValue.MapIndex(k)
		if !v.IsValid() {
			// Note: we can't retrieve the value, probably because
			// the key is NaN, so just do the best we can and
			// add a zero value of the correct type in that case.
			v = reflect.Zero(mapValue.Type().Elem())
		}
		value = append(value, v)
	}
	return key, value
}
