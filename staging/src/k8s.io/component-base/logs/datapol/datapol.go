/*
Copyright 2020 The Kubernetes Authors.

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

// Package datapol contains functions to determine if objects contain sensitive
// data to e.g. make decisions on whether to log them or not.
package datapol

import (
	"reflect"
	"strings"

	"k8s.io/klog/v2"
)

// Verify returns a list of the datatypes contained in the argument that can be
// considered sensitive w.r.t. to logging
func Verify(value interface{}) []string {
	defer func() {
		if r := recover(); r != nil {
			//TODO maybe export a metric
			klog.Warningf("Error while inspecting arguments for sensitive data: %v", r)
		}
	}()
	t := reflect.ValueOf(value)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return datatypes(t)
}

func datatypes(v reflect.Value) []string {
	if types := byType(v.Type()); len(types) > 0 {
		// Slices, and maps can be nil or empty, only the nil case is zero
		switch v.Kind() {
		case reflect.Slice, reflect.Map:
			if !v.IsZero() && v.Len() > 0 {
				return types
			}
		default:
			if !v.IsZero() {
				return types
			}
		}
	}
	switch v.Kind() {
	case reflect.Interface:
		return datatypes(v.Elem())
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if types := datatypes(v.Index(i)); len(types) > 0 {
				return types
			}
		}
	case reflect.Map:
		mapIter := v.MapRange()
		for mapIter.Next() {
			k := mapIter.Key()
			v := mapIter.Value()
			if types := datatypes(k); len(types) > 0 {
				return types
			}
			if types := datatypes(v); len(types) > 0 {
				return types
			}
		}
	case reflect.Struct:
		t := v.Type()
		numField := t.NumField()

		for i := 0; i < numField; i++ {
			f := t.Field(i)
			if f.Type.Kind() == reflect.Pointer {
				continue
			}
			if reason, ok := f.Tag.Lookup("datapolicy"); ok {
				if !v.Field(i).IsZero() {
					return strings.Split(reason, ",")
				}
			}
			if types := datatypes(v.Field(i)); len(types) > 0 {
				return types
			}
		}
	}
	return nil
}
