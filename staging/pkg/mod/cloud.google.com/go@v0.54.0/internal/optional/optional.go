// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package optional provides versions of primitive types that can
// be nil. These are useful in methods that update some of an API object's
// fields.
package optional

import (
	"fmt"
	"strings"
	"time"
)

type (
	// Bool is either a bool or nil.
	Bool interface{}

	// String is either a string or nil.
	String interface{}

	// Int is either an int or nil.
	Int interface{}

	// Uint is either a uint or nil.
	Uint interface{}

	// Float64 is either a float64 or nil.
	Float64 interface{}

	// Duration is either a time.Duration or nil.
	Duration interface{}
)

// ToBool returns its argument as a bool.
// It panics if its argument is nil or not a bool.
func ToBool(v Bool) bool {
	x, ok := v.(bool)
	if !ok {
		doPanic("Bool", v)
	}
	return x
}

// ToString returns its argument as a string.
// It panics if its argument is nil or not a string.
func ToString(v String) string {
	x, ok := v.(string)
	if !ok {
		doPanic("String", v)
	}
	return x
}

// ToInt returns its argument as an int.
// It panics if its argument is nil or not an int.
func ToInt(v Int) int {
	x, ok := v.(int)
	if !ok {
		doPanic("Int", v)
	}
	return x
}

// ToUint returns its argument as a uint.
// It panics if its argument is nil or not a uint.
func ToUint(v Uint) uint {
	x, ok := v.(uint)
	if !ok {
		doPanic("Uint", v)
	}
	return x
}

// ToFloat64 returns its argument as a float64.
// It panics if its argument is nil or not a float64.
func ToFloat64(v Float64) float64 {
	x, ok := v.(float64)
	if !ok {
		doPanic("Float64", v)
	}
	return x
}

// ToDuration returns its argument as a time.Duration.
// It panics if its argument is nil or not a time.Duration.
func ToDuration(v Duration) time.Duration {
	x, ok := v.(time.Duration)
	if !ok {
		doPanic("Duration", v)
	}
	return x
}

func doPanic(capType string, v interface{}) {
	panic(fmt.Sprintf("optional.%s value should be %s, got %T", capType, strings.ToLower(capType), v))
}
