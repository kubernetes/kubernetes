/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"strings"
)

// StringSliceFlag is a string slice flag compatible with flags and pflags that keeps track of whether it had a value supplied or not.
type StringSliceFlag struct {
	// If Set has been invoked this value is true
	provided bool
	// The exact value provided on the flag
	value []string
}

func NewStringSliceFlag(defaultVal []string) StringSliceFlag {
	return StringSliceFlag{value: defaultVal}
}

func (f *StringSliceFlag) Default(value []string) {
	f.value = value
}

func (f StringSliceFlag) String() string {
	return "[" + strings.Join(f.value, ",") + "]"
}

func (f StringSliceFlag) Value() []string {
	return f.value
}

func (f *StringSliceFlag) Set(value string) error {
	v := strings.Split(value, ",")
	if !f.provided {
		f.value = v
	} else {
		f.value = append(f.value, v...)
	}
	f.provided = true
	return nil
}

func (f StringSliceFlag) Provided() bool {
	return f.provided
}

func (f *StringSliceFlag) Type() string {
	return "stringSlice"
}
