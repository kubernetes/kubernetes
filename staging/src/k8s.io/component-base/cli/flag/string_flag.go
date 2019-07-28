/*
Copyright 2014 The Kubernetes Authors.

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

package flag

// StringFlag is a string flag compatible with flags and pflags that keeps track of whether it had a value supplied or not.
type StringFlag struct {
	// If Set has been invoked this value is true
	provided bool
	// The exact value provided on the flag
	value string
}

func NewStringFlag(defaultVal string) StringFlag {
	return StringFlag{value: defaultVal}
}

func (f *StringFlag) Default(value string) {
	f.value = value
}

func (f StringFlag) String() string {
	return f.value
}

func (f StringFlag) Value() string {
	return f.value
}

func (f *StringFlag) Set(value string) error {
	f.value = value
	f.provided = true

	return nil
}

func (f StringFlag) Provided() bool {
	return f.provided
}

func (f *StringFlag) Type() string {
	return "string"
}
