// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import "fmt"

// multiFlag allows repeated string flags to be collected into a slice
type multiFlag []string

func (m *multiFlag) String() string {
	if len(*m) == 0 {
		return ""
	}
	return fmt.Sprint(*m)
}

func (m *multiFlag) Set(v string) error {
	(*m) = append(*m, v)
	return nil
}

// explicitFlag is a string flag that tracks whether it was set.
type explicitFlag struct {
	set   bool
	value string
}

func (f *explicitFlag) Set(value string) error {
	f.set = true
	f.value = value
	return nil
}

func (f *explicitFlag) String() string {
	if f == nil {
		return ""
	}
	return f.value
}
