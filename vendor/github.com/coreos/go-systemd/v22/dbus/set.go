// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dbus

type set struct {
	data map[string]bool
}

func (s *set) Add(value string) {
	s.data[value] = true
}

func (s *set) Remove(value string) {
	delete(s.data, value)
}

func (s *set) Contains(value string) (exists bool) {
	_, exists = s.data[value]
	return
}

func (s *set) Length() int {
	return len(s.data)
}

func (s *set) Values() (values []string) {
	for val := range s.data {
		values = append(values, val)
	}
	return
}

func newSet() *set {
	return &set{make(map[string]bool)}
}
