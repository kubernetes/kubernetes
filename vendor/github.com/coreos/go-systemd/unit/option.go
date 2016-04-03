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

package unit

import (
	"fmt"
)

type UnitOption struct {
	Section string
	Name    string
	Value   string
}

func NewUnitOption(section, name, value string) *UnitOption {
	return &UnitOption{Section: section, Name: name, Value: value}
}

func (uo *UnitOption) String() string {
	return fmt.Sprintf("{Section: %q, Name: %q, Value: %q}", uo.Section, uo.Name, uo.Value)
}

func (uo *UnitOption) Match(other *UnitOption) bool {
	return uo.Section == other.Section &&
		uo.Name == other.Name &&
		uo.Value == other.Value
}

func AllMatch(u1 []*UnitOption, u2 []*UnitOption) bool {
	length := len(u1)
	if length != len(u2) {
		return false
	}

	for i := 0; i < length; i++ {
		if !u1[i].Match(u2[i]) {
			return false
		}
	}

	return true
}
