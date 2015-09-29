// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
)

var (
	isolatorMap map[ACIdentifier]IsolatorValueConstructor
)

func init() {
	isolatorMap = make(map[ACIdentifier]IsolatorValueConstructor)
}

type IsolatorValueConstructor func() IsolatorValue

func AddIsolatorValueConstructor(n ACIdentifier, i IsolatorValueConstructor) {
	isolatorMap[n] = i
}

func AddIsolatorName(n ACIdentifier, ns map[ACIdentifier]struct{}) {
	ns[n] = struct{}{}
}

type Isolators []Isolator

// GetByName returns the last isolator in the list by the given name.
func (is *Isolators) GetByName(name ACIdentifier) *Isolator {
	var i Isolator
	for j := len(*is) - 1; j >= 0; j-- {
		i = []Isolator(*is)[j]
		if i.Name == name {
			return &i
		}
	}
	return nil
}

// Unrecognized returns a set of isolators that are not recognized.
// An isolator is not recognized if it has not had an associated
// constructor registered with AddIsolatorValueConstructor.
func (is *Isolators) Unrecognized() Isolators {
	u := Isolators{}
	for _, i := range *is {
		if i.value == nil {
			u = append(u, i)
		}
	}
	return u
}

type IsolatorValue interface {
	UnmarshalJSON(b []byte) error
	AssertValid() error
}
type Isolator struct {
	Name     ACIdentifier     `json:"name"`
	ValueRaw *json.RawMessage `json:"value"`
	value    IsolatorValue
}
type isolator Isolator

func (i *Isolator) Value() IsolatorValue {
	return i.value
}

func (i *Isolator) UnmarshalJSON(b []byte) error {
	var ii isolator
	err := json.Unmarshal(b, &ii)
	if err != nil {
		return err
	}

	var dst IsolatorValue
	con, ok := isolatorMap[ii.Name]
	if ok {
		dst = con()
		err = dst.UnmarshalJSON(*ii.ValueRaw)
		if err != nil {
			return err
		}
		err = dst.AssertValid()
		if err != nil {
			return err
		}
	}

	i.value = dst
	i.ValueRaw = ii.ValueRaw
	i.Name = ii.Name

	return nil
}
