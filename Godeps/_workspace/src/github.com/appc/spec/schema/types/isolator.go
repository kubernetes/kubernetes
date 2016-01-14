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

// Isolators encapsulates a list of individual Isolators for the ImageManifest
// and PodManifest schemas.
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

// IsolatorValue encapsulates the actual value of an Isolator which may be
// serialized as any arbitrary JSON blob. Specific Isolator types should
// implement this interface to facilitate unmarshalling and validation.
type IsolatorValue interface {
	UnmarshalJSON(b []byte) error
	AssertValid() error
}

// Isolator is a model for unmarshalling isolator types from their JSON-encoded
// representation.
type Isolator struct {
	// Name is the name of the Isolator type as defined in the specification.
	Name ACIdentifier `json:"name"`
	// ValueRaw captures the raw JSON value of an Isolator that was
	// unmarshalled. This field is used for unmarshalling only. It MUST NOT
	// be referenced by external users of the Isolator struct. It is
	// exported only to satisfy Go's unfortunate requirement that fields
	// must be capitalized to be unmarshalled successfully.
	ValueRaw *json.RawMessage `json:"value"`
	// value captures the "true" value of the isolator.
	value IsolatorValue
}

// isolator is a shadow type used for unmarshalling.
type isolator Isolator

// Value returns the raw Value of this Isolator. Users should perform a type
// switch/assertion on this value to extract the underlying isolator type.
func (i *Isolator) Value() IsolatorValue {
	return i.value
}

// UnmarshalJSON populates this Isolator from a JSON-encoded representation. To
// unmarshal the Value of the Isolator, it will use the appropriate constructor
// as registered by AddIsolatorValueConstructor.
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
