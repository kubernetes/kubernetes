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
	"errors"
	"fmt"
)

var (
	isolatorMap map[ACIdentifier]IsolatorValueConstructor

	// ErrIncompatibleIsolator is returned whenever an Isolators set contains
	// conflicting IsolatorValue instances
	ErrIncompatibleIsolator = errors.New("isolators set contains incompatible types")
	// ErrInvalidIsolator is returned upon validation failures due to improper
	// or partially constructed Isolator instances (eg. from incomplete direct construction)
	ErrInvalidIsolator = errors.New("invalid isolator")
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

// assertValid checks that every single isolator is valid and that
// the whole set is well built
func (isolators Isolators) assertValid() error {
	typesMap := make(map[ACIdentifier]bool)
	for _, i := range isolators {
		v := i.Value()
		if v == nil {
			return ErrInvalidIsolator
		}
		if err := v.AssertValid(); err != nil {
			return err
		}
		if _, ok := typesMap[i.Name]; ok {
			if !v.multipleAllowed() {
				return fmt.Errorf(`isolators set contains too many instances of type %s"`, i.Name)
			}
		}
		for _, c := range v.Conflicts() {
			if _, found := typesMap[c]; found {
				return ErrIncompatibleIsolator
			}
		}
		typesMap[i.Name] = true
	}
	return nil
}

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

// ReplaceIsolatorsByName overrides matching isolator types with a new
// isolator, deleting them all and appending the new one instead
func (is *Isolators) ReplaceIsolatorsByName(newIs Isolator, oldNames []ACIdentifier) {
	var i Isolator
	for j := len(*is) - 1; j >= 0; j-- {
		i = []Isolator(*is)[j]
		for _, name := range oldNames {
			if i.Name == name {
				*is = append((*is)[:j], (*is)[j+1:]...)
			}
		}
	}
	*is = append((*is)[:], newIs)
	return
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
	// UnmarshalJSON unserialize a JSON-encoded isolator
	UnmarshalJSON(b []byte) error
	// AssertValid returns a non-nil error value if an IsolatorValue is not valid
	// according to appc spec
	AssertValid() error
	// Conflicts returns a list of conflicting isolators types, which cannot co-exist
	// together with this IsolatorValue
	Conflicts() []ACIdentifier
	// multipleAllowed specifies whether multiple isolator instances are allowed
	// for this isolator type
	multipleAllowed() bool
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
