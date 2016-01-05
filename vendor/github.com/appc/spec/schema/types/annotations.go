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
	"fmt"
)

type Annotations []Annotation

type annotations Annotations

type Annotation struct {
	Name  ACIdentifier `json:"name"`
	Value string       `json:"value"`
}

func (a Annotations) assertValid() error {
	seen := map[ACIdentifier]string{}
	for _, anno := range a {
		_, ok := seen[anno.Name]
		if ok {
			return fmt.Errorf(`duplicate annotations of name %q`, anno.Name)
		}
		seen[anno.Name] = anno.Value
	}
	if c, ok := seen["created"]; ok {
		if _, err := NewDate(c); err != nil {
			return err
		}
	}
	if h, ok := seen["homepage"]; ok {
		if _, err := NewURL(h); err != nil {
			return err
		}
	}
	if d, ok := seen["documentation"]; ok {
		if _, err := NewURL(d); err != nil {
			return err
		}
	}

	return nil
}

func (a Annotations) MarshalJSON() ([]byte, error) {
	if err := a.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(annotations(a))
}

func (a *Annotations) UnmarshalJSON(data []byte) error {
	var ja annotations
	if err := json.Unmarshal(data, &ja); err != nil {
		return err
	}
	na := Annotations(ja)
	if err := na.assertValid(); err != nil {
		return err
	}
	*a = na
	return nil
}

// Retrieve the value of an annotation by the given name from Annotations, if
// it exists.
func (a Annotations) Get(name string) (val string, ok bool) {
	for _, anno := range a {
		if anno.Name.String() == name {
			return anno.Value, true
		}
	}
	return "", false
}

// Set sets the value of an annotation by the given name, overwriting if one already exists.
func (a *Annotations) Set(name ACIdentifier, value string) {
	for i, anno := range *a {
		if anno.Name.Equals(name) {
			(*a)[i] = Annotation{
				Name:  name,
				Value: value,
			}
			return
		}
	}
	anno := Annotation{
		Name:  name,
		Value: value,
	}
	*a = append(*a, anno)
}
