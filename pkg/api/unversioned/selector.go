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

package unversioned

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

// FieldSelector is a wrapper around fields.Selector that allows for
// Marshaling/Unmarshaling underlying selector.
type FieldSelector struct {
	Selector fields.Selector
}

func (sh FieldSelector) MarshalJSON() ([]byte, error) {
	return json.Marshal(sh.Selector.String())
}

func (sh *FieldSelector) UnmarshalJSON(b []byte) error {
	var unmarshalled string
	if err := json.Unmarshal(b, &unmarshalled); err != nil {
		return err
	}
	selector, err := fields.ParseSelector(unmarshalled)
	if err != nil {
		return err
	}
	sh.Selector = selector
	return nil
}

// LabelSelector is a wrapper around labels.Selector that allow for
// Marshaling/Unmarshaling underlying selector.
type LabelSelector struct {
	Selector labels.Selector
}

func (sh LabelSelector) MarshalJSON() ([]byte, error) {
	return json.Marshal(sh.Selector.String())
}

func (sh *LabelSelector) UnmarshalJSON(b []byte) error {
	var unmarshalled string
	if err := json.Unmarshal(b, &unmarshalled); err != nil {
		return err
	}
	selector, err := labels.Parse(unmarshalled)
	if err != nil {
		return err
	}
	sh.Selector = selector
	return nil
}
