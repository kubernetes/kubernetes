/*
Copyright 2017 The Kubernetes Authors.

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

package core

import "encoding/json"

// This file implements json marshaling/unmarshaling interfaces on objects that are currently marshaled into annotations
// to prevent anyone from marshaling these internal structs.

var _ = json.Marshaler(&AvoidPods{})
var _ = json.Unmarshaler(&AvoidPods{})

// MarshalJSON panics to prevent marshalling of internal structs
func (AvoidPods) MarshalJSON() ([]byte, error) { panic("do not marshal internal struct") }

// UnmarshalJSON panics to prevent unmarshalling of internal structs
func (*AvoidPods) UnmarshalJSON([]byte) error { panic("do not unmarshal to internal struct") }
