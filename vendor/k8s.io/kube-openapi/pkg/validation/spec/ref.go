// Copyright 2015 go-swagger maintainers
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

package spec

import (
	"encoding/json"
	"github.com/go-openapi/jsonreference"

	"k8s.io/kube-openapi/pkg/internal"
)

// Refable is a struct for things that accept a $ref property
type Refable struct {
	Ref Ref
}

// MarshalJSON marshals the ref to json
func (r Refable) MarshalJSON() ([]byte, error) {
	return r.Ref.MarshalJSON()
}

// UnmarshalJSON unmarshalss the ref from json
func (r *Refable) UnmarshalJSON(d []byte) error {
	return json.Unmarshal(d, &r.Ref)
}

// Ref represents a json reference that is potentially resolved
type Ref struct {
	jsonreference.Ref
}

// RemoteURI gets the remote uri part of the ref
func (r *Ref) RemoteURI() string {
	if r.String() == "" {
		return r.String()
	}

	u := *r.GetURL()
	u.Fragment = ""
	return u.String()
}

// Inherits creates a new reference from a parent and a child
// If the child cannot inherit from the parent, an error is returned
func (r *Ref) Inherits(child Ref) (*Ref, error) {
	ref, err := r.Ref.Inherits(child.Ref)
	if err != nil {
		return nil, err
	}
	return &Ref{Ref: *ref}, nil
}

// NewRef creates a new instance of a ref object
// returns an error when the reference uri is an invalid uri
func NewRef(refURI string) (Ref, error) {
	ref, err := jsonreference.New(refURI)
	if err != nil {
		return Ref{}, err
	}
	return Ref{Ref: ref}, nil
}

// MustCreateRef creates a ref object but panics when refURI is invalid.
// Use the NewRef method for a version that returns an error.
func MustCreateRef(refURI string) Ref {
	return Ref{Ref: jsonreference.MustCreateRef(refURI)}
}

// MarshalJSON marshals this ref into a JSON object
func (r Ref) MarshalJSON() ([]byte, error) {
	str := r.String()
	if str == "" {
		if r.IsRoot() {
			return []byte(`{"$ref":""}`), nil
		}
		return []byte("{}"), nil
	}
	v := map[string]interface{}{"$ref": str}
	return json.Marshal(v)
}

// UnmarshalJSON unmarshals this ref from a JSON object
func (r *Ref) UnmarshalJSON(d []byte) error {
	var v map[string]interface{}
	if err := json.Unmarshal(d, &v); err != nil {
		return err
	}
	return r.fromMap(v)
}

func (r *Ref) fromMap(v map[string]interface{}) error {
	return internal.JSONRefFromMap(&r.Ref, v)
}
