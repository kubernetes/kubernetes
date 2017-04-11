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
	"net/http"
	"os"
	"path/filepath"

	"github.com/go-openapi/jsonreference"
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

// IsValidURI returns true when the url the ref points to can be found
func (r *Ref) IsValidURI() bool {
	if r.String() == "" {
		return true
	}

	v := r.RemoteURI()
	if v == "" {
		return true
	}

	if r.HasFullURL {
		rr, err := http.Get(v)
		if err != nil {
			return false
		}

		return rr.StatusCode/100 == 2
	}

	if !(r.HasFileScheme || r.HasFullFilePath || r.HasURLPathOnly) {
		return false
	}

	// check for local file
	pth := v
	if r.HasURLPathOnly {
		p, e := filepath.Abs(pth)
		if e != nil {
			return false
		}
		pth = p
	}

	fi, err := os.Stat(pth)
	if err != nil {
		return false
	}

	return !fi.IsDir()
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

// MustCreateRef creates a ref object but
func MustCreateRef(refURI string) Ref {
	return Ref{Ref: jsonreference.MustCreateRef(refURI)}
}

// // NewResolvedRef creates a resolved ref
// func NewResolvedRef(refURI string, data interface{}) Ref {
// 	return Ref{
// 		Ref:      jsonreference.MustCreateRef(refURI),
// 		Resolved: data,
// 	}
// }

// MarshalJSON marshals this ref into a JSON object
func (r Ref) MarshalJSON() ([]byte, error) {
	str := r.String()
	if str == "" {
		if r.IsRoot() {
			return []byte(`{"$ref":"#"}`), nil
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

	if v == nil {
		return nil
	}

	if vv, ok := v["$ref"]; ok {
		if str, ok := vv.(string); ok {
			ref, err := jsonreference.New(str)
			if err != nil {
				return err
			}
			*r = Ref{Ref: ref}
		}
	}

	return nil
}
