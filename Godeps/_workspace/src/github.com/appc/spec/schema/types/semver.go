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

	"github.com/coreos/go-semver/semver"
)

var (
	ErrNoZeroSemVer = ACVersionError("SemVer cannot be zero")
	ErrBadSemVer    = ACVersionError("SemVer is bad")
)

// SemVer implements the Unmarshaler interface to define a field that must be
// a semantic version string
// TODO(jonboulle): extend upstream instead of wrapping?
type SemVer semver.Version

// NewSemVer generates a new SemVer from a string. If the given string does
// not represent a valid SemVer, nil and an error are returned.
func NewSemVer(s string) (*SemVer, error) {
	nsv, err := semver.NewVersion(s)
	if err != nil {
		return nil, ErrBadSemVer
	}
	v := SemVer(*nsv)
	if v.Empty() {
		return nil, ErrNoZeroSemVer
	}
	return &v, nil
}

func (sv SemVer) String() string {
	s := semver.Version(sv)
	return s.String()
}

func (sv SemVer) Empty() bool {
	return semver.Version(sv) == semver.Version{}
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (sv *SemVer) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	v, err := NewSemVer(s)
	if err != nil {
		return err
	}
	*sv = *v
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (sv SemVer) MarshalJSON() ([]byte, error) {
	if sv.Empty() {
		return nil, ErrNoZeroSemVer
	}
	return json.Marshal(sv.String())
}
