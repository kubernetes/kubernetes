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
	"net/url"
)

// URL wraps url.URL to marshal/unmarshal to/from JSON strings and enforce
// that the scheme is HTTP/HTTPS only
type URL url.URL

func NewURL(s string) (*URL, error) {
	uu, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("bad URL: %v", err)
	}
	nu := URL(*uu)
	if err := nu.assertValidScheme(); err != nil {
		return nil, err
	}
	return &nu, nil
}

func (u URL) String() string {
	uu := url.URL(u)
	return uu.String()
}

func (u URL) assertValidScheme() error {
	switch u.Scheme {
	case "http", "https":
		return nil
	default:
		return fmt.Errorf("bad URL scheme, must be http/https")
	}
}

func (u *URL) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	nu, err := NewURL(s)
	if err != nil {
		return err
	}
	*u = *nu
	return nil
}

func (u URL) MarshalJSON() ([]byte, error) {
	if err := u.assertValidScheme(); err != nil {
		return nil, err
	}
	return json.Marshal(u.String())
}
