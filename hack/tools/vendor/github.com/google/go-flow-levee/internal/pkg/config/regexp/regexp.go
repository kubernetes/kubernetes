// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package regexp contains functionality for unmarshalling regular expressions from a config.
package regexp

import (
	"encoding/json"
	"regexp"
)

// Regexp delegates to a Regexp while enabling unmarshalling.
// Any unspecified / nil matcher will return vacuous truth in MatchString
type Regexp struct {
	r *regexp.Regexp
}

func New(s string) (*Regexp, error) {
	r, err := regexp.Compile(s)
	return &Regexp{r: r}, err
}

// MatchString delegates matching to the regex package.
func (mr *Regexp) MatchString(s string) bool {
	return mr.r == nil || mr.r.MatchString(s)
}

// UnmarshalJSON implementation of json.UnmarshalJSON interface.
func (mr *Regexp) UnmarshalJSON(data []byte) error {
	var matcher string
	if err := json.Unmarshal(data, &matcher); err != nil {
		return err
	}

	var err error
	if mr.r, err = regexp.Compile(matcher); err != nil {
		return err
	}
	return nil
}
