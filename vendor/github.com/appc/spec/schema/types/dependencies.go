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
)

type Dependencies []Dependency

type Dependency struct {
	ImageName ACIdentifier `json:"imageName"`
	ImageID   *Hash        `json:"imageID,omitempty"`
	Labels    Labels       `json:"labels,omitempty"`
	Size      uint         `json:"size,omitempty"`
}

type dependency Dependency

func (d Dependency) assertValid() error {
	if len(d.ImageName) < 1 {
		return errors.New(`imageName cannot be empty`)
	}
	return nil
}

func (d Dependency) MarshalJSON() ([]byte, error) {
	if err := d.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(dependency(d))
}

func (d *Dependency) UnmarshalJSON(data []byte) error {
	var jd dependency
	if err := json.Unmarshal(data, &jd); err != nil {
		return err
	}
	nd := Dependency(jd)
	if err := nd.assertValid(); err != nil {
		return err
	}
	*d = nd
	return nil
}
