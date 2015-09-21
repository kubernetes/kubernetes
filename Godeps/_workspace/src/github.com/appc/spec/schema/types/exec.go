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
	"path/filepath"
)

type Exec []string

type exec Exec

func (e Exec) assertValid() error {
	if len(e) < 1 {
		return errors.New(`Exec cannot be empty`)
	}
	if !filepath.IsAbs(e[0]) {
		return errors.New(`Exec[0] must be absolute path`)
	}
	return nil
}

func (e Exec) MarshalJSON() ([]byte, error) {
	if err := e.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(exec(e))
}

func (e *Exec) UnmarshalJSON(data []byte) error {
	var je exec
	err := json.Unmarshal(data, &je)
	if err != nil {
		return err
	}
	ne := Exec(je)
	if err := ne.assertValid(); err != nil {
		return err
	}
	*e = ne
	return nil
}
