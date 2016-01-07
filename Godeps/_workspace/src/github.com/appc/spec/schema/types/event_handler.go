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
	"fmt"
)

type EventHandler struct {
	Name string `json:"name"`
	Exec Exec   `json:"exec"`
}

type eventHandler EventHandler

func (e EventHandler) assertValid() error {
	s := e.Name
	switch s {
	case "pre-start", "post-stop":
		return nil
	case "":
		return errors.New(`eventHandler "name" cannot be empty`)
	default:
		return fmt.Errorf(`bad eventHandler "name": %q`, s)
	}
}

func (e EventHandler) MarshalJSON() ([]byte, error) {
	if err := e.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(eventHandler(e))
}

func (e *EventHandler) UnmarshalJSON(data []byte) error {
	var je eventHandler
	err := json.Unmarshal(data, &je)
	if err != nil {
		return err
	}
	ne := EventHandler(je)
	if err := ne.assertValid(); err != nil {
		return err
	}
	*e = ne
	return nil
}
