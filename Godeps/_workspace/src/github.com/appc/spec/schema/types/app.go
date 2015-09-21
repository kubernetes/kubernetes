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
	"path"
)

type App struct {
	Exec             Exec           `json:"exec"`
	EventHandlers    []EventHandler `json:"eventHandlers,omitempty"`
	User             string         `json:"user"`
	Group            string         `json:"group"`
	WorkingDirectory string         `json:"workingDirectory,omitempty"`
	Environment      Environment    `json:"environment,omitempty"`
	MountPoints      []MountPoint   `json:"mountPoints,omitempty"`
	Ports            []Port         `json:"ports,omitempty"`
	Isolators        Isolators      `json:"isolators,omitempty"`
}

// app is a model to facilitate extra validation during the
// unmarshalling of the App
type app App

func (a *App) UnmarshalJSON(data []byte) error {
	ja := app(*a)
	err := json.Unmarshal(data, &ja)
	if err != nil {
		return err
	}
	na := App(ja)
	if err := na.assertValid(); err != nil {
		return err
	}
	if na.Environment == nil {
		na.Environment = make(Environment, 0)
	}
	*a = na
	return nil
}

func (a App) MarshalJSON() ([]byte, error) {
	if err := a.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(app(a))
}

func (a *App) assertValid() error {
	if err := a.Exec.assertValid(); err != nil {
		return err
	}
	if a.User == "" {
		return errors.New(`User is required`)
	}
	if a.Group == "" {
		return errors.New(`Group is required`)
	}
	if !path.IsAbs(a.WorkingDirectory) && a.WorkingDirectory != "" {
		return errors.New("WorkingDirectory must be an absolute path")
	}
	eh := make(map[string]bool)
	for _, e := range a.EventHandlers {
		name := e.Name
		if eh[name] {
			return fmt.Errorf("Only one eventHandler of name %q allowed", name)
		}
		eh[name] = true
	}
	if err := a.Environment.assertValid(); err != nil {
		return err
	}
	return nil
}
