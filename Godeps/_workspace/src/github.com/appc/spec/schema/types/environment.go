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
	"regexp"
)

var (
	envPattern = regexp.MustCompile("^[A-Za-z_][A-Za-z_0-9]*$")
)

type Environment []EnvironmentVariable

type environment Environment

type EnvironmentVariable struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

func (ev EnvironmentVariable) assertValid() error {
	if len(ev.Name) == 0 {
		return fmt.Errorf(`environment variable name must not be empty`)
	}
	if !envPattern.MatchString(ev.Name) {
		return fmt.Errorf(`environment variable does not have valid identifier %q`, ev.Name)
	}
	return nil
}

func (e Environment) assertValid() error {
	seen := map[string]bool{}
	for _, env := range e {
		if err := env.assertValid(); err != nil {
			return err
		}
		_, ok := seen[env.Name]
		if ok {
			return fmt.Errorf(`duplicate environment variable of name %q`, env.Name)
		}
		seen[env.Name] = true
	}

	return nil
}

func (e Environment) MarshalJSON() ([]byte, error) {
	if err := e.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(environment(e))
}

func (e *Environment) UnmarshalJSON(data []byte) error {
	var je environment
	if err := json.Unmarshal(data, &je); err != nil {
		return err
	}
	ne := Environment(je)
	if err := ne.assertValid(); err != nil {
		return err
	}
	*e = ne
	return nil
}

// Retrieve the value of an environment variable by the given name from
// Environment, if it exists.
func (e Environment) Get(name string) (value string, ok bool) {
	for _, env := range e {
		if env.Name == name {
			return env.Value, true
		}
	}
	return "", false
}

// Set sets the value of an environment variable by the given name,
// overwriting if one already exists.
func (e *Environment) Set(name string, value string) {
	for i, env := range *e {
		if env.Name == name {
			(*e)[i] = EnvironmentVariable{
				Name:  name,
				Value: value,
			}
			return
		}
	}
	env := EnvironmentVariable{
		Name:  name,
		Value: value,
	}
	*e = append(*e, env)
}
