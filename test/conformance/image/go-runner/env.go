/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"os"
)

// Getenver is the interface we use to mock out the env for easier testing. OS env
// vars can't be as easily tested since internally it uses sync.Once.
type Getenver interface {
	Getenv(string) string
}

// osEnv uses the actual os.Getenv methods to lookup values.
type osEnv struct{}

// Getenv gets the value of the requested environment variable.
func (*osEnv) Getenv(s string) string {
	return os.Getenv(s)
}

// explicitEnv uses a map instead of os.Getenv methods to lookup values.
type explicitEnv struct {
	vals map[string]string
}

// Getenv returns the value of the requested environment variable (in this
// implementation, really just a map lookup).
func (e *explicitEnv) Getenv(s string) string {
	return e.vals[s]
}

// defaultOSEnv uses a Getenver to lookup values but if it does
// not have that value, it falls back to its internal set of defaults.
type defaultEnver struct {
	firstChoice Getenver
	defaults    map[string]string
}

// Getenv returns the value of the environment variable or its default if unset.
func (e *defaultEnver) Getenv(s string) string {
	v := e.firstChoice.Getenv(s)
	if len(v) == 0 {
		return e.defaults[s]
	}
	return v
}

func envWithDefaults(defaults map[string]string) Getenver {
	return &defaultEnver{firstChoice: &osEnv{}, defaults: defaults}
}
