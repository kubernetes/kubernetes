/*
Copyright 2023 The Kubernetes Authors.

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

package execution

import (
	"io"
	"os"
)

// New returns a new Vars with the given options applied.
func New(options ...Option) *Vars {
	ex := new(Vars)
	for _, option := range options {
		option(ex)
	}
	ex.fillDefaults()
	return ex
}

// Option is a functional option for Vars.
type Option func(v *Vars)

// Vars is the execution context for the code-generator.
type Vars struct {
	Out  io.Writer
	Args []string
	Exit func(int)
}

func (v *Vars) fillDefaults() {
	if v.Out == nil {
		v.Out = os.Stderr
	}
	if v.Exit == nil {
		v.Exit = os.Exit
	}
	if v.Args == nil {
		v.Args = os.Args[1:]
	}
}
