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

package knftables

import (
	"os/exec"
)

// execer is a mockable wrapper around os/exec.
type execer interface {
	// LookPath wraps exec.LookPath
	LookPath(file string) (string, error)

	// Run runs cmd as with cmd.Output(). If an error occurs, and the process outputs
	// stderr, then that output will be returned in the error.
	Run(cmd *exec.Cmd) (string, error)
}

// realExec implements execer by actually using os/exec
type realExec struct{}

// LookPath is part of execer
func (realExec) LookPath(file string) (string, error) {
	return exec.LookPath(file)
}

// Run is part of execer
func (realExec) Run(cmd *exec.Cmd) (string, error) {
	out, err := cmd.Output()
	if err != nil {
		err = wrapError(err)
	}
	return string(out), err
}
