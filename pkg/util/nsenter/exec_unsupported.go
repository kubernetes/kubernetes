// +build !linux

/*
Copyright 2017 The Kubernetes Authors.

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

package nsenter

import (
	"context"
	"fmt"

	"k8s.io/utils/exec"
)

// Executor wraps executor interface to be executed via nsenter
type Executor struct {
	// Exec implementation
	executor exec.Interface
	// Path to the host's root proc path
	hostProcMountNsPath string
}

// NewNsenterExecutor returns new nsenter based executor
func NewNsenterExecutor(hostRootFsPath string, executor exec.Interface) *Executor {
	nsExecutor := &Executor{
		hostProcMountNsPath: hostRootFsPath,
		executor:            executor,
	}
	return nsExecutor
}

// Command returns a command wrapped with nenter
func (nsExecutor *Executor) Command(cmd string, args ...string) exec.Cmd {
	return nil
}

// CommandContext returns a CommandContext wrapped with nsenter
func (nsExecutor *Executor) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return nil
}

// LookPath returns a LookPath wrapped with nsenter
func (nsExecutor *Executor) LookPath(file string) (string, error) {
	return "", fmt.Errorf("not implemented, error looking up : %s", file)
}
