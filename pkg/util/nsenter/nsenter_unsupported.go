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
	"k8s.io/utils/exec"
)

// Nsenter is part of experimental support for running the kubelet
// in a container.
type Nsenter struct {
	// a map of commands to their paths on the host filesystem
	Paths map[string]string
}

// NewNsenter constructs a new instance of Nsenter
func NewNsenter() *Nsenter {
	return &Nsenter{}
}

// Exec executes nsenter commands in hostProcMountNsPath mount namespace
func (ne *Nsenter) Exec(cmd string, args []string) exec.Cmd {
	return nil
}

// AbsHostPath returns the absolute runnable path for a specified command
func (ne *Nsenter) AbsHostPath(command string) string {
	return ""
}

// SupportsSystemd checks whether command systemd-run exists
func (ne *Nsenter) SupportsSystemd() (string, bool) {
	return "", false
}
