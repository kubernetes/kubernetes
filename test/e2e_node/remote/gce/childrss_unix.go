//go:build unix

/*
Copyright The Kubernetes Authors.

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

package gce

import (
	"os/exec"
	"syscall"
)

// childRSSkB returns the peak resident set size in kB of a finished command's
// process, or 0 when it is unavailable.
func childRSSkB(cmd *exec.Cmd) int64 {
	if cmd.ProcessState == nil {
		return 0
	}
	if ru, ok := cmd.ProcessState.SysUsage().(*syscall.Rusage); ok {
		return ru.Maxrss
	}
	return 0
}
