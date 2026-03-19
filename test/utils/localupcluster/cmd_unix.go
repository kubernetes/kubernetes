//go:build !windows

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

package localupcluster

import (
	"fmt"
	"os/exec"
	"syscall"

	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// setProcessGroup places cmd in its own process group so that killProcessGroup
// can terminate the entire group, including sudo-launched children.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// killProcessGroup sends SIGKILL to the process group whose leader has the
// given PID. For sudo-wrapped processes it attempts "sudo -n kill -KILL -<pgid>"
// so that root-owned children (e.g. kubelet launched by sudo) are also killed,
// but falls back to a non-sudo kill if that fails.
func killProcessGroup(tCtx ktesting.TContext, pid int, useSudo bool) {
	if useSudo {
		// Use sudo to kill root-owned processes (e.g. kubelet launched via sudo).
		// -n avoids interactive password prompts; failure is non-fatal because
		// the process may have already exited (ESRCH) or sudo may not be available.
		err := exec.Command("sudo", "-n", "kill", "-KILL", fmt.Sprintf("-%d", pid)).Run()
		if err == nil {
			return
		}
		tCtx.Logf("sudo kill of process group -%d failed: %v; falling back to non-sudo kill", pid, err)
	}
	// ESRCH means the process group no longer exists — that's the desired outcome.
	if err := syscall.Kill(-pid, syscall.SIGKILL); err != nil && err != syscall.ESRCH {
		tCtx.Errorf("kill of process group -%d failed: %v", pid, err)
	}
}
