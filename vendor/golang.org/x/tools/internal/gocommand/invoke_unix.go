// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package gocommand

import (
	"errors"
	"os/exec"
	"syscall"
)

// sigStuckProcess is the signal to send to kill a hanging subprocess.
// Send SIGQUIT to get a stack trace.
var sigStuckProcess = syscall.SIGQUIT

// setProcessGroup sets Setpgid so that the child process runs as the leader of
// its own process group. Any subprocesses spawned by the Go command (such as
// cgo, compilers, or assemblers) will inherit this process group, allowing
// cancellation signals to reach all descendant processes rather than orphaning
// them (see golang/go#81408).
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// interruptProcess sends SIGINT to the entire process group (-pid) so that both
// the Go command and any subprocesses (such as C compilers) get
// the interrupt signal and terminate promptly.
//
// Unlike cmd.Process.Signal, syscall.Kill(-pid, ...) does not synchronize with
// cmd.Process.Wait via os.Process's internal lock. PGID reuse is not a
// problem, as the kernel reserves the PGID as long as any member
// of the process group is alive, and runCmdContext only signals the
// process group while cmd.Wait is still blocked on the process or its open pipes.
func interruptProcess(cmd *exec.Cmd) error {
	return syscall.Kill(-cmd.Process.Pid, syscall.SIGINT)
}

// killProcess sends SIGKILL to the entire process group (-pid) to forcefully
// terminate any remaining descendant processes. It ignores ESRCH if the process
// group has already terminated.
func killProcess(cmd *exec.Cmd) error {
	err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}
