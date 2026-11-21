// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix

package gocommand

import (
	"errors"
	"os"
	"os/exec"
)

// sigStuckProcess is the signal to send to kill a hanging subprocess.
// On Unix we send SIGQUIT, but on non-Unix we only have os.Kill.
var sigStuckProcess = os.Kill

// setProcessGroup is a no-op on non-Unix platforms.
//
// As a consequence of not using process groups (or Windows job objects),
// signals sent to cmd.Process will not propagate to descendant processes
// spawned by the Go command (such as cgo or C compilers). If the command
// is cancelled, those subprocesses may continue running in the background
// until they finish or fail due to broken pipes.
// TODO(pjw): consider adding Windows job objects.
func setProcessGroup(cmd *exec.Cmd) {
}

// interruptProcess sends an interrupt signal to the process. On Windows,
// os.Interrupt is not supported and will return an error, causing the caller
// to fall back to killProcess immediately.
func interruptProcess(cmd *exec.Cmd) error {
	return cmd.Process.Signal(os.Interrupt)
}

// killProcess terminates the process directly. Subprocesses spawned by the
// Go command will not be killed directly.
func killProcess(cmd *exec.Cmd) error {
	err := cmd.Process.Kill()
	if errors.Is(err, os.ErrProcessDone) {
		return nil
	}
	return err
}
