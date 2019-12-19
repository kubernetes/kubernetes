package main

import (
	"os/exec"
	"syscall"

	"github.com/pkg/errors"
)

// Copied from gotestyourself/icmd

// GetExitCode returns the ExitStatus of a process from the error returned by
// exec.Run(). If the exit status is not available an error is returned.
func GetExitCode(err error) (int, error) {
	if exiterr, ok := err.(*exec.ExitError); ok {
		if procExit, ok := exiterr.Sys().(syscall.WaitStatus); ok {
			return procExit.ExitStatus(), nil
		}
	}
	return 0, errors.Wrap(err, "failed to get exit code")
}

// ExitCodeWithDefault returns ExitStatus of a process from the error returned by
// exec.Run(). If the exit status is not available return a default of 127.
func ExitCodeWithDefault(err error) int {
	if err == nil {
		return 0
	}
	exitCode, exiterr := GetExitCode(err)
	if exiterr != nil {
		// we've failed to retrieve exit code, so we set it to 127
		return 127
	}
	return exitCode
}
