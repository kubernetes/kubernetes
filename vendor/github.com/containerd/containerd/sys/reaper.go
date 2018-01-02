// +build !windows

package sys

import "golang.org/x/sys/unix"

// Exit is the wait4 information from an exited process
type Exit struct {
	Pid    int
	Status int
}

// Reap reaps all child processes for the calling process and returns their
// exit information
func Reap(wait bool) (exits []Exit, err error) {
	var (
		ws  unix.WaitStatus
		rus unix.Rusage
	)
	flag := unix.WNOHANG
	if wait {
		flag = 0
	}
	for {
		pid, err := unix.Wait4(-1, &ws, flag, &rus)
		if err != nil {
			if err == unix.ECHILD {
				return exits, nil
			}
			return exits, err
		}
		if pid <= 0 {
			return exits, nil
		}
		exits = append(exits, Exit{
			Pid:    pid,
			Status: exitStatus(ws),
		})
	}
}

const exitSignalOffset = 128

// exitStatus returns the correct exit status for a process based on if it
// was signaled or exited cleanly
func exitStatus(status unix.WaitStatus) int {
	if status.Signaled() {
		return exitSignalOffset + int(status.Signal())
	}
	return status.ExitStatus()
}
