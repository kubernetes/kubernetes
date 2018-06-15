// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package gps

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"syscall"
	"time"

	"github.com/pkg/errors"
)

type cmd struct {
	// ctx is provided by the caller; SIGINT is sent when it is cancelled.
	ctx context.Context
	Cmd *exec.Cmd
}

func commandContext(ctx context.Context, name string, arg ...string) cmd {
	c := exec.Command(name, arg...)

	// Force subprocesses into their own process group, rather than being in the
	// same process group as the dep process. Because Ctrl-C sent from a
	// terminal will send the signal to the entire currently running process
	// group, this allows us to directly manage the issuance of signals to
	// subprocesses.
	c.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
		Pgid:    0,
	}

	return cmd{ctx: ctx, Cmd: c}
}

// CombinedOutput is like (*os/exec.Cmd).CombinedOutput except that it
// terminates subprocesses gently (via os.Interrupt), but resorts to Kill if
// the subprocess fails to exit after 1 minute.
func (c cmd) CombinedOutput() ([]byte, error) {
	// Adapted from (*os/exec.Cmd).CombinedOutput
	if c.Cmd.Stdout != nil {
		return nil, errors.New("exec: Stdout already set")
	}
	if c.Cmd.Stderr != nil {
		return nil, errors.New("exec: Stderr already set")
	}
	var b bytes.Buffer
	c.Cmd.Stdout = &b
	c.Cmd.Stderr = &b
	if err := c.Cmd.Start(); err != nil {
		return nil, err
	}

	// Adapted from (*os/exec.Cmd).Start
	waitDone := make(chan struct{})
	defer close(waitDone)
	go func() {
		select {
		case <-c.ctx.Done():
			if err := c.Cmd.Process.Signal(os.Interrupt); err != nil {
				// If an error comes back from attempting to signal, proceed
				// immediately to hard kill.
				_ = c.Cmd.Process.Kill()
			} else {
				defer time.AfterFunc(time.Minute, func() {
					_ = c.Cmd.Process.Kill()
				}).Stop()
				<-waitDone
			}
		case <-waitDone:
		}
	}()

	err := c.Cmd.Wait()
	return b.Bytes(), err
}
