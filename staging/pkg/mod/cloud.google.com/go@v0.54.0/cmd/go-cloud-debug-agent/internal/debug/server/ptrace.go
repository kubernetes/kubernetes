// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux

package server

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
	"time"
)

// ptraceRun runs all the closures from fc on a dedicated OS thread. Errors
// are returned on ec. Both channels must be unbuffered, to ensure that the
// resultant error is sent back to the same goroutine that sent the closure.
func ptraceRun(fc chan func() error, ec chan error) {
	if cap(fc) != 0 || cap(ec) != 0 {
		panic("ptraceRun was given buffered channels")
	}
	runtime.LockOSThread()
	for f := range fc {
		ec <- f()
	}
}

func (s *Server) startProcess(name string, argv []string, attr *os.ProcAttr) (proc *os.Process, err error) {
	s.fc <- func() error {
		var err1 error
		proc, err1 = os.StartProcess(name, argv, attr)
		return err1
	}
	err = <-s.ec
	return
}

func (s *Server) ptraceCont(pid int, signal int) (err error) {
	s.fc <- func() error {
		return syscall.PtraceCont(pid, signal)
	}
	return <-s.ec
}

func (s *Server) ptraceGetRegs(pid int, regsout *syscall.PtraceRegs) (err error) {
	s.fc <- func() error {
		return syscall.PtraceGetRegs(pid, regsout)
	}
	return <-s.ec
}

func (s *Server) ptracePeek(pid int, addr uintptr, out []byte) (err error) {
	s.fc <- func() error {
		n, err := syscall.PtracePeekText(pid, addr, out)
		if err != nil {
			return err
		}
		if n != len(out) {
			return fmt.Errorf("ptracePeek: peeked %d bytes, want %d", n, len(out))
		}
		return nil
	}
	return <-s.ec
}

func (s *Server) ptracePoke(pid int, addr uintptr, data []byte) (err error) {
	s.fc <- func() error {
		n, err := syscall.PtracePokeText(pid, addr, data)
		if err != nil {
			return err
		}
		if n != len(data) {
			return fmt.Errorf("ptracePoke: poked %d bytes, want %d", n, len(data))
		}
		return nil
	}
	return <-s.ec
}

func (s *Server) ptraceSetOptions(pid int, options int) (err error) {
	s.fc <- func() error {
		return syscall.PtraceSetOptions(pid, options)
	}
	return <-s.ec
}

func (s *Server) ptraceSetRegs(pid int, regs *syscall.PtraceRegs) (err error) {
	s.fc <- func() error {
		return syscall.PtraceSetRegs(pid, regs)
	}
	return <-s.ec
}

func (s *Server) ptraceSingleStep(pid int) (err error) {
	s.fc <- func() error {
		return syscall.PtraceSingleStep(pid)
	}
	return <-s.ec
}

type breakpointsChangedError struct {
	call call
}

func (*breakpointsChangedError) Error() string {
	return "breakpoints changed"
}

func (s *Server) wait(pid int, allowBreakpointsChange bool) (wpid int, status syscall.WaitStatus, err error) {
	// We poll syscall.Wait4 with WNOHANG, sleeping in between, as a poor man's
	// waitpid-with-timeout. This allows adding and removing breakpoints
	// concurrently with waiting to hit an existing breakpoint.
	f := func() error {
		var err1 error
		wpid, err1 = syscall.Wait4(pid, &status, syscall.WALL|syscall.WNOHANG, nil)
		return err1
	}

	const (
		minSleep = 1 * time.Microsecond
		maxSleep = 100 * time.Millisecond
	)
	for sleep := minSleep; ; {
		s.fc <- f
		err = <-s.ec

		// wpid == 0 means that wait found nothing (and returned due to WNOHANG).
		if wpid != 0 {
			return
		}

		if allowBreakpointsChange {
			select {
			case c := <-s.breakpointc:
				return 0, 0, &breakpointsChangedError{c}
			default:
			}
		}

		time.Sleep(sleep)
		if sleep < maxSleep {
			sleep *= 10
		}
	}
}
