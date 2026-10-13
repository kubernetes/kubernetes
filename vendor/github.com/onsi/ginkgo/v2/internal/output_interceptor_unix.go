//go:build freebsd || openbsd || netbsd || dragonfly || darwin || linux || solaris
// +build freebsd openbsd netbsd dragonfly darwin linux solaris

package internal

import (
	"os"

	"golang.org/x/sys/unix"
)

func NewOutputInterceptor() OutputInterceptor {
	return &genericOutputInterceptor{
		interceptedContent: make(chan string),
		pipeChannel:        make(chan pipePair),
		shutdown:           make(chan any),
		implementation:     &dupSyscallOutputInterceptorImpl{},
	}
}

type dupSyscallOutputInterceptorImpl struct{}

func (impl *dupSyscallOutputInterceptorImpl) CreateStdoutStderrClones() (*os.File, *os.File) {
	// To clone stdout and stderr we:
	// First, create two clone file descriptors that point to the stdout and stderr file descriptions
	stdoutCloneFD, _ := unix.Dup(1)
	stderrCloneFD, _ := unix.Dup(2)

	// Important, set the fds to FD_CLOEXEC to prevent them leaking into childs
	// https://github.com/onsi/ginkgo/issues/1191
	flags, err := unix.FcntlInt(uintptr(stdoutCloneFD), unix.F_GETFD, 0)
	if err == nil {
		unix.FcntlInt(uintptr(stdoutCloneFD), unix.F_SETFD, flags|unix.FD_CLOEXEC)
	}
	flags, err = unix.FcntlInt(uintptr(stderrCloneFD), unix.F_GETFD, 0)
	if err == nil {
		unix.FcntlInt(uintptr(stderrCloneFD), unix.F_SETFD, flags|unix.FD_CLOEXEC)
	}

	// And then wrap the clone file descriptors in files.
	// One benefit of this (that we don't use yet) is that we can actually write
	// to these files to emit output to the console even though we're intercepting output
	stdoutClone := os.NewFile(uintptr(stdoutCloneFD), "stdout-clone")
	stderrClone := os.NewFile(uintptr(stderrCloneFD), "stderr-clone")

	//these clones remain alive throughout the lifecycle of the suite and don't need to be recreated
	//this speeds things up a bit, actually.
	return stdoutClone, stderrClone
}

func (impl *dupSyscallOutputInterceptorImpl) ConnectPipeToStdoutStderr(pipeWriter *os.File) {
	// To redirect output to our pipe we need to point the 1 and 2 file descriptors (which is how the world tries to log things)
	// to the write end of the pipe.
	// We do this with Dup2 (possibly Dup3 on some architectures) to have file descriptors 1 and 2 point to the same file description as the pipeWriter
	// This effectively shunts data written to stdout and stderr to the write end of our pipe
	unix.Dup2(int(pipeWriter.Fd()), 1)
	unix.Dup2(int(pipeWriter.Fd()), 2)
}

func (impl *dupSyscallOutputInterceptorImpl) RestoreStdoutStderrFromClones(stdoutClone *os.File, stderrClone *os.File) {
	// To restore stdour/stderr from the clones we have the 1 and 2 file descriptors
	// point to the original file descriptions that we saved off in the clones.
	// This has the added benefit of closing the connection between these descriptors and the write end of the pipe
	// which is important to cause the io.Copy on the pipe.Reader to end.
	unix.Dup2(int(stdoutClone.Fd()), 1)
	unix.Dup2(int(stderrClone.Fd()), 2)
}

func (impl *dupSyscallOutputInterceptorImpl) ShutdownClones(stdoutClone *os.File, stderrClone *os.File) {
	// We're done with the clones so we can close them to clean up after ourselves
	stdoutClone.Close()
	stderrClone.Close()
}
