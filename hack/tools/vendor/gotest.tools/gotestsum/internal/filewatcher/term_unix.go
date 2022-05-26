// +build !windows

package filewatcher

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
	"gotest.tools/gotestsum/log"
)

type redoHandler struct {
	ch    chan RunOptions
	reset func()
}

func newRedoHandler() *redoHandler {
	h := &redoHandler{ch: make(chan RunOptions)}
	h.SetupTerm()
	return h
}

func (r *redoHandler) SetupTerm() {
	if r == nil {
		return
	}
	fd := int(os.Stdin.Fd())
	reset, err := enableNonBlockingRead(fd)
	if err != nil {
		log.Warnf("failed to put terminal (fd %d) into raw mode: %v", fd, err)
		return
	}
	r.reset = reset
}

func enableNonBlockingRead(fd int) (func(), error) {
	term, err := unix.IoctlGetTermios(fd, tcGet)
	if err != nil {
		return nil, err
	}

	state := *term
	reset := func() {
		if err := unix.IoctlSetTermios(fd, tcSet, &state); err != nil {
			log.Debugf("failed to reset fd %d: %v", fd, err)
		}
	}

	term.Lflag &^= unix.ECHO | unix.ICANON
	term.Cc[unix.VMIN] = 1
	term.Cc[unix.VTIME] = 0
	if err := unix.IoctlSetTermios(fd, tcSet, term); err != nil {
		reset()
		return nil, err
	}
	return reset, nil
}

func (r *redoHandler) Run(ctx context.Context) {
	if r == nil {
		return
	}
	in := bufio.NewReader(os.Stdin)
	for {
		char, err := in.ReadByte()
		if err != nil {
			log.Warnf("failed to read input: %v", err)
			return
		}
		log.Debugf("received byte %v (%v)", char, string(char))

		var chResume chan struct{}
		switch char {
		case 'r':
			chResume = make(chan struct{})
			r.ch <- RunOptions{resume: chResume}
		case 'd':
			chResume = make(chan struct{})
			r.ch <- RunOptions{Debug: true, resume: chResume}
		case '\n':
			fmt.Println()
			continue
		default:
			continue
		}

		select {
		case <-ctx.Done():
			return
		case <-chResume:
		}
	}
}

func (r *redoHandler) Ch() <-chan RunOptions {
	if r == nil {
		return nil
	}
	return r.ch
}

func (r *redoHandler) ResetTerm() {
	if r != nil && r.reset != nil {
		r.reset()
	}
}
