// +build linux

package main

import (
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/docker/docker/pkg/term"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/utils"
)

type tty struct {
	console   libcontainer.Console
	state     *term.State
	closers   []io.Closer
	postStart []io.Closer
	wg        sync.WaitGroup
	consoleC  chan error
}

func (t *tty) copyIO(w io.Writer, r io.ReadCloser) {
	defer t.wg.Done()
	io.Copy(w, r)
	r.Close()
}

// setup pipes for the process so that advanced features like c/r are able to easily checkpoint
// and restore the process's IO without depending on a host specific path or device
func setupProcessPipes(p *libcontainer.Process, rootuid, rootgid int) (*tty, error) {
	i, err := p.InitializeIO(rootuid, rootgid)
	if err != nil {
		return nil, err
	}
	t := &tty{
		closers: []io.Closer{
			i.Stdin,
			i.Stdout,
			i.Stderr,
		},
	}
	// add the process's io to the post start closers if they support close
	for _, cc := range []interface{}{
		p.Stdin,
		p.Stdout,
		p.Stderr,
	} {
		if c, ok := cc.(io.Closer); ok {
			t.postStart = append(t.postStart, c)
		}
	}
	go func() {
		io.Copy(i.Stdin, os.Stdin)
		i.Stdin.Close()
	}()
	t.wg.Add(2)
	go t.copyIO(os.Stdout, i.Stdout)
	go t.copyIO(os.Stderr, i.Stderr)
	return t, nil
}

func inheritStdio(process *libcontainer.Process) error {
	process.Stdin = os.Stdin
	process.Stdout = os.Stdout
	process.Stderr = os.Stderr
	return nil
}

func (t *tty) recvtty(process *libcontainer.Process, socket *os.File) error {
	f, err := utils.RecvFd(socket)
	if err != nil {
		return err
	}
	if err = libcontainer.SaneTerminal(f); err != nil {
		return err
	}
	console := libcontainer.ConsoleFromFile(f)
	go io.Copy(console, os.Stdin)
	t.wg.Add(1)
	go t.copyIO(os.Stdout, console)
	state, err := term.SetRawTerminal(os.Stdin.Fd())
	if err != nil {
		return fmt.Errorf("failed to set the terminal from the stdin: %v", err)
	}
	t.state = state
	t.console = console
	t.closers = []io.Closer{console}
	return nil
}

func (t *tty) waitConsole() error {
	if t.consoleC != nil {
		return <-t.consoleC
	}
	return nil
}

// ClosePostStart closes any fds that are provided to the container and dup2'd
// so that we no longer have copy in our process.
func (t *tty) ClosePostStart() error {
	for _, c := range t.postStart {
		c.Close()
	}
	return nil
}

// Close closes all open fds for the tty and/or restores the orignal
// stdin state to what it was prior to the container execution
func (t *tty) Close() error {
	// ensure that our side of the fds are always closed
	for _, c := range t.postStart {
		c.Close()
	}
	// wait for the copy routines to finish before closing the fds
	t.wg.Wait()
	for _, c := range t.closers {
		c.Close()
	}
	if t.state != nil {
		term.RestoreTerminal(os.Stdin.Fd(), t.state)
	}
	return nil
}

func (t *tty) resize() error {
	if t.console == nil {
		return nil
	}
	ws, err := term.GetWinsize(os.Stdin.Fd())
	if err != nil {
		return err
	}
	return term.SetWinsize(t.console.File().Fd(), ws)
}
