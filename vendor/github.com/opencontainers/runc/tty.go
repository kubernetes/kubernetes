// +build linux

package main

import (
	"fmt"
	"io"
	"os"

	"github.com/docker/docker/pkg/term"
	"github.com/opencontainers/runc/libcontainer"
)

// newTty creates a new tty for use with the container.  If a tty is not to be
// created for the process, pipes are created so that the TTY of the parent
// process are not inherited by the container.
func newTty(create bool, p *libcontainer.Process, rootuid int, console string) (*tty, error) {
	if create {
		return createTty(p, rootuid, console)
	}
	return createStdioPipes(p, rootuid)
}

// setup standard pipes so that the TTY of the calling runc process
// is not inherited by the container.
func createStdioPipes(p *libcontainer.Process, rootuid int) (*tty, error) {
	i, err := p.InitializeIO(rootuid)
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
	go io.Copy(i.Stdin, os.Stdin)
	go io.Copy(os.Stdout, i.Stdout)
	go io.Copy(os.Stderr, i.Stderr)
	return t, nil
}

func createTty(p *libcontainer.Process, rootuid int, consolePath string) (*tty, error) {
	if consolePath != "" {
		if err := p.ConsoleFromPath(consolePath); err != nil {
			return nil, err
		}
		return &tty{}, nil
	}
	console, err := p.NewConsole(rootuid)
	if err != nil {
		return nil, err
	}
	go io.Copy(console, os.Stdin)
	go io.Copy(os.Stdout, console)

	state, err := term.SetRawTerminal(os.Stdin.Fd())
	if err != nil {
		return nil, fmt.Errorf("failed to set the terminal from the stdin: %v", err)
	}
	t := &tty{
		console: console,
		state:   state,
		closers: []io.Closer{
			console,
		},
	}
	return t, nil
}

type tty struct {
	console libcontainer.Console
	state   *term.State
	closers []io.Closer
}

func (t *tty) Close() error {
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
	return term.SetWinsize(t.console.Fd(), ws)
}
