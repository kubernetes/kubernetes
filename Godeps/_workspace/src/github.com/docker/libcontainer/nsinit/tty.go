package main

import (
	"io"
	"os"

	"github.com/codegangsta/cli"
	"github.com/docker/docker/pkg/term"
	"github.com/docker/libcontainer"
)

func newTty(context *cli.Context, p *libcontainer.Process, rootuid int) (*tty, error) {
	if context.Bool("tty") {
		console, err := p.NewConsole(rootuid)
		if err != nil {
			return nil, err
		}
		return &tty{
			console: console,
		}, nil
	}
	return &tty{}, nil
}

type tty struct {
	console libcontainer.Console
	state   *term.State
}

func (t *tty) Close() error {
	if t.console != nil {
		t.console.Close()
	}
	if t.state != nil {
		term.RestoreTerminal(os.Stdin.Fd(), t.state)
	}
	return nil
}

func (t *tty) attach(process *libcontainer.Process) error {
	if t.console != nil {
		go io.Copy(t.console, os.Stdin)
		go io.Copy(os.Stdout, t.console)
		state, err := term.SetRawTerminal(os.Stdin.Fd())
		if err != nil {
			return err
		}
		t.state = state
		process.Stderr = nil
		process.Stdout = nil
		process.Stdin = nil
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
