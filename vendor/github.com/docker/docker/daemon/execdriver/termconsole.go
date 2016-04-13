package execdriver

import (
	"io"
	"os/exec"
)

type StdConsole struct {
}

func NewStdConsole(processConfig *ProcessConfig, pipes *Pipes) (*StdConsole, error) {
	std := &StdConsole{}

	if err := std.AttachPipes(&processConfig.Cmd, pipes); err != nil {
		return nil, err
	}
	return std, nil
}

func (s *StdConsole) AttachPipes(command *exec.Cmd, pipes *Pipes) error {
	command.Stdout = pipes.Stdout
	command.Stderr = pipes.Stderr

	if pipes.Stdin != nil {
		stdin, err := command.StdinPipe()
		if err != nil {
			return err
		}

		go func() {
			defer stdin.Close()
			io.Copy(stdin, pipes.Stdin)
		}()
	}
	return nil
}

func (s *StdConsole) Resize(h, w int) error {
	// we do not need to reside a non tty
	return nil
}

func (s *StdConsole) Close() error {
	// nothing to close here
	return nil
}
