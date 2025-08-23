package client

import (
	"io"
	"os"
	"os/exec"
)

// Program is an interface to execute external programs.
type Program interface {
	Output() ([]byte, error)
	Input(in io.Reader)
}

// ProgramFunc is a type of function that initializes programs based on arguments.
type ProgramFunc func(args ...string) Program

// NewShellProgramFunc creates a [ProgramFunc] to run command in a [Shell].
func NewShellProgramFunc(command string) ProgramFunc {
	return func(args ...string) Program {
		return createProgramCmdRedirectErr(command, args, nil)
	}
}

// NewShellProgramFuncWithEnv creates a [ProgramFunc] tu run command
// in a [Shell] with the given environment variables.
func NewShellProgramFuncWithEnv(command string, env *map[string]string) ProgramFunc {
	return func(args ...string) Program {
		return createProgramCmdRedirectErr(command, args, env)
	}
}

func createProgramCmdRedirectErr(command string, args []string, env *map[string]string) *Shell {
	ec := exec.Command(command, args...)
	if env != nil {
		for k, v := range *env {
			ec.Env = append(ec.Environ(), k+"="+v)
		}
	}
	ec.Stderr = os.Stderr
	return &Shell{cmd: ec}
}

// Shell invokes shell commands to talk with a remote credentials-helper.
type Shell struct {
	cmd *exec.Cmd
}

// Output returns responses from the remote credentials-helper.
func (s *Shell) Output() ([]byte, error) {
	return s.cmd.Output()
}

// Input sets the input to send to a remote credentials-helper.
func (s *Shell) Input(in io.Reader) {
	s.cmd.Stdin = in
}
