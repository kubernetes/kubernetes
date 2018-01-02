package cmd

import (
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestRunCommand(t *testing.T) {
	// TODO Windows: Port this test
	if runtime.GOOS == "windows" {
		t.Skip("Needs porting to Windows")
	}

	var cmd string
	if runtime.GOOS == "solaris" {
		cmd = "gls"
	} else {
		cmd = "ls"
	}
	result := RunCommand(cmd)
	result.Assert(t, Expected{})

	result = RunCommand("doesnotexists")
	expectedError := `exec: "doesnotexists": executable file not found`
	result.Assert(t, Expected{ExitCode: 127, Error: expectedError})

	result = RunCommand(cmd, "-z")
	result.Assert(t, Expected{
		ExitCode: 2,
		Error:    "exit status 2",
		Err:      "invalid option",
	})
	assert.Contains(t, result.Combined(), "invalid option")
}

func TestRunCommandWithCombined(t *testing.T) {
	// TODO Windows: Port this test
	if runtime.GOOS == "windows" {
		t.Skip("Needs porting to Windows")
	}

	result := RunCommand("ls", "-a")
	result.Assert(t, Expected{})

	assert.Contains(t, result.Combined(), "..")
	assert.Contains(t, result.Stdout(), "..")
}

func TestRunCommandWithTimeoutFinished(t *testing.T) {
	// TODO Windows: Port this test
	if runtime.GOOS == "windows" {
		t.Skip("Needs porting to Windows")
	}

	result := RunCmd(Cmd{
		Command: []string{"ls", "-a"},
		Timeout: 50 * time.Millisecond,
	})
	result.Assert(t, Expected{Out: ".."})
}

func TestRunCommandWithTimeoutKilled(t *testing.T) {
	// TODO Windows: Port this test
	if runtime.GOOS == "windows" {
		t.Skip("Needs porting to Windows")
	}

	command := []string{"sh", "-c", "while true ; do echo 1 ; sleep .5 ; done"}
	result := RunCmd(Cmd{Command: command, Timeout: 1250 * time.Millisecond})
	result.Assert(t, Expected{Timeout: true})

	ones := strings.Split(result.Stdout(), "\n")
	assert.Len(t, ones, 4)
}

func TestRunCommandWithErrors(t *testing.T) {
	result := RunCommand("/foobar")
	result.Assert(t, Expected{Error: "foobar", ExitCode: 127})
}

func TestRunCommandWithStdoutStderr(t *testing.T) {
	result := RunCommand("echo", "hello", "world")
	result.Assert(t, Expected{Out: "hello world\n", Err: None})
}

func TestRunCommandWithStdoutStderrError(t *testing.T) {
	result := RunCommand("doesnotexists")

	expected := `exec: "doesnotexists": executable file not found`
	result.Assert(t, Expected{Out: None, Err: None, ExitCode: 127, Error: expected})

	switch runtime.GOOS {
	case "windows":
		expected = "ls: unknown option"
	case "solaris":
		expected = "gls: invalid option"
	default:
		expected = "ls: invalid option"
	}

	var cmd string
	if runtime.GOOS == "solaris" {
		cmd = "gls"
	} else {
		cmd = "ls"
	}
	result = RunCommand(cmd, "-z")
	result.Assert(t, Expected{
		Out:      None,
		Err:      expected,
		ExitCode: 2,
		Error:    "exit status 2",
	})
}
