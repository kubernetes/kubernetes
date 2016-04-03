package runconfig

import (
	"fmt"
	"io/ioutil"
	"testing"

	flag "github.com/docker/docker/pkg/mflag"
)

type arguments struct {
	args []string
}

func TestParseExec(t *testing.T) {
	invalids := map[*arguments]error{
		&arguments{[]string{"-unknown"}}: fmt.Errorf("flag provided but not defined: -unknown"),
		&arguments{[]string{"-u"}}:       fmt.Errorf("flag needs an argument: -u"),
		&arguments{[]string{"--user"}}:   fmt.Errorf("flag needs an argument: --user"),
	}
	valids := map[*arguments]*ExecConfig{
		&arguments{
			[]string{"container", "command"},
		}: {
			Container:    "container",
			Cmd:          []string{"command"},
			AttachStdout: true,
			AttachStderr: true,
		},
		&arguments{
			[]string{"container", "command1", "command2"},
		}: {
			Container:    "container",
			Cmd:          []string{"command1", "command2"},
			AttachStdout: true,
			AttachStderr: true,
		},
		&arguments{
			[]string{"-i", "-t", "-u", "uid", "container", "command"},
		}: {
			User:         "uid",
			AttachStdin:  true,
			AttachStdout: true,
			AttachStderr: true,
			Tty:          true,
			Container:    "container",
			Cmd:          []string{"command"},
		},
		&arguments{
			[]string{"-d", "container", "command"},
		}: {
			AttachStdin:  false,
			AttachStdout: false,
			AttachStderr: false,
			Detach:       true,
			Container:    "container",
			Cmd:          []string{"command"},
		},
		&arguments{
			[]string{"-t", "-i", "-d", "container", "command"},
		}: {
			AttachStdin:  false,
			AttachStdout: false,
			AttachStderr: false,
			Detach:       true,
			Tty:          true,
			Container:    "container",
			Cmd:          []string{"command"},
		},
	}
	for invalid, expectedError := range invalids {
		cmd := flag.NewFlagSet("exec", flag.ContinueOnError)
		cmd.ShortUsage = func() {}
		cmd.SetOutput(ioutil.Discard)
		_, err := ParseExec(cmd, invalid.args)
		if err == nil || err.Error() != expectedError.Error() {
			t.Fatalf("Expected an error [%v] for %v, got %v", expectedError, invalid, err)
		}

	}
	for valid, expectedExecConfig := range valids {
		cmd := flag.NewFlagSet("exec", flag.ContinueOnError)
		cmd.ShortUsage = func() {}
		cmd.SetOutput(ioutil.Discard)
		execConfig, err := ParseExec(cmd, valid.args)
		if err != nil {
			t.Fatal(err)
		}
		if !compareExecConfig(expectedExecConfig, execConfig) {
			t.Fatalf("Expected [%v] for %v, got [%v]", expectedExecConfig, valid, execConfig)
		}
	}
}

func compareExecConfig(config1 *ExecConfig, config2 *ExecConfig) bool {
	if config1.AttachStderr != config2.AttachStderr {
		return false
	}
	if config1.AttachStdin != config2.AttachStdin {
		return false
	}
	if config1.AttachStdout != config2.AttachStdout {
		return false
	}
	if config1.Container != config2.Container {
		return false
	}
	if config1.Detach != config2.Detach {
		return false
	}
	if config1.Privileged != config2.Privileged {
		return false
	}
	if config1.Tty != config2.Tty {
		return false
	}
	if config1.User != config2.User {
		return false
	}
	if len(config1.Cmd) != len(config2.Cmd) {
		return false
	}
	for index, value := range config1.Cmd {
		if value != config2.Cmd[index] {
			return false
		}
	}
	return true
}
