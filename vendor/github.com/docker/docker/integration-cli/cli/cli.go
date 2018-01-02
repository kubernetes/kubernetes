package cli

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/integration-cli/daemon"
	"github.com/docker/docker/integration-cli/environment"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/pkg/errors"
)

var (
	testEnv  *environment.Execution
	onlyOnce sync.Once
)

// EnsureTestEnvIsLoaded make sure the test environment is loaded for this package
func EnsureTestEnvIsLoaded(t testingT) {
	var doIt bool
	var err error
	onlyOnce.Do(func() {
		doIt = true
	})

	if !doIt {
		return
	}
	testEnv, err = environment.New()
	if err != nil {
		t.Fatalf("error loading testenv : %v", err)
	}
}

// CmdOperator defines functions that can modify a command
type CmdOperator func(*icmd.Cmd) func()

type testingT interface {
	Fatal(args ...interface{})
	Fatalf(string, ...interface{})
}

// DockerCmd executes the specified docker command and expect a success
func DockerCmd(t testingT, args ...string) *icmd.Result {
	return Docker(Args(args...)).Assert(t, icmd.Success)
}

// BuildCmd executes the specified docker build command and expect a success
func BuildCmd(t testingT, name string, cmdOperators ...CmdOperator) *icmd.Result {
	return Docker(Build(name), cmdOperators...).Assert(t, icmd.Success)
}

// InspectCmd executes the specified docker inspect command and expect a success
func InspectCmd(t testingT, name string, cmdOperators ...CmdOperator) *icmd.Result {
	return Docker(Inspect(name), cmdOperators...).Assert(t, icmd.Success)
}

// WaitRun will wait for the specified container to be running, maximum 5 seconds.
func WaitRun(t testingT, name string, cmdOperators ...CmdOperator) {
	WaitForInspectResult(t, name, "{{.State.Running}}", "true", 5*time.Second, cmdOperators...)
}

// WaitExited will wait for the specified container to state exit, subject
// to a maximum time limit in seconds supplied by the caller
func WaitExited(t testingT, name string, timeout time.Duration, cmdOperators ...CmdOperator) {
	WaitForInspectResult(t, name, "{{.State.Status}}", "exited", timeout, cmdOperators...)
}

// WaitRestart will wait for the specified container to restart once
func WaitRestart(t testingT, name string, timeout time.Duration, cmdOperators ...CmdOperator) {
	WaitForInspectResult(t, name, "{{.RestartCount}}", "1", timeout, cmdOperators...)
}

// WaitForInspectResult waits for the specified expression to be equals to the specified expected string in the given time.
func WaitForInspectResult(t testingT, name, expr, expected string, timeout time.Duration, cmdOperators ...CmdOperator) {
	after := time.After(timeout)

	args := []string{"inspect", "-f", expr, name}
	for {
		result := Docker(Args(args...), cmdOperators...)
		if result.Error != nil {
			if !strings.Contains(strings.ToLower(result.Stderr()), "no such") {
				t.Fatalf("error executing docker inspect: %v\n%s",
					result.Stderr(), result.Stdout())
			}
			select {
			case <-after:
				t.Fatal(result.Error)
			default:
				time.Sleep(10 * time.Millisecond)
				continue
			}
		}

		out := strings.TrimSpace(result.Stdout())
		if out == expected {
			break
		}

		select {
		case <-after:
			t.Fatalf("condition \"%q == %q\" not true in time (%v)", out, expected, timeout)
		default:
		}

		time.Sleep(100 * time.Millisecond)
	}
}

// Docker executes the specified docker command
func Docker(cmd icmd.Cmd, cmdOperators ...CmdOperator) *icmd.Result {
	for _, op := range cmdOperators {
		deferFn := op(&cmd)
		if deferFn != nil {
			defer deferFn()
		}
	}
	appendDocker(&cmd)
	if err := validateArgs(cmd.Command...); err != nil {
		return &icmd.Result{
			Error: err,
		}
	}
	return icmd.RunCmd(cmd)
}

// validateArgs is a checker to ensure tests are not running commands which are
// not supported on platforms. Specifically on Windows this is 'busybox top'.
func validateArgs(args ...string) error {
	if testEnv.DaemonPlatform() != "windows" {
		return nil
	}
	foundBusybox := -1
	for key, value := range args {
		if strings.ToLower(value) == "busybox" {
			foundBusybox = key
		}
		if (foundBusybox != -1) && (key == foundBusybox+1) && (strings.ToLower(value) == "top") {
			return errors.New("cannot use 'busybox top' in tests on Windows. Use runSleepingContainer()")
		}
	}
	return nil
}

// Build executes the specified docker build command
func Build(name string) icmd.Cmd {
	return icmd.Command("build", "-t", name)
}

// Inspect executes the specified docker inspect command
func Inspect(name string) icmd.Cmd {
	return icmd.Command("inspect", name)
}

// Format sets the specified format with --format flag
func Format(format string) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Command = append(
			[]string{cmd.Command[0]},
			append([]string{"--format", fmt.Sprintf("{{%s}}", format)}, cmd.Command[1:]...)...,
		)
		return nil
	}
}

func appendDocker(cmd *icmd.Cmd) {
	cmd.Command = append([]string{testEnv.DockerBinary()}, cmd.Command...)
}

// Args build an icmd.Cmd struct from the specified arguments
func Args(args ...string) icmd.Cmd {
	switch len(args) {
	case 0:
		return icmd.Cmd{}
	case 1:
		return icmd.Command(args[0])
	default:
		return icmd.Command(args[0], args[1:]...)
	}
}

// Daemon points to the specified daemon
func Daemon(d *daemon.Daemon) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Command = append([]string{"--host", d.Sock()}, cmd.Command...)
		return nil
	}
}

// WithTimeout sets the timeout for the command to run
func WithTimeout(timeout time.Duration) func(cmd *icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Timeout = timeout
		return nil
	}
}

// WithEnvironmentVariables sets the specified environment variables for the command to run
func WithEnvironmentVariables(envs ...string) func(cmd *icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Env = envs
		return nil
	}
}

// WithFlags sets the specified flags for the command to run
func WithFlags(flags ...string) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Command = append(cmd.Command, flags...)
		return nil
	}
}

// InDir sets the folder in which the command should be executed
func InDir(path string) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Dir = path
		return nil
	}
}

// WithStdout sets the standard output writer of the command
func WithStdout(writer io.Writer) func(*icmd.Cmd) func() {
	return func(cmd *icmd.Cmd) func() {
		cmd.Stdout = writer
		return nil
	}
}
