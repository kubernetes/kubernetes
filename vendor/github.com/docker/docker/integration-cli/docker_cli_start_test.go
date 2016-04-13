package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/go-check/check"
)

// Regression test for https://github.com/docker/docker/issues/7843
func (s *DockerSuite) TestStartAttachReturnsOnError(c *check.C) {
	dockerCmd(c, "run", "-d", "--name", "test", "busybox")
	dockerCmd(c, "wait", "test")

	// Expect this to fail because the above container is stopped, this is what we want
	if _, _, err := dockerCmdWithError(c, "run", "-d", "--name", "test2", "--link", "test:test", "busybox"); err == nil {
		c.Fatal("Expected error but got none")
	}

	ch := make(chan error)
	go func() {
		// Attempt to start attached to the container that won't start
		// This should return an error immediately since the container can't be started
		if _, _, err := dockerCmdWithError(c, "start", "-a", "test2"); err == nil {
			ch <- fmt.Errorf("Expected error but got none")
		}
		close(ch)
	}()

	select {
	case err := <-ch:
		c.Assert(err, check.IsNil)
	case <-time.After(time.Second):
		c.Fatalf("Attach did not exit properly")
	}
}

// gh#8555: Exit code should be passed through when using start -a
func (s *DockerSuite) TestStartAttachCorrectExitCode(c *check.C) {
	out, _, _ := dockerCmdWithStdoutStderr(c, "run", "-d", "busybox", "sh", "-c", "sleep 2; exit 1")
	out = strings.TrimSpace(out)

	// make sure the container has exited before trying the "start -a"
	dockerCmd(c, "wait", out)

	startOut, exitCode, err := dockerCmdWithError(c, "start", "-a", out)
	if err != nil && !strings.Contains("exit status 1", fmt.Sprintf("%s", err)) {
		c.Fatalf("start command failed unexpectedly with error: %v, output: %q", err, startOut)
	}
	if exitCode != 1 {
		c.Fatalf("start -a did not respond with proper exit code: expected 1, got %d", exitCode)
	}

}

func (s *DockerSuite) TestStartAttachSilent(c *check.C) {
	name := "teststartattachcorrectexitcode"
	dockerCmd(c, "run", "--name", name, "busybox", "echo", "test")

	// make sure the container has exited before trying the "start -a"
	dockerCmd(c, "wait", name)

	startOut, _ := dockerCmd(c, "start", "-a", name)
	if expected := "test\n"; startOut != expected {
		c.Fatalf("start -a produced unexpected output: expected %q, got %q", expected, startOut)
	}
}

func (s *DockerSuite) TestStartRecordError(c *check.C) {

	// when container runs successfully, we should not have state.Error
	dockerCmd(c, "run", "-d", "-p", "9999:9999", "--name", "test", "busybox", "top")
	stateErr, err := inspectField("test", "State.Error")
	c.Assert(err, check.IsNil)
	if stateErr != "" {
		c.Fatalf("Expected to not have state error but got state.Error(%q)", stateErr)
	}

	// Expect this to fail and records error because of ports conflict
	out, _, err := dockerCmdWithError(c, "run", "-d", "--name", "test2", "-p", "9999:9999", "busybox", "top")
	if err == nil {
		c.Fatalf("Expected error but got none, output %q", out)
	}

	stateErr, err = inspectField("test2", "State.Error")
	c.Assert(err, check.IsNil)
	expected := "port is already allocated"
	if stateErr == "" || !strings.Contains(stateErr, expected) {
		c.Fatalf("State.Error(%q) does not include %q", stateErr, expected)
	}

	// Expect the conflict to be resolved when we stop the initial container
	dockerCmd(c, "stop", "test")
	dockerCmd(c, "start", "test2")
	stateErr, err = inspectField("test2", "State.Error")
	c.Assert(err, check.IsNil)
	if stateErr != "" {
		c.Fatalf("Expected to not have state error but got state.Error(%q)", stateErr)
	}
}

func (s *DockerSuite) TestStartPausedContainer(c *check.C) {
	defer unpauseAllContainers()

	dockerCmd(c, "run", "-d", "--name", "testing", "busybox", "top")

	dockerCmd(c, "pause", "testing")

	if out, _, err := dockerCmdWithError(c, "start", "testing"); err == nil || !strings.Contains(out, "Cannot start a paused container, try unpause instead.") {
		c.Fatalf("an error should have been shown that you cannot start paused container: %s\n%v", out, err)
	}
}

func (s *DockerSuite) TestStartMultipleContainers(c *check.C) {
	// run a container named 'parent' and create two container link to `parent`
	dockerCmd(c, "run", "-d", "--name", "parent", "busybox", "top")

	for _, container := range []string{"child_first", "child_second"} {
		dockerCmd(c, "create", "--name", container, "--link", "parent:parent", "busybox", "top")
	}

	// stop 'parent' container
	dockerCmd(c, "stop", "parent")

	out, err := inspectField("parent", "State.Running")
	c.Assert(err, check.IsNil)
	if out != "false" {
		c.Fatal("Container should be stopped")
	}

	// start all the three containers, container `child_first` start first which should be failed
	// container 'parent' start second and then start container 'child_second'
	out, _, err = dockerCmdWithError(c, "start", "child_first", "parent", "child_second")
	if !strings.Contains(out, "Cannot start container child_first") || err == nil {
		c.Fatal("Expected error but got none")
	}

	for container, expected := range map[string]string{"parent": "true", "child_first": "false", "child_second": "true"} {
		out, err := inspectField(container, "State.Running")
		c.Assert(err, check.IsNil)
		if out != expected {
			c.Fatal("Container running state wrong")
		}

	}
}

func (s *DockerSuite) TestStartAttachMultipleContainers(c *check.C) {
	// run  multiple containers to test
	for _, container := range []string{"test1", "test2", "test3"} {
		dockerCmd(c, "run", "-d", "--name", container, "busybox", "top")
	}

	// stop all the containers
	for _, container := range []string{"test1", "test2", "test3"} {
		dockerCmd(c, "stop", container)
	}

	// test start and attach multiple containers at once, expected error
	for _, option := range []string{"-a", "-i", "-ai"} {
		out, _, err := dockerCmdWithError(c, "start", option, "test1", "test2", "test3")
		if !strings.Contains(out, "You cannot start and attach multiple containers at once.") || err == nil {
			c.Fatal("Expected error but got none")
		}
	}

	// confirm the state of all the containers be stopped
	for container, expected := range map[string]string{"test1": "false", "test2": "false", "test3": "false"} {
		out, err := inspectField(container, "State.Running")
		if err != nil {
			c.Fatal(out, err)
		}
		if out != expected {
			c.Fatal("Container running state wrong")
		}
	}
}
