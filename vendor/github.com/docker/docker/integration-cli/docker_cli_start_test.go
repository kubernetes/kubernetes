package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

// Regression test for https://github.com/docker/docker/issues/7843
func (s *DockerSuite) TestStartAttachReturnsOnError(c *check.C) {
	// Windows does not support link
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "--name", "test", "busybox")

	// Expect this to fail because the above container is stopped, this is what we want
	out, _, err := dockerCmdWithError("run", "--name", "test2", "--link", "test:test", "busybox")
	// err shouldn't be nil because container test2 try to link to stopped container
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))

	ch := make(chan error)
	go func() {
		// Attempt to start attached to the container that won't start
		// This should return an error immediately since the container can't be started
		if out, _, err := dockerCmdWithError("start", "-a", "test2"); err == nil {
			ch <- fmt.Errorf("Expected error but got none:\n%s", out)
		}
		close(ch)
	}()

	select {
	case err := <-ch:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatalf("Attach did not exit properly")
	}
}

// gh#8555: Exit code should be passed through when using start -a
func (s *DockerSuite) TestStartAttachCorrectExitCode(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out := cli.DockerCmd(c, "run", "-d", "busybox", "sh", "-c", "sleep 2; exit 1").Stdout()
	out = strings.TrimSpace(out)

	// make sure the container has exited before trying the "start -a"
	cli.DockerCmd(c, "wait", out)

	cli.Docker(cli.Args("start", "-a", out)).Assert(c, icmd.Expected{
		ExitCode: 1,
	})
}

func (s *DockerSuite) TestStartAttachSilent(c *check.C) {
	name := "teststartattachcorrectexitcode"
	dockerCmd(c, "run", "--name", name, "busybox", "echo", "test")

	// make sure the container has exited before trying the "start -a"
	dockerCmd(c, "wait", name)

	startOut, _ := dockerCmd(c, "start", "-a", name)
	// start -a produced unexpected output
	c.Assert(startOut, checker.Equals, "test\n")
}

func (s *DockerSuite) TestStartRecordError(c *check.C) {
	// TODO Windows CI: Requires further porting work. Should be possible.
	testRequires(c, DaemonIsLinux)
	// when container runs successfully, we should not have state.Error
	dockerCmd(c, "run", "-d", "-p", "9999:9999", "--name", "test", "busybox", "top")
	stateErr := inspectField(c, "test", "State.Error")
	// Expected to not have state error
	c.Assert(stateErr, checker.Equals, "")

	// Expect this to fail and records error because of ports conflict
	out, _, err := dockerCmdWithError("run", "-d", "--name", "test2", "-p", "9999:9999", "busybox", "top")
	// err shouldn't be nil because docker run will fail
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))

	stateErr = inspectField(c, "test2", "State.Error")
	c.Assert(stateErr, checker.Contains, "port is already allocated")

	// Expect the conflict to be resolved when we stop the initial container
	dockerCmd(c, "stop", "test")
	dockerCmd(c, "start", "test2")
	stateErr = inspectField(c, "test2", "State.Error")
	// Expected to not have state error but got one
	c.Assert(stateErr, checker.Equals, "")
}

func (s *DockerSuite) TestStartPausedContainer(c *check.C) {
	// Windows does not support pausing containers
	testRequires(c, IsPausable)

	runSleepingContainer(c, "-d", "--name", "testing")

	dockerCmd(c, "pause", "testing")

	out, _, err := dockerCmdWithError("start", "testing")
	// an error should have been shown that you cannot start paused container
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
	// an error should have been shown that you cannot start paused container
	c.Assert(out, checker.Contains, "Cannot start a paused container, try unpause instead.")
}

func (s *DockerSuite) TestStartMultipleContainers(c *check.C) {
	// Windows does not support --link
	testRequires(c, DaemonIsLinux)
	// run a container named 'parent' and create two container link to `parent`
	dockerCmd(c, "run", "-d", "--name", "parent", "busybox", "top")

	for _, container := range []string{"child_first", "child_second"} {
		dockerCmd(c, "create", "--name", container, "--link", "parent:parent", "busybox", "top")
	}

	// stop 'parent' container
	dockerCmd(c, "stop", "parent")

	out := inspectField(c, "parent", "State.Running")
	// Container should be stopped
	c.Assert(out, checker.Equals, "false")

	// start all the three containers, container `child_first` start first which should be failed
	// container 'parent' start second and then start container 'child_second'
	expOut := "Cannot link to a non running container"
	expErr := "failed to start containers: [child_first]"
	out, _, err := dockerCmdWithError("start", "child_first", "parent", "child_second")
	// err shouldn't be nil because start will fail
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
	// output does not correspond to what was expected
	if !(strings.Contains(out, expOut) || strings.Contains(err.Error(), expErr)) {
		c.Fatalf("Expected out: %v with err: %v  but got out: %v with err: %v", expOut, expErr, out, err)
	}

	for container, expected := range map[string]string{"parent": "true", "child_first": "false", "child_second": "true"} {
		out := inspectField(c, container, "State.Running")
		// Container running state wrong
		c.Assert(out, checker.Equals, expected)
	}
}

func (s *DockerSuite) TestStartAttachMultipleContainers(c *check.C) {
	// run  multiple containers to test
	for _, container := range []string{"test1", "test2", "test3"} {
		runSleepingContainer(c, "--name", container)
	}

	// stop all the containers
	for _, container := range []string{"test1", "test2", "test3"} {
		dockerCmd(c, "stop", container)
	}

	// test start and attach multiple containers at once, expected error
	for _, option := range []string{"-a", "-i", "-ai"} {
		out, _, err := dockerCmdWithError("start", option, "test1", "test2", "test3")
		// err shouldn't be nil because start will fail
		c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
		// output does not correspond to what was expected
		c.Assert(out, checker.Contains, "you cannot start and attach multiple containers at once")
	}

	// confirm the state of all the containers be stopped
	for container, expected := range map[string]string{"test1": "false", "test2": "false", "test3": "false"} {
		out := inspectField(c, container, "State.Running")
		// Container running state wrong
		c.Assert(out, checker.Equals, expected)
	}
}

// Test case for #23716
func (s *DockerSuite) TestStartAttachWithRename(c *check.C) {
	testRequires(c, DaemonIsLinux)
	cli.DockerCmd(c, "create", "-t", "--name", "before", "busybox")
	go func() {
		cli.WaitRun(c, "before")
		cli.DockerCmd(c, "rename", "before", "after")
		cli.DockerCmd(c, "stop", "--time=2", "after")
	}()
	// FIXME(vdemeester) the intent is not clear and potentially racey
	result := cli.Docker(cli.Args("start", "-a", "before")).Assert(c, icmd.Expected{
		ExitCode: 137,
	})
	c.Assert(result.Stderr(), checker.Not(checker.Contains), "No such container")
}

func (s *DockerSuite) TestStartReturnCorrectExitCode(c *check.C) {
	dockerCmd(c, "create", "--restart=on-failure:2", "--name", "withRestart", "busybox", "sh", "-c", "exit 11")
	dockerCmd(c, "create", "--rm", "--name", "withRm", "busybox", "sh", "-c", "exit 12")

	_, exitCode, err := dockerCmdWithError("start", "-a", "withRestart")
	c.Assert(err, checker.NotNil)
	c.Assert(exitCode, checker.Equals, 11)
	_, exitCode, err = dockerCmdWithError("start", "-a", "withRm")
	c.Assert(err, checker.NotNil)
	c.Assert(exitCode, checker.Equals, 12)
}
