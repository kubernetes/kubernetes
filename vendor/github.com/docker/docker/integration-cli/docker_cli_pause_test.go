package main

import (
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestPause(c *check.C) {
	testRequires(c, IsPausable)

	name := "testeventpause"
	runSleepingContainer(c, "-d", "--name", name)

	cli.DockerCmd(c, "pause", name)
	pausedContainers := strings.Fields(
		cli.DockerCmd(c, "ps", "-f", "status=paused", "-q", "-a").Combined(),
	)
	c.Assert(len(pausedContainers), checker.Equals, 1)

	cli.DockerCmd(c, "unpause", name)

	out := cli.DockerCmd(c, "events", "--since=0", "--until", daemonUnixTime(c)).Combined()
	events := strings.Split(strings.TrimSpace(out), "\n")
	actions := eventActionsByIDAndType(c, events, name, "container")

	c.Assert(actions[len(actions)-2], checker.Equals, "pause")
	c.Assert(actions[len(actions)-1], checker.Equals, "unpause")
}

func (s *DockerSuite) TestPauseMultipleContainers(c *check.C) {
	testRequires(c, IsPausable)

	containers := []string{
		"testpausewithmorecontainers1",
		"testpausewithmorecontainers2",
	}
	for _, name := range containers {
		runSleepingContainer(c, "-d", "--name", name)
	}
	cli.DockerCmd(c, append([]string{"pause"}, containers...)...)
	pausedContainers := strings.Fields(
		cli.DockerCmd(c, "ps", "-f", "status=paused", "-q", "-a").Combined(),
	)
	c.Assert(len(pausedContainers), checker.Equals, len(containers))

	cli.DockerCmd(c, append([]string{"unpause"}, containers...)...)

	out := cli.DockerCmd(c, "events", "--since=0", "--until", daemonUnixTime(c)).Combined()
	events := strings.Split(strings.TrimSpace(out), "\n")

	for _, name := range containers {
		actions := eventActionsByIDAndType(c, events, name, "container")

		c.Assert(actions[len(actions)-2], checker.Equals, "pause")
		c.Assert(actions[len(actions)-1], checker.Equals, "unpause")
	}
}

func (s *DockerSuite) TestPauseFailsOnWindowsServerContainers(c *check.C) {
	testRequires(c, DaemonIsWindows, NotPausable)
	runSleepingContainer(c, "-d", "--name=test")
	out, _, _ := dockerCmdWithError("pause", "test")
	c.Assert(out, checker.Contains, "cannot pause Windows Server Containers")
}

func (s *DockerSuite) TestStopPausedContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)

	id := runSleepingContainer(c)
	cli.WaitRun(c, id)
	cli.DockerCmd(c, "pause", id)
	cli.DockerCmd(c, "stop", id)
	cli.WaitForInspectResult(c, id, "{{.State.Running}}", "false", 30*time.Second)
}
