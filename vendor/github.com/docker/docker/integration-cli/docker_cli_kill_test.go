package main

import (
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestKillContainer(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)
	c.Assert(waitRun(cleanedContainerID), check.IsNil)

	dockerCmd(c, "kill", cleanedContainerID)

	out, _ = dockerCmd(c, "ps", "-q")
	if strings.Contains(out, cleanedContainerID) {
		c.Fatal("killed container is still running")
	}
}

func (s *DockerSuite) TestKillofStoppedContainer(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "stop", cleanedContainerID)

	_, _, err := dockerCmdWithError(c, "kill", "-s", "30", cleanedContainerID)
	c.Assert(err, check.Not(check.IsNil), check.Commentf("Container %s is not running", cleanedContainerID))
}

func (s *DockerSuite) TestKillDifferentUserContainer(c *check.C) {
	out, _ := dockerCmd(c, "run", "-u", "daemon", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)
	c.Assert(waitRun(cleanedContainerID), check.IsNil)

	dockerCmd(c, "kill", cleanedContainerID)

	out, _ = dockerCmd(c, "ps", "-q")
	if strings.Contains(out, cleanedContainerID) {
		c.Fatal("killed container is still running")
	}
}

// regression test about correct signal parsing see #13665
func (s *DockerSuite) TestKillWithSignal(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	dockerCmd(c, "kill", "-s", "SIGWINCH", cid)

	running, _ := inspectField(cid, "State.Running")
	if running != "true" {
		c.Fatal("Container should be in running state after SIGWINCH")
	}
}

func (s *DockerSuite) TestKillWithInvalidSignal(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	out, _, err := dockerCmdWithError(c, "kill", "-s", "0", cid)
	c.Assert(err, check.NotNil)
	if !strings.ContainsAny(out, "Invalid signal: 0") {
		c.Fatal("Kill with an invalid signal didn't error out correctly")
	}

	running, _ := inspectField(cid, "State.Running")
	if running != "true" {
		c.Fatal("Container should be in running state after an invalid signal")
	}

	out, _ = dockerCmd(c, "run", "-d", "busybox", "top")
	cid = strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	out, _, err = dockerCmdWithError(c, "kill", "-s", "SIG42", cid)
	c.Assert(err, check.NotNil)
	if !strings.ContainsAny(out, "Invalid signal: SIG42") {
		c.Fatal("Kill with an invalid signal error out correctly")
	}

	running, _ = inspectField(cid, "State.Running")
	if running != "true" {
		c.Fatal("Container should be in running state after an invalid signal")
	}
}
