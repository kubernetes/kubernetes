package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/request"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestKillContainer(c *check.C) {
	out := runSleepingContainer(c, "-d")
	cleanedContainerID := strings.TrimSpace(out)
	cli.WaitRun(c, cleanedContainerID)

	cli.DockerCmd(c, "kill", cleanedContainerID)
	cli.WaitExited(c, cleanedContainerID, 10*time.Second)

	out = cli.DockerCmd(c, "ps", "-q").Combined()
	c.Assert(out, checker.Not(checker.Contains), cleanedContainerID, check.Commentf("killed container is still running"))

}

func (s *DockerSuite) TestKillOffStoppedContainer(c *check.C) {
	out := runSleepingContainer(c, "-d")
	cleanedContainerID := strings.TrimSpace(out)

	cli.DockerCmd(c, "stop", cleanedContainerID)
	cli.WaitExited(c, cleanedContainerID, 10*time.Second)

	cli.Docker(cli.Args("kill", "-s", "30", cleanedContainerID)).Assert(c, icmd.Expected{
		ExitCode: 1,
	})
}

func (s *DockerSuite) TestKillDifferentUserContainer(c *check.C) {
	// TODO Windows: Windows does not yet support -u (Feb 2016).
	testRequires(c, DaemonIsLinux)
	out := cli.DockerCmd(c, "run", "-u", "daemon", "-d", "busybox", "top").Combined()
	cleanedContainerID := strings.TrimSpace(out)
	cli.WaitRun(c, cleanedContainerID)

	cli.DockerCmd(c, "kill", cleanedContainerID)
	cli.WaitExited(c, cleanedContainerID, 10*time.Second)

	out = cli.DockerCmd(c, "ps", "-q").Combined()
	c.Assert(out, checker.Not(checker.Contains), cleanedContainerID, check.Commentf("killed container is still running"))

}

// regression test about correct signal parsing see #13665
func (s *DockerSuite) TestKillWithSignal(c *check.C) {
	// Cannot port to Windows - does not support signals in the same way Linux does
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	dockerCmd(c, "kill", "-s", "SIGWINCH", cid)
	time.Sleep(250 * time.Millisecond)

	running := inspectField(c, cid, "State.Running")

	c.Assert(running, checker.Equals, "true", check.Commentf("Container should be in running state after SIGWINCH"))
}

func (s *DockerSuite) TestKillWithStopSignalWithSameSignalShouldDisableRestartPolicy(c *check.C) {
	// Cannot port to Windows - does not support signals int the same way as Linux does
	testRequires(c, DaemonIsLinux)
	out := cli.DockerCmd(c, "run", "-d", "--stop-signal=TERM", "--restart=always", "busybox", "top").Combined()
	cid := strings.TrimSpace(out)
	cli.WaitRun(c, cid)

	// Let docker send a TERM signal to the container
	// It will kill the process and disable the restart policy
	cli.DockerCmd(c, "kill", "-s", "TERM", cid)
	cli.WaitExited(c, cid, 10*time.Second)

	out = cli.DockerCmd(c, "ps", "-q").Combined()
	c.Assert(out, checker.Not(checker.Contains), cid, check.Commentf("killed container is still running"))
}

func (s *DockerSuite) TestKillWithStopSignalWithDifferentSignalShouldKeepRestartPolicy(c *check.C) {
	// Cannot port to Windows - does not support signals int the same way as Linux does
	testRequires(c, DaemonIsLinux)
	out := cli.DockerCmd(c, "run", "-d", "--stop-signal=CONT", "--restart=always", "busybox", "top").Combined()
	cid := strings.TrimSpace(out)
	cli.WaitRun(c, cid)

	// Let docker send a TERM signal to the container
	// It will kill the process, but not disable the restart policy
	cli.DockerCmd(c, "kill", "-s", "TERM", cid)
	cli.WaitRestart(c, cid, 10*time.Second)

	// Restart policy should still be in place, so it should be still running
	cli.WaitRun(c, cid)
}

// FIXME(vdemeester) should be a unit test
func (s *DockerSuite) TestKillWithInvalidSignal(c *check.C) {
	out := runSleepingContainer(c, "-d")
	cid := strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	out, _, err := dockerCmdWithError("kill", "-s", "0", cid)
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Invalid signal: 0", check.Commentf("Kill with an invalid signal didn't error out correctly"))

	running := inspectField(c, cid, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("Container should be in running state after an invalid signal"))

	out = runSleepingContainer(c, "-d")
	cid = strings.TrimSpace(out)
	c.Assert(waitRun(cid), check.IsNil)

	out, _, err = dockerCmdWithError("kill", "-s", "SIG42", cid)
	c.Assert(err, check.NotNil)
	c.Assert(out, checker.Contains, "Invalid signal: SIG42", check.Commentf("Kill with an invalid signal error out correctly"))

	running = inspectField(c, cid, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("Container should be in running state after an invalid signal"))

}

func (s *DockerSuite) TestKillStoppedContainerAPIPre120(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows only supports 1.25 or later
	runSleepingContainer(c, "--name", "docker-kill-test-api", "-d")
	dockerCmd(c, "stop", "docker-kill-test-api")

	status, _, err := request.SockRequest("POST", fmt.Sprintf("/v1.19/containers/%s/kill", "docker-kill-test-api"), nil, daemonHost())
	c.Assert(err, check.IsNil)
	c.Assert(status, check.Equals, http.StatusNoContent)
}
