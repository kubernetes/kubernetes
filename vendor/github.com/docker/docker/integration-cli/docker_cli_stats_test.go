package main

import (
	"bufio"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestStatsNoStream(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	statsCmd := exec.Command(dockerBinary, "stats", "--no-stream", id)
	type output struct {
		out []byte
		err error
	}

	ch := make(chan output)
	go func() {
		out, err := statsCmd.Output()
		ch <- output{out, err}
	}()

	select {
	case outerr := <-ch:
		c.Assert(outerr.err, checker.IsNil, check.Commentf("Error running stats: %v", outerr.err))
		c.Assert(string(outerr.out), checker.Contains, id) //running container wasn't present in output
	case <-time.After(3 * time.Second):
		statsCmd.Process.Kill()
		c.Fatalf("stats did not return immediately when not streaming")
	}
}

func (s *DockerSuite) TestStatsContainerNotFound(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)

	out, _, err := dockerCmdWithError("stats", "notfound")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "No such container: notfound", check.Commentf("Expected to fail on not found container stats, got %q instead", out))

	out, _, err = dockerCmdWithError("stats", "--no-stream", "notfound")
	c.Assert(err, checker.NotNil)
	c.Assert(out, checker.Contains, "No such container: notfound", check.Commentf("Expected to fail on not found container stats with --no-stream, got %q instead", out))
}

func (s *DockerSuite) TestStatsAllRunningNoStream(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id1 := strings.TrimSpace(out)[:12]
	c.Assert(waitRun(id1), check.IsNil)
	out, _ = dockerCmd(c, "run", "-d", "busybox", "top")
	id2 := strings.TrimSpace(out)[:12]
	c.Assert(waitRun(id2), check.IsNil)
	out, _ = dockerCmd(c, "run", "-d", "busybox", "top")
	id3 := strings.TrimSpace(out)[:12]
	c.Assert(waitRun(id3), check.IsNil)
	dockerCmd(c, "stop", id3)

	out, _ = dockerCmd(c, "stats", "--no-stream")
	if !strings.Contains(out, id1) || !strings.Contains(out, id2) {
		c.Fatalf("Expected stats output to contain both %s and %s, got %s", id1, id2, out)
	}
	if strings.Contains(out, id3) {
		c.Fatalf("Did not expect %s in stats, got %s", id3, out)
	}

	// check output contains real data, but not all zeros
	reg, _ := regexp.Compile("[1-9]+")
	// split output with "\n", outLines[1] is id2's output
	// outLines[2] is id1's output
	outLines := strings.Split(out, "\n")
	// check stat result of id2 contains real data
	realData := reg.Find([]byte(outLines[1][12:]))
	c.Assert(realData, checker.NotNil, check.Commentf("stat result are empty: %s", out))
	// check stat result of id1 contains real data
	realData = reg.Find([]byte(outLines[2][12:]))
	c.Assert(realData, checker.NotNil, check.Commentf("stat result are empty: %s", out))
}

func (s *DockerSuite) TestStatsAllNoStream(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)

	out, _ := dockerCmd(c, "run", "-d", "busybox", "top")
	id1 := strings.TrimSpace(out)[:12]
	c.Assert(waitRun(id1), check.IsNil)
	dockerCmd(c, "stop", id1)
	out, _ = dockerCmd(c, "run", "-d", "busybox", "top")
	id2 := strings.TrimSpace(out)[:12]
	c.Assert(waitRun(id2), check.IsNil)

	out, _ = dockerCmd(c, "stats", "--all", "--no-stream")
	if !strings.Contains(out, id1) || !strings.Contains(out, id2) {
		c.Fatalf("Expected stats output to contain both %s and %s, got %s", id1, id2, out)
	}

	// check output contains real data, but not all zeros
	reg, _ := regexp.Compile("[1-9]+")
	// split output with "\n", outLines[1] is id2's output
	outLines := strings.Split(out, "\n")
	// check stat result of id2 contains real data
	realData := reg.Find([]byte(outLines[1][12:]))
	c.Assert(realData, checker.NotNil, check.Commentf("stat result of %s is empty: %s", id2, out))
	// check stat result of id1 contains all zero
	realData = reg.Find([]byte(outLines[2][12:]))
	c.Assert(realData, checker.IsNil, check.Commentf("stat result of %s should be empty : %s", id1, out))
}

func (s *DockerSuite) TestStatsAllNewContainersAdded(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)

	id := make(chan string)
	addedChan := make(chan struct{})

	runSleepingContainer(c, "-d")
	statsCmd := exec.Command(dockerBinary, "stats")
	stdout, err := statsCmd.StdoutPipe()
	c.Assert(err, check.IsNil)
	c.Assert(statsCmd.Start(), check.IsNil)
	defer statsCmd.Process.Kill()

	go func() {
		containerID := <-id
		matchID := regexp.MustCompile(containerID)

		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			switch {
			case matchID.MatchString(scanner.Text()):
				close(addedChan)
				return
			}
		}
	}()

	out := runSleepingContainer(c, "-d")
	c.Assert(waitRun(strings.TrimSpace(out)), check.IsNil)
	id <- strings.TrimSpace(out)[:12]

	select {
	case <-time.After(30 * time.Second):
		c.Fatal("failed to observe new container created added to stats")
	case <-addedChan:
		// ignore, done
	}
}

func (s *DockerSuite) TestStatsFormatAll(c *check.C) {
	// Windows does not support stats
	testRequires(c, DaemonIsLinux)

	cli.DockerCmd(c, "run", "-d", "--name=RunningOne", "busybox", "top")
	cli.WaitRun(c, "RunningOne")
	cli.DockerCmd(c, "run", "-d", "--name=ExitedOne", "busybox", "top")
	cli.DockerCmd(c, "stop", "ExitedOne")
	cli.WaitExited(c, "ExitedOne", 5*time.Second)

	out := cli.DockerCmd(c, "stats", "--no-stream", "--format", "{{.Name}}").Combined()
	c.Assert(out, checker.Contains, "RunningOne")
	c.Assert(out, checker.Not(checker.Contains), "ExitedOne")

	out = cli.DockerCmd(c, "stats", "--all", "--no-stream", "--format", "{{.Name}}").Combined()
	c.Assert(out, checker.Contains, "RunningOne")
	c.Assert(out, checker.Contains, "ExitedOne")
}
