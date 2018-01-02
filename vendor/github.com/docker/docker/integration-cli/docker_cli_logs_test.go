package main

import (
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/testutil"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

// This used to work, it test a log of PageSize-1 (gh#4851)
func (s *DockerSuite) TestLogsContainerSmallerThanPage(c *check.C) {
	testLogsContainerPagination(c, 32767)
}

// Regression test: When going over the PageSize, it used to panic (gh#4851)
func (s *DockerSuite) TestLogsContainerBiggerThanPage(c *check.C) {
	testLogsContainerPagination(c, 32768)
}

// Regression test: When going much over the PageSize, it used to block (gh#4851)
func (s *DockerSuite) TestLogsContainerMuchBiggerThanPage(c *check.C) {
	testLogsContainerPagination(c, 33000)
}

func testLogsContainerPagination(c *check.C, testLen int) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo -n = >> a.a; done; echo >> a.a; cat a.a", testLen))
	id := strings.TrimSpace(out)
	dockerCmd(c, "wait", id)
	out, _ = dockerCmd(c, "logs", id)
	c.Assert(out, checker.HasLen, testLen+1)
}

func (s *DockerSuite) TestLogsTimestamps(c *check.C) {
	testLen := 100
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo = >> a.a; done; cat a.a", testLen))

	id := strings.TrimSpace(out)
	dockerCmd(c, "wait", id)

	out, _ = dockerCmd(c, "logs", "-t", id)

	lines := strings.Split(out, "\n")

	c.Assert(lines, checker.HasLen, testLen+1)

	ts := regexp.MustCompile(`^.* `)

	for _, l := range lines {
		if l != "" {
			_, err := time.Parse(jsonlog.RFC3339NanoFixed+" ", ts.FindString(l))
			c.Assert(err, checker.IsNil, check.Commentf("Failed to parse timestamp from %v", l))
			// ensure we have padded 0's
			c.Assert(l[29], checker.Equals, uint8('Z'))
		}
	}
}

func (s *DockerSuite) TestLogsSeparateStderr(c *check.C) {
	msg := "stderr_log"
	out := cli.DockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("echo %s 1>&2", msg)).Combined()
	id := strings.TrimSpace(out)
	cli.DockerCmd(c, "wait", id)
	cli.DockerCmd(c, "logs", id).Assert(c, icmd.Expected{
		Out: "",
		Err: msg,
	})
}

func (s *DockerSuite) TestLogsStderrInStdout(c *check.C) {
	// TODO Windows: Needs investigation why this fails. Obtained string includes
	// a bunch of ANSI escape sequences before the "stderr_log" message.
	testRequires(c, DaemonIsLinux)
	msg := "stderr_log"
	out := cli.DockerCmd(c, "run", "-d", "-t", "busybox", "sh", "-c", fmt.Sprintf("echo %s 1>&2", msg)).Combined()
	id := strings.TrimSpace(out)
	cli.DockerCmd(c, "wait", id)

	cli.DockerCmd(c, "logs", id).Assert(c, icmd.Expected{
		Out: msg,
		Err: "",
	})
}

func (s *DockerSuite) TestLogsTail(c *check.C) {
	testLen := 100
	out := cli.DockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo =; done;", testLen)).Combined()

	id := strings.TrimSpace(out)
	cli.DockerCmd(c, "wait", id)

	out = cli.DockerCmd(c, "logs", "--tail", "0", id).Combined()
	lines := strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, 1)

	out = cli.DockerCmd(c, "logs", "--tail", "5", id).Combined()
	lines = strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, 6)

	out = cli.DockerCmd(c, "logs", "--tail", "99", id).Combined()
	lines = strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, 100)

	out = cli.DockerCmd(c, "logs", "--tail", "all", id).Combined()
	lines = strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, testLen+1)

	out = cli.DockerCmd(c, "logs", "--tail", "-1", id).Combined()
	lines = strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, testLen+1)

	out = cli.DockerCmd(c, "logs", "--tail", "random", id).Combined()
	lines = strings.Split(out, "\n")
	c.Assert(lines, checker.HasLen, testLen+1)
}

func (s *DockerSuite) TestLogsFollowStopped(c *check.C) {
	dockerCmd(c, "run", "--name=test", "busybox", "echo", "hello")
	id := getIDByName(c, "test")

	logsCmd := exec.Command(dockerBinary, "logs", "-f", id)
	c.Assert(logsCmd.Start(), checker.IsNil)

	errChan := make(chan error)
	go func() {
		errChan <- logsCmd.Wait()
		close(errChan)
	}()

	select {
	case err := <-errChan:
		c.Assert(err, checker.IsNil)
	case <-time.After(30 * time.Second):
		c.Fatal("Following logs is hanged")
	}
}

func (s *DockerSuite) TestLogsSince(c *check.C) {
	name := "testlogssince"
	dockerCmd(c, "run", "--name="+name, "busybox", "/bin/sh", "-c", "for i in $(seq 1 3); do sleep 2; echo log$i; done")
	out, _ := dockerCmd(c, "logs", "-t", name)

	log2Line := strings.Split(strings.Split(out, "\n")[1], " ")
	t, err := time.Parse(time.RFC3339Nano, log2Line[0]) // the timestamp log2 is written
	c.Assert(err, checker.IsNil)
	since := t.Unix() + 1 // add 1s so log1 & log2 doesn't show up
	out, _ = dockerCmd(c, "logs", "-t", fmt.Sprintf("--since=%v", since), name)

	// Skip 2 seconds
	unexpected := []string{"log1", "log2"}
	for _, v := range unexpected {
		c.Assert(out, checker.Not(checker.Contains), v, check.Commentf("unexpected log message returned, since=%v", since))
	}

	// Test to make sure a bad since format is caught by the client
	out, _, _ = dockerCmdWithError("logs", "-t", "--since=2006-01-02T15:04:0Z", name)
	c.Assert(out, checker.Contains, "cannot parse \"0Z\" as \"05\"", check.Commentf("bad since format passed to server"))

	// Test with default value specified and parameter omitted
	expected := []string{"log1", "log2", "log3"}
	for _, cmd := range [][]string{
		{"logs", "-t", name},
		{"logs", "-t", "--since=0", name},
	} {
		result := icmd.RunCommand(dockerBinary, cmd...)
		result.Assert(c, icmd.Success)
		for _, v := range expected {
			c.Assert(result.Combined(), checker.Contains, v)
		}
	}
}

func (s *DockerSuite) TestLogsSinceFutureFollow(c *check.C) {
	// TODO Windows TP5 - Figure out why this test is so flakey. Disabled for now.
	testRequires(c, DaemonIsLinux)
	name := "testlogssincefuturefollow"
	out, _ := dockerCmd(c, "run", "-d", "--name", name, "busybox", "/bin/sh", "-c", `for i in $(seq 1 5); do echo log$i; sleep 1; done`)

	// Extract one timestamp from the log file to give us a starting point for
	// our `--since` argument. Because the log producer runs in the background,
	// we need to check repeatedly for some output to be produced.
	var timestamp string
	for i := 0; i != 100 && timestamp == ""; i++ {
		if out, _ = dockerCmd(c, "logs", "-t", name); out == "" {
			time.Sleep(time.Millisecond * 100) // Retry
		} else {
			timestamp = strings.Split(strings.Split(out, "\n")[0], " ")[0]
		}
	}

	c.Assert(timestamp, checker.Not(checker.Equals), "")
	t, err := time.Parse(time.RFC3339Nano, timestamp)
	c.Assert(err, check.IsNil)

	since := t.Unix() + 2
	out, _ = dockerCmd(c, "logs", "-t", "-f", fmt.Sprintf("--since=%v", since), name)
	c.Assert(out, checker.Not(checker.HasLen), 0, check.Commentf("cannot read from empty log"))
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for _, v := range lines {
		ts, err := time.Parse(time.RFC3339Nano, strings.Split(v, " ")[0])
		c.Assert(err, checker.IsNil, check.Commentf("cannot parse timestamp output from log: '%v'", v))
		c.Assert(ts.Unix() >= since, checker.Equals, true, check.Commentf("earlier log found. since=%v logdate=%v", since, ts))
	}
}

// Regression test for #8832
func (s *DockerSuite) TestLogsFollowSlowStdoutConsumer(c *check.C) {
	// TODO Windows: Fix this test for TP5.
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", `usleep 600000;yes X | head -c 200000`)

	id := strings.TrimSpace(out)

	stopSlowRead := make(chan bool)

	go func() {
		exec.Command(dockerBinary, "wait", id).Run()
		stopSlowRead <- true
	}()

	logCmd := exec.Command(dockerBinary, "logs", "-f", id)
	stdout, err := logCmd.StdoutPipe()
	c.Assert(err, checker.IsNil)
	c.Assert(logCmd.Start(), checker.IsNil)

	// First read slowly
	bytes1, err := testutil.ConsumeWithSpeed(stdout, 10, 50*time.Millisecond, stopSlowRead)
	c.Assert(err, checker.IsNil)

	// After the container has finished we can continue reading fast
	bytes2, err := testutil.ConsumeWithSpeed(stdout, 32*1024, 0, nil)
	c.Assert(err, checker.IsNil)

	actual := bytes1 + bytes2
	expected := 200000
	c.Assert(actual, checker.Equals, expected)
}

func (s *DockerSuite) TestLogsFollowGoroutinesWithStdout(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true; do echo hello; sleep 2; done")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	nroutines, err := getGoroutineNumber()
	c.Assert(err, checker.IsNil)
	cmd := exec.Command(dockerBinary, "logs", "-f", id)
	r, w := io.Pipe()
	cmd.Stdout = w
	c.Assert(cmd.Start(), checker.IsNil)

	// Make sure pipe is written to
	chErr := make(chan error)
	go func() {
		b := make([]byte, 1)
		_, err := r.Read(b)
		chErr <- err
	}()
	c.Assert(<-chErr, checker.IsNil)
	c.Assert(cmd.Process.Kill(), checker.IsNil)
	r.Close()
	// NGoroutines is not updated right away, so we need to wait before failing
	c.Assert(waitForGoroutines(nroutines), checker.IsNil)
}

func (s *DockerSuite) TestLogsFollowGoroutinesNoOutput(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true; do sleep 2; done")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), checker.IsNil)

	nroutines, err := getGoroutineNumber()
	c.Assert(err, checker.IsNil)
	cmd := exec.Command(dockerBinary, "logs", "-f", id)
	c.Assert(cmd.Start(), checker.IsNil)
	time.Sleep(200 * time.Millisecond)
	c.Assert(cmd.Process.Kill(), checker.IsNil)

	// NGoroutines is not updated right away, so we need to wait before failing
	c.Assert(waitForGoroutines(nroutines), checker.IsNil)
}

func (s *DockerSuite) TestLogsCLIContainerNotFound(c *check.C) {
	name := "testlogsnocontainer"
	out, _, _ := dockerCmdWithError("logs", name)
	message := fmt.Sprintf("No such container: %s\n", name)
	c.Assert(out, checker.Contains, message)
}

func (s *DockerSuite) TestLogsWithDetails(c *check.C) {
	dockerCmd(c, "run", "--name=test", "--label", "foo=bar", "-e", "baz=qux", "--log-opt", "labels=foo", "--log-opt", "env=baz", "busybox", "echo", "hello")
	out, _ := dockerCmd(c, "logs", "--details", "--timestamps", "test")

	logFields := strings.Fields(strings.TrimSpace(out))
	c.Assert(len(logFields), checker.Equals, 3, check.Commentf(out))

	details := strings.Split(logFields[1], ",")
	c.Assert(details, checker.HasLen, 2)
	c.Assert(details[0], checker.Equals, "baz=qux")
	c.Assert(details[1], checker.Equals, "foo=bar")
}
