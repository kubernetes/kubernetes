package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/pkg/timeutils"
	"github.com/go-check/check"
)

// This used to work, it test a log of PageSize-1 (gh#4851)
func (s *DockerSuite) TestLogsContainerSmallerThanPage(c *check.C) {
	testLen := 32767
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo -n =; done; echo", testLen))
	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "wait", cleanedContainerID)
	out, _ = dockerCmd(c, "logs", cleanedContainerID)
	if len(out) != testLen+1 {
		c.Fatalf("Expected log length of %d, received %d\n", testLen+1, len(out))
	}
}

// Regression test: When going over the PageSize, it used to panic (gh#4851)
func (s *DockerSuite) TestLogsContainerBiggerThanPage(c *check.C) {
	testLen := 32768
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo -n =; done; echo", testLen))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "logs", cleanedContainerID)

	if len(out) != testLen+1 {
		c.Fatalf("Expected log length of %d, received %d\n", testLen+1, len(out))
	}
}

// Regression test: When going much over the PageSize, it used to block (gh#4851)
func (s *DockerSuite) TestLogsContainerMuchBiggerThanPage(c *check.C) {
	testLen := 33000
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo -n =; done; echo", testLen))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "logs", cleanedContainerID)

	if len(out) != testLen+1 {
		c.Fatalf("Expected log length of %d, received %d\n", testLen+1, len(out))
	}
}

func (s *DockerSuite) TestLogsTimestamps(c *check.C) {
	testLen := 100
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo =; done;", testLen))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "logs", "-t", cleanedContainerID)

	lines := strings.Split(out, "\n")

	if len(lines) != testLen+1 {
		c.Fatalf("Expected log %d lines, received %d\n", testLen+1, len(lines))
	}

	ts := regexp.MustCompile(`^.* `)

	for _, l := range lines {
		if l != "" {
			_, err := time.Parse(timeutils.RFC3339NanoFixed+" ", ts.FindString(l))
			if err != nil {
				c.Fatalf("Failed to parse timestamp from %v: %v", l, err)
			}
			if l[29] != 'Z' { // ensure we have padded 0's
				c.Fatalf("Timestamp isn't padded properly: %s", l)
			}
		}
	}
}

func (s *DockerSuite) TestLogsSeparateStderr(c *check.C) {
	msg := "stderr_log"
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("echo %s 1>&2", msg))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	stdout, stderr, _ := dockerCmdWithStdoutStderr(c, "logs", cleanedContainerID)

	if stdout != "" {
		c.Fatalf("Expected empty stdout stream, got %v", stdout)
	}

	stderr = strings.TrimSpace(stderr)
	if stderr != msg {
		c.Fatalf("Expected %v in stderr stream, got %v", msg, stderr)
	}
}

func (s *DockerSuite) TestLogsStderrInStdout(c *check.C) {
	msg := "stderr_log"
	out, _ := dockerCmd(c, "run", "-d", "-t", "busybox", "sh", "-c", fmt.Sprintf("echo %s 1>&2", msg))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	stdout, stderr, _ := dockerCmdWithStdoutStderr(c, "logs", cleanedContainerID)
	if stderr != "" {
		c.Fatalf("Expected empty stderr stream, got %v", stderr)
	}

	stdout = strings.TrimSpace(stdout)
	if stdout != msg {
		c.Fatalf("Expected %v in stdout stream, got %v", msg, stdout)
	}
}

func (s *DockerSuite) TestLogsTail(c *check.C) {
	testLen := 100
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", fmt.Sprintf("for i in $(seq 1 %d); do echo =; done;", testLen))

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "logs", "--tail", "5", cleanedContainerID)

	lines := strings.Split(out, "\n")

	if len(lines) != 6 {
		c.Fatalf("Expected log %d lines, received %d\n", 6, len(lines))
	}
	out, _ = dockerCmd(c, "logs", "--tail", "all", cleanedContainerID)

	lines = strings.Split(out, "\n")

	if len(lines) != testLen+1 {
		c.Fatalf("Expected log %d lines, received %d\n", testLen+1, len(lines))
	}
	out, _, _ = dockerCmdWithStdoutStderr(c, "logs", "--tail", "random", cleanedContainerID)

	lines = strings.Split(out, "\n")

	if len(lines) != testLen+1 {
		c.Fatalf("Expected log %d lines, received %d\n", testLen+1, len(lines))
	}
}

func (s *DockerSuite) TestLogsFollowStopped(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "echo", "hello")

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	logsCmd := exec.Command(dockerBinary, "logs", "-f", cleanedContainerID)
	if err := logsCmd.Start(); err != nil {
		c.Fatal(err)
	}

	errChan := make(chan error)
	go func() {
		errChan <- logsCmd.Wait()
		close(errChan)
	}()

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(1 * time.Second):
		c.Fatal("Following logs is hanged")
	}
}

func (s *DockerSuite) TestLogsSince(c *check.C) {
	name := "testlogssince"
	out, _ := dockerCmd(c, "run", "--name="+name, "busybox", "/bin/sh", "-c", "for i in $(seq 1 3); do sleep 2; echo `date +%s` log$i; done")

	log2Line := strings.Split(strings.Split(out, "\n")[1], " ")
	t, err := strconv.ParseInt(log2Line[0], 10, 64) // the timestamp log2 is writen
	c.Assert(err, check.IsNil)
	since := t + 1 // add 1s so log1 & log2 doesn't show up
	out, _ = dockerCmd(c, "logs", "-t", fmt.Sprintf("--since=%v", since), name)

	// Skip 2 seconds
	unexpected := []string{"log1", "log2"}
	for _, v := range unexpected {
		if strings.Contains(out, v) {
			c.Fatalf("unexpected log message returned=%v, since=%v\nout=%v", v, since, out)
		}
	}
	// Test with default value specified and parameter omitted
	expected := []string{"log1", "log2", "log3"}
	for _, cmd := range []*exec.Cmd{
		exec.Command(dockerBinary, "logs", "-t", name),
		exec.Command(dockerBinary, "logs", "-t", "--since=0", name),
	} {
		out, _, err = runCommandWithOutput(cmd)
		if err != nil {
			c.Fatalf("failed to log container: %s, %v", out, err)
		}
		for _, v := range expected {
			if !strings.Contains(out, v) {
				c.Fatalf("'%v' does not contain=%v\nout=%s", cmd.Args, v, out)
			}
		}
	}
}

func (s *DockerSuite) TestLogsSinceFutureFollow(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", `for i in $(seq 1 5); do date +%s; sleep 1; done`)
	cleanedContainerID := strings.TrimSpace(out)

	now := daemonTime(c).Unix()
	since := now + 2
	out, _ = dockerCmd(c, "logs", "-f", fmt.Sprintf("--since=%v", since), cleanedContainerID)
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 {
		c.Fatal("got no log lines")
	}
	for _, v := range lines {
		ts, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			c.Fatalf("cannot parse timestamp output from log: '%v'\nout=%s", v, out)
		}
		if ts < since {
			c.Fatalf("earlier log found. since=%v logdate=%v", since, ts)
		}
	}
}

// Regression test for #8832
func (s *DockerSuite) TestLogsFollowSlowStdoutConsumer(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", `usleep 200000;yes X | head -c 200000`)

	cleanedContainerID := strings.TrimSpace(out)

	stopSlowRead := make(chan bool)

	go func() {
		exec.Command(dockerBinary, "wait", cleanedContainerID).Run()
		stopSlowRead <- true
	}()

	logCmd := exec.Command(dockerBinary, "logs", "-f", cleanedContainerID)

	stdout, err := logCmd.StdoutPipe()
	c.Assert(err, check.IsNil)

	if err := logCmd.Start(); err != nil {
		c.Fatal(err)
	}

	// First read slowly
	bytes1, err := consumeWithSpeed(stdout, 10, 50*time.Millisecond, stopSlowRead)
	c.Assert(err, check.IsNil)

	// After the container has finished we can continue reading fast
	bytes2, err := consumeWithSpeed(stdout, 32*1024, 0, nil)
	c.Assert(err, check.IsNil)

	actual := bytes1 + bytes2
	expected := 200000
	if actual != expected {
		c.Fatalf("Invalid bytes read: %d, expected %d", actual, expected)
	}

}

func (s *DockerSuite) TestLogsFollowGoroutinesWithStdout(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true; do echo hello; sleep 2; done")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	type info struct {
		NGoroutines int
	}
	getNGoroutines := func() int {
		var i info
		status, b, err := sockRequest("GET", "/info", nil)
		c.Assert(err, check.IsNil)
		c.Assert(status, check.Equals, 200)
		c.Assert(json.Unmarshal(b, &i), check.IsNil)
		return i.NGoroutines
	}

	nroutines := getNGoroutines()

	cmd := exec.Command(dockerBinary, "logs", "-f", id)
	r, w := io.Pipe()
	cmd.Stdout = w
	c.Assert(cmd.Start(), check.IsNil)

	// Make sure pipe is written to
	chErr := make(chan error)
	go func() {
		b := make([]byte, 1)
		_, err := r.Read(b)
		chErr <- err
	}()
	c.Assert(<-chErr, check.IsNil)
	c.Assert(cmd.Process.Kill(), check.IsNil)

	// NGoroutines is not updated right away, so we need to wait before failing
	t := time.After(30 * time.Second)
	for {
		select {
		case <-t:
			if n := getNGoroutines(); n > nroutines {
				c.Fatalf("leaked goroutines: expected less than or equal to %d, got: %d", nroutines, n)
			}
		default:
			if n := getNGoroutines(); n <= nroutines {
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}

func (s *DockerSuite) TestLogsFollowGoroutinesNoOutput(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "while true; do sleep 2; done")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	type info struct {
		NGoroutines int
	}
	getNGoroutines := func() int {
		var i info
		status, b, err := sockRequest("GET", "/info", nil)
		c.Assert(err, check.IsNil)
		c.Assert(status, check.Equals, 200)
		c.Assert(json.Unmarshal(b, &i), check.IsNil)
		return i.NGoroutines
	}

	nroutines := getNGoroutines()

	cmd := exec.Command(dockerBinary, "logs", "-f", id)
	c.Assert(cmd.Start(), check.IsNil)
	time.Sleep(200 * time.Millisecond)
	c.Assert(cmd.Process.Kill(), check.IsNil)

	// NGoroutines is not updated right away, so we need to wait before failing
	t := time.After(30 * time.Second)
	for {
		select {
		case <-t:
			if n := getNGoroutines(); n > nroutines {
				c.Fatalf("leaked goroutines: expected less than or equal to %d, got: %d", nroutines, n)
			}
		default:
			if n := getNGoroutines(); n <= nroutines {
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}
