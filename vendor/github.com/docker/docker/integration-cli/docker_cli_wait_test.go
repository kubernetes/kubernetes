package main

import (
	"bytes"
	"os/exec"
	"strings"
	"time"

	"github.com/go-check/check"
)

// non-blocking wait with 0 exit code
func (s *DockerSuite) TestWaitNonBlockedExitZero(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", "true")
	containerID := strings.TrimSpace(out)

	status := "true"
	var err error
	for i := 0; status != "false"; i++ {
		status, err = inspectField(containerID, "State.Running")
		c.Assert(err, check.IsNil)

		time.Sleep(time.Second)
		if i >= 60 {
			c.Fatal("Container should have stopped by now")
		}
	}

	out, _ = dockerCmd(c, "wait", containerID)
	if strings.TrimSpace(out) != "0" {
		c.Fatal("failed to set up container", out)
	}

}

// blocking wait with 0 exit code
func (s *DockerSuite) TestWaitBlockedExitZero(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "trap 'exit 0' TERM; while true; do sleep 0.01; done")
	containerID := strings.TrimSpace(out)

	if err := waitRun(containerID); err != nil {
		c.Fatal(err)
	}

	chWait := make(chan string)
	go func() {
		out, _, _ := runCommandWithOutput(exec.Command(dockerBinary, "wait", containerID))
		chWait <- out
	}()

	time.Sleep(100 * time.Millisecond)
	dockerCmd(c, "stop", containerID)

	select {
	case status := <-chWait:
		if strings.TrimSpace(status) != "0" {
			c.Fatalf("expected exit 0, got %s", status)
		}
	case <-time.After(2 * time.Second):
		c.Fatal("timeout waiting for `docker wait` to exit")
	}

}

// non-blocking wait with random exit code
func (s *DockerSuite) TestWaitNonBlockedExitRandom(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", "exit 99")
	containerID := strings.TrimSpace(out)

	status := "true"
	var err error
	for i := 0; status != "false"; i++ {
		status, err = inspectField(containerID, "State.Running")
		c.Assert(err, check.IsNil)

		time.Sleep(time.Second)
		if i >= 60 {
			c.Fatal("Container should have stopped by now")
		}
	}

	out, _ = dockerCmd(c, "wait", containerID)
	if strings.TrimSpace(out) != "99" {
		c.Fatal("failed to set up container", out)
	}

}

// blocking wait with random exit code
func (s *DockerSuite) TestWaitBlockedExitRandom(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "/bin/sh", "-c", "trap 'exit 99' TERM; while true; do sleep 0.01; done")
	containerID := strings.TrimSpace(out)
	if err := waitRun(containerID); err != nil {
		c.Fatal(err)
	}
	if err := waitRun(containerID); err != nil {
		c.Fatal(err)
	}

	chWait := make(chan error)
	waitCmd := exec.Command(dockerBinary, "wait", containerID)
	waitCmdOut := bytes.NewBuffer(nil)
	waitCmd.Stdout = waitCmdOut
	if err := waitCmd.Start(); err != nil {
		c.Fatal(err)
	}

	go func() {
		chWait <- waitCmd.Wait()
	}()

	dockerCmd(c, "stop", containerID)

	select {
	case err := <-chWait:
		if err != nil {
			c.Fatal(err)
		}
		status, err := waitCmdOut.ReadString('\n')
		if err != nil {
			c.Fatal(err)
		}
		if strings.TrimSpace(status) != "99" {
			c.Fatalf("expected exit 99, got %s", status)
		}
	case <-time.After(2 * time.Second):
		waitCmd.Process.Kill()
		c.Fatal("timeout waiting for `docker wait` to exit")
	}
}
