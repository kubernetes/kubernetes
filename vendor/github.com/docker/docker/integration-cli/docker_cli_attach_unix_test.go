// +build !windows

package main

import (
	"bufio"
	"io/ioutil"
	"os/exec"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/pkg/stringid"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

// #9860 Make sure attach ends when container ends (with no errors)
func (s *DockerSuite) TestAttachClosedOnContainerStop(c *check.C) {
	testRequires(c, SameHostDaemon)

	out, _ := dockerCmd(c, "run", "-dti", "busybox", "/bin/sh", "-c", `trap 'exit 0' SIGTERM; while true; do sleep 1; done`)

	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	pty, tty, err := pty.Open()
	c.Assert(err, check.IsNil)

	attachCmd := exec.Command(dockerBinary, "attach", id)
	attachCmd.Stdin = tty
	attachCmd.Stdout = tty
	attachCmd.Stderr = tty
	err = attachCmd.Start()
	c.Assert(err, check.IsNil)

	errChan := make(chan error)
	go func() {
		time.Sleep(300 * time.Millisecond)
		defer close(errChan)
		// Container is waiting for us to signal it to stop
		dockerCmd(c, "stop", id)
		// And wait for the attach command to end
		errChan <- attachCmd.Wait()
	}()

	// Wait for the docker to end (should be done by the
	// stop command in the go routine)
	dockerCmd(c, "wait", id)

	select {
	case err := <-errChan:
		tty.Close()
		out, _ := ioutil.ReadAll(pty)
		c.Assert(err, check.IsNil, check.Commentf("out: %v", string(out)))
	case <-time.After(attachWait):
		c.Fatal("timed out without attach returning")
	}

}

func (s *DockerSuite) TestAttachAfterDetach(c *check.C) {
	name := "detachtest"

	cpty, tty, err := pty.Open()
	c.Assert(err, checker.IsNil, check.Commentf("Could not open pty: %v", err))
	cmd := exec.Command(dockerBinary, "run", "-ti", "--name", name, "busybox")
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty

	errChan := make(chan error)
	go func() {
		errChan <- cmd.Run()
		close(errChan)
	}()

	c.Assert(waitRun(name), check.IsNil)

	cpty.Write([]byte{16})
	time.Sleep(100 * time.Millisecond)
	cpty.Write([]byte{17})

	select {
	case err := <-errChan:
		if err != nil {
			buff := make([]byte, 200)
			tty.Read(buff)
			c.Fatalf("%s: %s", err, buff)
		}
	case <-time.After(5 * time.Second):
		c.Fatal("timeout while detaching")
	}

	cpty, tty, err = pty.Open()
	c.Assert(err, checker.IsNil, check.Commentf("Could not open pty: %v", err))

	cmd = exec.Command(dockerBinary, "attach", name)
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty

	err = cmd.Start()
	c.Assert(err, checker.IsNil)

	bytes := make([]byte, 10)
	var nBytes int
	readErr := make(chan error, 1)

	go func() {
		time.Sleep(500 * time.Millisecond)
		cpty.Write([]byte("\n"))
		time.Sleep(500 * time.Millisecond)

		nBytes, err = cpty.Read(bytes)
		cpty.Close()
		readErr <- err
	}()

	select {
	case err := <-readErr:
		c.Assert(err, check.IsNil)
	case <-time.After(2 * time.Second):
		c.Fatal("timeout waiting for attach read")
	}

	err = cmd.Wait()
	c.Assert(err, checker.IsNil)

	c.Assert(string(bytes[:nBytes]), checker.Contains, "/ #")

}

// TestAttachDetach checks that attach in tty mode can be detached using the long container ID
func (s *DockerSuite) TestAttachDetach(c *check.C) {
	out, _ := dockerCmd(c, "run", "-itd", "busybox", "cat")
	id := strings.TrimSpace(out)
	c.Assert(waitRun(id), check.IsNil)

	cpty, tty, err := pty.Open()
	c.Assert(err, check.IsNil)
	defer cpty.Close()

	cmd := exec.Command(dockerBinary, "attach", id)
	cmd.Stdin = tty
	stdout, err := cmd.StdoutPipe()
	c.Assert(err, check.IsNil)
	defer stdout.Close()
	err = cmd.Start()
	c.Assert(err, check.IsNil)
	c.Assert(waitRun(id), check.IsNil)

	_, err = cpty.Write([]byte("hello\n"))
	c.Assert(err, check.IsNil)
	out, err = bufio.NewReader(stdout).ReadString('\n')
	c.Assert(err, check.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello", check.Commentf("expected 'hello', got %q", out))

	// escape sequence
	_, err = cpty.Write([]byte{16})
	c.Assert(err, checker.IsNil)
	time.Sleep(100 * time.Millisecond)
	_, err = cpty.Write([]byte{17})
	c.Assert(err, checker.IsNil)

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	running := inspectField(c, id, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))

	go func() {
		dockerCmd(c, "kill", id)
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Millisecond):
		c.Fatal("timed out waiting for container to exit")
	}

}

// TestAttachDetachTruncatedID checks that attach in tty mode can be detached
func (s *DockerSuite) TestAttachDetachTruncatedID(c *check.C) {
	out, _ := dockerCmd(c, "run", "-itd", "busybox", "cat")
	id := stringid.TruncateID(strings.TrimSpace(out))
	c.Assert(waitRun(id), check.IsNil)

	cpty, tty, err := pty.Open()
	c.Assert(err, checker.IsNil)
	defer cpty.Close()

	cmd := exec.Command(dockerBinary, "attach", id)
	cmd.Stdin = tty
	stdout, err := cmd.StdoutPipe()
	c.Assert(err, checker.IsNil)
	defer stdout.Close()
	err = cmd.Start()
	c.Assert(err, checker.IsNil)

	_, err = cpty.Write([]byte("hello\n"))
	c.Assert(err, checker.IsNil)
	out, err = bufio.NewReader(stdout).ReadString('\n')
	c.Assert(err, checker.IsNil)
	c.Assert(strings.TrimSpace(out), checker.Equals, "hello", check.Commentf("expected 'hello', got %q", out))

	// escape sequence
	_, err = cpty.Write([]byte{16})
	c.Assert(err, checker.IsNil)
	time.Sleep(100 * time.Millisecond)
	_, err = cpty.Write([]byte{17})
	c.Assert(err, checker.IsNil)

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	running := inspectField(c, id, "State.Running")
	c.Assert(running, checker.Equals, "true", check.Commentf("expected container to still be running"))

	go func() {
		dockerCmd(c, "kill", id)
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Millisecond):
		c.Fatal("timed out waiting for container to exit")
	}

}
