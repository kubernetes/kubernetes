// +build !windows

package main

import (
	"bufio"
	"os/exec"
	"strings"
	"time"

	"github.com/docker/docker/pkg/stringid"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

// #9860
func (s *DockerSuite) TestAttachClosedOnContainerStop(c *check.C) {

	out, _ := dockerCmd(c, "run", "-dti", "busybox", "sleep", "2")

	id := strings.TrimSpace(out)
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	errChan := make(chan error)
	go func() {
		defer close(errChan)

		_, tty, err := pty.Open()
		if err != nil {
			errChan <- err
			return
		}
		attachCmd := exec.Command(dockerBinary, "attach", id)
		attachCmd.Stdin = tty
		attachCmd.Stdout = tty
		attachCmd.Stderr = tty

		if err := attachCmd.Run(); err != nil {
			errChan <- err
			return
		}
	}()

	dockerCmd(c, "wait", id)

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(attachWait):
		c.Fatal("timed out without attach returning")
	}

}

func (s *DockerSuite) TestAttachAfterDetach(c *check.C) {

	name := "detachtest"

	cpty, tty, err := pty.Open()
	if err != nil {
		c.Fatalf("Could not open pty: %v", err)
	}
	cmd := exec.Command(dockerBinary, "run", "-ti", "--name", name, "busybox")
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty

	errChan := make(chan error)
	go func() {
		errChan <- cmd.Run()
		close(errChan)
	}()

	time.Sleep(500 * time.Millisecond)
	if err := waitRun(name); err != nil {
		c.Fatal(err)
	}
	cpty.Write([]byte{16})
	time.Sleep(100 * time.Millisecond)
	cpty.Write([]byte{17})

	select {
	case err := <-errChan:
		c.Assert(err, check.IsNil)
	case <-time.After(5 * time.Second):
		c.Fatal("timeout while detaching")
	}

	cpty, tty, err = pty.Open()
	if err != nil {
		c.Fatalf("Could not open pty: %v", err)
	}

	cmd = exec.Command(dockerBinary, "attach", name)
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty

	if err := cmd.Start(); err != nil {
		c.Fatal(err)
	}

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

	if err := cmd.Wait(); err != nil {
		c.Fatal(err)
	}

	if !strings.Contains(string(bytes[:nBytes]), "/ #") {
		c.Fatalf("failed to get a new prompt. got %s", string(bytes[:nBytes]))
	}

}

// TestAttachDetach checks that attach in tty mode can be detached using the long container ID
func (s *DockerSuite) TestAttachDetach(c *check.C) {
	out, _ := dockerCmd(c, "run", "-itd", "busybox", "cat")
	id := strings.TrimSpace(out)
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	cpty, tty, err := pty.Open()
	if err != nil {
		c.Fatal(err)
	}
	defer cpty.Close()

	cmd := exec.Command(dockerBinary, "attach", id)
	cmd.Stdin = tty
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		c.Fatal(err)
	}
	defer stdout.Close()
	if err := cmd.Start(); err != nil {
		c.Fatal(err)
	}
	if err := waitRun(id); err != nil {
		c.Fatalf("error waiting for container to start: %v", err)
	}

	if _, err := cpty.Write([]byte("hello\n")); err != nil {
		c.Fatal(err)
	}
	out, err = bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	if strings.TrimSpace(out) != "hello" {
		c.Fatalf("expected 'hello', got %q", out)
	}

	// escape sequence
	if _, err := cpty.Write([]byte{16}); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write([]byte{17}); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	running, err := inspectField(id, "State.Running")
	if err != nil {
		c.Fatal(err)
	}
	if running != "true" {
		c.Fatal("expected container to still be running")
	}

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
	if err := waitRun(id); err != nil {
		c.Fatal(err)
	}

	cpty, tty, err := pty.Open()
	if err != nil {
		c.Fatal(err)
	}
	defer cpty.Close()

	cmd := exec.Command(dockerBinary, "attach", id)
	cmd.Stdin = tty
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		c.Fatal(err)
	}
	defer stdout.Close()
	if err := cmd.Start(); err != nil {
		c.Fatal(err)
	}

	if _, err := cpty.Write([]byte("hello\n")); err != nil {
		c.Fatal(err)
	}
	out, err = bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		c.Fatal(err)
	}
	if strings.TrimSpace(out) != "hello" {
		c.Fatalf("expected 'hello', got %q", out)
	}

	// escape sequence
	if _, err := cpty.Write([]byte{16}); err != nil {
		c.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	if _, err := cpty.Write([]byte{17}); err != nil {
		c.Fatal(err)
	}

	ch := make(chan struct{})
	go func() {
		cmd.Wait()
		ch <- struct{}{}
	}()

	running, err := inspectField(id, "State.Running")
	if err != nil {
		c.Fatal(err)
	}
	if running != "true" {
		c.Fatal("expected container to still be running")
	}

	go func() {
		dockerCmd(c, "kill", id)
	}()

	select {
	case <-ch:
	case <-time.After(10 * time.Millisecond):
		c.Fatal("timed out waiting for container to exit")
	}

}
