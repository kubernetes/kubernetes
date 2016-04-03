// +build !windows

package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"unicode"

	"github.com/go-check/check"
	"github.com/kr/pty"
)

// #5979
func (s *DockerSuite) TestEventsRedirectStdout(c *check.C) {
	since := daemonTime(c).Unix()
	dockerCmd(c, "run", "busybox", "true")

	file, err := ioutil.TempFile("", "")
	if err != nil {
		c.Fatalf("could not create temp file: %v", err)
	}
	defer os.Remove(file.Name())

	command := fmt.Sprintf("%s events --since=%d --until=%d > %s", dockerBinary, since, daemonTime(c).Unix(), file.Name())
	_, tty, err := pty.Open()
	if err != nil {
		c.Fatalf("Could not open pty: %v", err)
	}
	cmd := exec.Command("sh", "-c", command)
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty
	if err := cmd.Run(); err != nil {
		c.Fatalf("run err for command %q: %v", command, err)
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		for _, ch := range scanner.Text() {
			if unicode.IsControl(ch) {
				c.Fatalf("found control character %v", []byte(string(ch)))
			}
		}
	}
	if err := scanner.Err(); err != nil {
		c.Fatalf("Scan err for command %q: %v", command, err)
	}

}
