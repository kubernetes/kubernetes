package main

import (
	"bytes"
	"os/exec"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestLoginWithoutTTY(c *check.C) {
	cmd := exec.Command(dockerBinary, "login")

	// Send to stdin so the process does not get the TTY
	cmd.Stdin = bytes.NewBufferString("buffer test string \n")

	// run the command and block until it's done
	if err := cmd.Run(); err == nil {
		c.Fatal("Expected non nil err when loginning in & TTY not available")
	}

}
