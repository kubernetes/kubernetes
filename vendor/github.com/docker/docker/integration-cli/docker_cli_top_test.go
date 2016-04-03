package main

import (
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestTopMultipleArgs(c *check.C) {
	out, _ := dockerCmd(c, "run", "-i", "-d", "busybox", "top")

	cleanedContainerID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "top", cleanedContainerID, "-o", "pid")
	if !strings.Contains(out, "PID") {
		c.Fatalf("did not see PID after top -o pid: %s", out)
	}

}

func (s *DockerSuite) TestTopNonPrivileged(c *check.C) {
	out, _ := dockerCmd(c, "run", "-i", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)

	out1, _ := dockerCmd(c, "top", cleanedContainerID)
	out2, _ := dockerCmd(c, "top", cleanedContainerID)
	out, _ = dockerCmd(c, "kill", cleanedContainerID)

	if !strings.Contains(out1, "top") && !strings.Contains(out2, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed twice")
	} else if !strings.Contains(out1, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed the first time")
	} else if !strings.Contains(out2, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed the second itime")
	}

}

func (s *DockerSuite) TestTopPrivileged(c *check.C) {
	out, _ := dockerCmd(c, "run", "--privileged", "-i", "-d", "busybox", "top")
	cleanedContainerID := strings.TrimSpace(out)

	out1, _ := dockerCmd(c, "top", cleanedContainerID)
	out2, _ := dockerCmd(c, "top", cleanedContainerID)
	out, _ = dockerCmd(c, "kill", cleanedContainerID)

	if !strings.Contains(out1, "top") && !strings.Contains(out2, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed twice")
	} else if !strings.Contains(out1, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed the first time")
	} else if !strings.Contains(out2, "top") {
		c.Fatal("top should've listed `top` in the process list, but failed the second itime")
	}

}
