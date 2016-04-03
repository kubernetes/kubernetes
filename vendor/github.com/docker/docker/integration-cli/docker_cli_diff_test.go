package main

import (
	"strings"

	"github.com/go-check/check"
)

// ensure that an added file shows up in docker diff
func (s *DockerSuite) TestDiffFilenameShownInOutput(c *check.C) {
	containerCmd := `echo foo > /root/bar`
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", containerCmd)

	cleanCID := strings.TrimSpace(out)
	out, _ = dockerCmd(c, "diff", cleanCID)

	found := false
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains("A /root/bar", line) {
			found = true
			break
		}
	}
	if !found {
		c.Errorf("couldn't find the new file in docker diff's output: %v", out)
	}
}

// test to ensure GH #3840 doesn't occur any more
func (s *DockerSuite) TestDiffEnsureDockerinitFilesAreIgnored(c *check.C) {
	// this is a list of files which shouldn't show up in `docker diff`
	dockerinitFiles := []string{"/etc/resolv.conf", "/etc/hostname", "/etc/hosts", "/.dockerinit", "/.dockerenv"}
	containerCount := 5

	// we might not run into this problem from the first run, so start a few containers
	for i := 0; i < containerCount; i++ {
		containerCmd := `echo foo > /root/bar`
		out, _ := dockerCmd(c, "run", "-d", "busybox", "sh", "-c", containerCmd)

		cleanCID := strings.TrimSpace(out)
		out, _ = dockerCmd(c, "diff", cleanCID)

		for _, filename := range dockerinitFiles {
			if strings.Contains(out, filename) {
				c.Errorf("found file which should've been ignored %v in diff output", filename)
			}
		}
	}
}

func (s *DockerSuite) TestDiffEnsureOnlyKmsgAndPtmx(c *check.C) {
	out, _ := dockerCmd(c, "run", "-d", "busybox", "sleep", "0")

	cleanCID := strings.TrimSpace(out)
	out, _ = dockerCmd(c, "diff", cleanCID)

	expected := map[string]bool{
		"C /dev":         true,
		"A /dev/full":    true, // busybox
		"C /dev/ptmx":    true, // libcontainer
		"A /dev/kmsg":    true, // lxc
		"A /dev/fd":      true,
		"A /dev/fuse":    true,
		"A /dev/ptmx":    true,
		"A /dev/null":    true,
		"A /dev/random":  true,
		"A /dev/stdout":  true,
		"A /dev/stderr":  true,
		"A /dev/tty1":    true,
		"A /dev/stdin":   true,
		"A /dev/tty":     true,
		"A /dev/urandom": true,
		"A /dev/zero":    true,
	}

	for _, line := range strings.Split(out, "\n") {
		if line != "" && !expected[line] {
			c.Errorf("%q is shown in the diff but shouldn't", line)
		}
	}
}

// https://github.com/docker/docker/pull/14381#discussion_r33859347
func (s *DockerSuite) TestDiffEmptyArgClientError(c *check.C) {
	out, _, err := dockerCmdWithError(c, "diff", "")
	c.Assert(err, check.NotNil)
	c.Assert(strings.TrimSpace(out), check.Equals, "Container name cannot be empty")
}
