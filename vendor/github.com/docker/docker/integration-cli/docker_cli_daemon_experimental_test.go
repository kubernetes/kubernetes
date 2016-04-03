// +build daemon,experimental

package main

import (
	"os/exec"
	"strings"

	"github.com/go-check/check"
)

func assertNetwork(c *check.C, d *Daemon, name string) {
	out, err := d.Cmd("network", "ls")
	c.Assert(err, check.IsNil)
	lines := strings.Split(out, "\n")
	for i := 1; i < len(lines)-1; i++ {
		if strings.Contains(lines[i], name) {
			return
		}
	}
	c.Fatalf("Network %s not found in network ls o/p", name)
}

func (s *DockerDaemonSuite) TestDaemonDefaultNetwork(c *check.C) {
	d := s.d

	networkName := "testdefault"
	err := d.StartWithBusybox("--default-network", "bridge:"+networkName)
	c.Assert(err, check.IsNil)

	_, err = d.Cmd("run", "busybox", "true")
	c.Assert(err, check.IsNil)

	assertNetwork(c, d, networkName)

	ifconfigCmd := exec.Command("ifconfig", networkName)
	_, _, _, err = runCommandWithStdoutStderr(ifconfigCmd)
	c.Assert(err, check.IsNil)
}
