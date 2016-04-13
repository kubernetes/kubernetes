package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestPortList(c *check.C) {

	// one port
	out, _ := dockerCmd(c, "run", "-d", "-p", "9876:80", "busybox", "top")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	if !assertPortList(c, out, []string{"0.0.0.0:9876"}) {
		c.Error("Port list is not correct")
	}

	out, _ = dockerCmd(c, "port", firstID)

	if !assertPortList(c, out, []string{"80/tcp -> 0.0.0.0:9876"}) {
		c.Error("Port list is not correct")
	}
	dockerCmd(c, "rm", "-f", firstID)

	// three port
	out, _ = dockerCmd(c, "run", "-d",
		"-p", "9876:80",
		"-p", "9877:81",
		"-p", "9878:82",
		"busybox", "top")
	ID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", ID, "80")

	if !assertPortList(c, out, []string{"0.0.0.0:9876"}) {
		c.Error("Port list is not correct")
	}

	out, _ = dockerCmd(c, "port", ID)

	if !assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:9876",
		"81/tcp -> 0.0.0.0:9877",
		"82/tcp -> 0.0.0.0:9878"}) {
		c.Error("Port list is not correct")
	}
	dockerCmd(c, "rm", "-f", ID)

	// more and one port mapped to the same container port
	out, _ = dockerCmd(c, "run", "-d",
		"-p", "9876:80",
		"-p", "9999:80",
		"-p", "9877:81",
		"-p", "9878:82",
		"busybox", "top")
	ID = strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", ID, "80")

	if !assertPortList(c, out, []string{"0.0.0.0:9876", "0.0.0.0:9999"}) {
		c.Error("Port list is not correct")
	}

	out, _ = dockerCmd(c, "port", ID)

	if !assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:9876",
		"80/tcp -> 0.0.0.0:9999",
		"81/tcp -> 0.0.0.0:9877",
		"82/tcp -> 0.0.0.0:9878"}) {
		c.Error("Port list is not correct\n", out)
	}
	dockerCmd(c, "rm", "-f", ID)

}

func assertPortList(c *check.C, out string, expected []string) bool {
	//lines := strings.Split(out, "\n")
	lines := strings.Split(strings.Trim(out, "\n "), "\n")
	if len(lines) != len(expected) {
		c.Errorf("different size lists %s, %d, %d", out, len(lines), len(expected))
		return false
	}
	sort.Strings(lines)
	sort.Strings(expected)

	for i := 0; i < len(expected); i++ {
		if lines[i] != expected[i] {
			c.Error("|" + lines[i] + "!=" + expected[i] + "|")
			return false
		}
	}

	return true
}

func stopRemoveContainer(id string, c *check.C) {
	dockerCmd(c, "rm", "-f", id)
}

func (s *DockerSuite) TestUnpublishedPortsInPsOutput(c *check.C) {
	// Run busybox with command line expose (equivalent to EXPOSE in image's Dockerfile) for the following ports
	port1 := 80
	port2 := 443
	expose1 := fmt.Sprintf("--expose=%d", port1)
	expose2 := fmt.Sprintf("--expose=%d", port2)
	dockerCmd(c, "run", "-d", expose1, expose2, "busybox", "sleep", "5")

	// Check docker ps o/p for last created container reports the unpublished ports
	unpPort1 := fmt.Sprintf("%d/tcp", port1)
	unpPort2 := fmt.Sprintf("%d/tcp", port2)
	out, _ := dockerCmd(c, "ps", "-n=1")
	if !strings.Contains(out, unpPort1) || !strings.Contains(out, unpPort2) {
		c.Errorf("Missing unpublished ports(s) (%s, %s) in docker ps output: %s", unpPort1, unpPort2, out)
	}

	// Run the container forcing to publish the exposed ports
	dockerCmd(c, "run", "-d", "-P", expose1, expose2, "busybox", "sleep", "5")

	// Check docker ps o/p for last created container reports the exposed ports in the port bindings
	expBndRegx1 := regexp.MustCompile(`0.0.0.0:\d\d\d\d\d->` + unpPort1)
	expBndRegx2 := regexp.MustCompile(`0.0.0.0:\d\d\d\d\d->` + unpPort2)
	out, _ = dockerCmd(c, "ps", "-n=1")
	if !expBndRegx1.MatchString(out) || !expBndRegx2.MatchString(out) {
		c.Errorf("Cannot find expected port binding ports(s) (0.0.0.0:xxxxx->%s, 0.0.0.0:xxxxx->%s) in docker ps output:\n%s",
			unpPort1, unpPort2, out)
	}

	// Run the container specifying explicit port bindings for the exposed ports
	offset := 10000
	pFlag1 := fmt.Sprintf("%d:%d", offset+port1, port1)
	pFlag2 := fmt.Sprintf("%d:%d", offset+port2, port2)
	out, _ = dockerCmd(c, "run", "-d", "-p", pFlag1, "-p", pFlag2, expose1, expose2, "busybox", "sleep", "5")
	id := strings.TrimSpace(out)

	// Check docker ps o/p for last created container reports the specified port mappings
	expBnd1 := fmt.Sprintf("0.0.0.0:%d->%s", offset+port1, unpPort1)
	expBnd2 := fmt.Sprintf("0.0.0.0:%d->%s", offset+port2, unpPort2)
	out, _ = dockerCmd(c, "ps", "-n=1")
	if !strings.Contains(out, expBnd1) || !strings.Contains(out, expBnd2) {
		c.Errorf("Cannot find expected port binding(s) (%s, %s) in docker ps output: %s", expBnd1, expBnd2, out)
	}
	// Remove container now otherwise it will interfeer with next test
	stopRemoveContainer(id, c)

	// Run the container with explicit port bindings and no exposed ports
	out, _ = dockerCmd(c, "run", "-d", "-p", pFlag1, "-p", pFlag2, "busybox", "sleep", "5")
	id = strings.TrimSpace(out)

	// Check docker ps o/p for last created container reports the specified port mappings
	out, _ = dockerCmd(c, "ps", "-n=1")
	if !strings.Contains(out, expBnd1) || !strings.Contains(out, expBnd2) {
		c.Errorf("Cannot find expected port binding(s) (%s, %s) in docker ps output: %s", expBnd1, expBnd2, out)
	}
	// Remove container now otherwise it will interfeer with next test
	stopRemoveContainer(id, c)

	// Run the container with one unpublished exposed port and one explicit port binding
	dockerCmd(c, "run", "-d", expose1, "-p", pFlag2, "busybox", "sleep", "5")

	// Check docker ps o/p for last created container reports the specified unpublished port and port mapping
	out, _ = dockerCmd(c, "ps", "-n=1")
	if !strings.Contains(out, unpPort1) || !strings.Contains(out, expBnd2) {
		c.Errorf("Missing unpublished ports or port binding (%s, %s) in docker ps output: %s", unpPort1, expBnd2, out)
	}
}
