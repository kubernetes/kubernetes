package main

import (
	"fmt"
	"net"
	"regexp"
	"sort"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestPortList(c *check.C) {
	testRequires(c, DaemonIsLinux)
	// one port
	out, _ := dockerCmd(c, "run", "-d", "-p", "9876:80", "busybox", "top")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	err := assertPortList(c, out, []string{"0.0.0.0:9876"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

	out, _ = dockerCmd(c, "port", firstID)

	err = assertPortList(c, out, []string{"80/tcp -> 0.0.0.0:9876"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "rm", "-f", firstID)

	// three port
	out, _ = dockerCmd(c, "run", "-d",
		"-p", "9876:80",
		"-p", "9877:81",
		"-p", "9878:82",
		"busybox", "top")
	ID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", ID, "80")

	err = assertPortList(c, out, []string{"0.0.0.0:9876"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

	out, _ = dockerCmd(c, "port", ID)

	err = assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:9876",
		"81/tcp -> 0.0.0.0:9877",
		"82/tcp -> 0.0.0.0:9878"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

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

	err = assertPortList(c, out, []string{"0.0.0.0:9876", "0.0.0.0:9999"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

	out, _ = dockerCmd(c, "port", ID)

	err = assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:9876",
		"80/tcp -> 0.0.0.0:9999",
		"81/tcp -> 0.0.0.0:9877",
		"82/tcp -> 0.0.0.0:9878"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)
	dockerCmd(c, "rm", "-f", ID)

	testRange := func() {
		// host port ranges used
		IDs := make([]string, 3)
		for i := 0; i < 3; i++ {
			out, _ = dockerCmd(c, "run", "-d",
				"-p", "9090-9092:80",
				"busybox", "top")
			IDs[i] = strings.TrimSpace(out)

			out, _ = dockerCmd(c, "port", IDs[i])

			err = assertPortList(c, out, []string{fmt.Sprintf("80/tcp -> 0.0.0.0:%d", 9090+i)})
			// Port list is not correct
			c.Assert(err, checker.IsNil)
		}

		// test port range exhaustion
		out, _, err = dockerCmdWithError("run", "-d",
			"-p", "9090-9092:80",
			"busybox", "top")
		// Exhausted port range did not return an error
		c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))

		for i := 0; i < 3; i++ {
			dockerCmd(c, "rm", "-f", IDs[i])
		}
	}
	testRange()
	// Verify we ran re-use port ranges after they are no longer in use.
	testRange()

	// test invalid port ranges
	for _, invalidRange := range []string{"9090-9089:80", "9090-:80", "-9090:80"} {
		out, _, err = dockerCmdWithError("run", "-d",
			"-p", invalidRange,
			"busybox", "top")
		// Port range should have returned an error
		c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
	}

	// test host range:container range spec.
	out, _ = dockerCmd(c, "run", "-d",
		"-p", "9800-9803:80-83",
		"busybox", "top")
	ID = strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", ID)

	err = assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:9800",
		"81/tcp -> 0.0.0.0:9801",
		"82/tcp -> 0.0.0.0:9802",
		"83/tcp -> 0.0.0.0:9803"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)
	dockerCmd(c, "rm", "-f", ID)

	// test mixing protocols in same port range
	out, _ = dockerCmd(c, "run", "-d",
		"-p", "8000-8080:80",
		"-p", "8000-8080:80/udp",
		"busybox", "top")
	ID = strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", ID)

	err = assertPortList(c, out, []string{
		"80/tcp -> 0.0.0.0:8000",
		"80/udp -> 0.0.0.0:8000"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)
	dockerCmd(c, "rm", "-f", ID)
}

func assertPortList(c *check.C, out string, expected []string) error {
	lines := strings.Split(strings.Trim(out, "\n "), "\n")
	if len(lines) != len(expected) {
		return fmt.Errorf("different size lists %s, %d, %d", out, len(lines), len(expected))
	}
	sort.Strings(lines)
	sort.Strings(expected)

	for i := 0; i < len(expected); i++ {
		if lines[i] != expected[i] {
			return fmt.Errorf("|" + lines[i] + "!=" + expected[i] + "|")
		}
	}

	return nil
}

func stopRemoveContainer(id string, c *check.C) {
	dockerCmd(c, "rm", "-f", id)
}

func (s *DockerSuite) TestUnpublishedPortsInPsOutput(c *check.C) {
	testRequires(c, DaemonIsLinux)
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
	// Missing unpublished ports in docker ps output
	c.Assert(out, checker.Contains, unpPort1)
	// Missing unpublished ports in docker ps output
	c.Assert(out, checker.Contains, unpPort2)

	// Run the container forcing to publish the exposed ports
	dockerCmd(c, "run", "-d", "-P", expose1, expose2, "busybox", "sleep", "5")

	// Check docker ps o/p for last created container reports the exposed ports in the port bindings
	expBndRegx1 := regexp.MustCompile(`0.0.0.0:\d\d\d\d\d->` + unpPort1)
	expBndRegx2 := regexp.MustCompile(`0.0.0.0:\d\d\d\d\d->` + unpPort2)
	out, _ = dockerCmd(c, "ps", "-n=1")
	// Cannot find expected port binding port (0.0.0.0:xxxxx->unpPort1) in docker ps output
	c.Assert(expBndRegx1.MatchString(out), checker.Equals, true, check.Commentf("out: %s; unpPort1: %s", out, unpPort1))
	// Cannot find expected port binding port (0.0.0.0:xxxxx->unpPort2) in docker ps output
	c.Assert(expBndRegx2.MatchString(out), checker.Equals, true, check.Commentf("out: %s; unpPort2: %s", out, unpPort2))

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
	// Cannot find expected port binding (expBnd1) in docker ps output
	c.Assert(out, checker.Contains, expBnd1)
	// Cannot find expected port binding (expBnd2) in docker ps output
	c.Assert(out, checker.Contains, expBnd2)

	// Remove container now otherwise it will interfere with next test
	stopRemoveContainer(id, c)

	// Run the container with explicit port bindings and no exposed ports
	out, _ = dockerCmd(c, "run", "-d", "-p", pFlag1, "-p", pFlag2, "busybox", "sleep", "5")
	id = strings.TrimSpace(out)

	// Check docker ps o/p for last created container reports the specified port mappings
	out, _ = dockerCmd(c, "ps", "-n=1")
	// Cannot find expected port binding (expBnd1) in docker ps output
	c.Assert(out, checker.Contains, expBnd1)
	// Cannot find expected port binding (expBnd2) in docker ps output
	c.Assert(out, checker.Contains, expBnd2)
	// Remove container now otherwise it will interfere with next test
	stopRemoveContainer(id, c)

	// Run the container with one unpublished exposed port and one explicit port binding
	dockerCmd(c, "run", "-d", expose1, "-p", pFlag2, "busybox", "sleep", "5")

	// Check docker ps o/p for last created container reports the specified unpublished port and port mapping
	out, _ = dockerCmd(c, "ps", "-n=1")
	// Missing unpublished exposed ports (unpPort1) in docker ps output
	c.Assert(out, checker.Contains, unpPort1)
	// Missing port binding (expBnd2) in docker ps output
	c.Assert(out, checker.Contains, expBnd2)
}

func (s *DockerSuite) TestPortHostBinding(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	out, _ := dockerCmd(c, "run", "-d", "-p", "9876:80", "busybox",
		"nc", "-l", "-p", "80")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	err := assertPortList(c, out, []string{"0.0.0.0:9876"})
	// Port list is not correct
	c.Assert(err, checker.IsNil)

	dockerCmd(c, "run", "--net=host", "busybox",
		"nc", "localhost", "9876")

	dockerCmd(c, "rm", "-f", firstID)

	out, _, err = dockerCmdWithError("run", "--net=host", "busybox", "nc", "localhost", "9876")
	// Port is still bound after the Container is removed
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
}

func (s *DockerSuite) TestPortExposeHostBinding(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	out, _ := dockerCmd(c, "run", "-d", "-P", "--expose", "80", "busybox",
		"nc", "-l", "-p", "80")
	firstID := strings.TrimSpace(out)

	out, _ = dockerCmd(c, "port", firstID, "80")

	_, exposedPort, err := net.SplitHostPort(out)
	c.Assert(err, checker.IsNil, check.Commentf("out: %s", out))

	dockerCmd(c, "run", "--net=host", "busybox",
		"nc", "localhost", strings.TrimSpace(exposedPort))

	dockerCmd(c, "rm", "-f", firstID)

	out, _, err = dockerCmdWithError("run", "--net=host", "busybox",
		"nc", "localhost", strings.TrimSpace(exposedPort))
	// Port is still bound after the Container is removed
	c.Assert(err, checker.NotNil, check.Commentf("out: %s", out))
}

func (s *DockerSuite) TestPortBindingOnSandbox(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	dockerCmd(c, "network", "create", "--internal", "-d", "bridge", "internal-net")
	nr := getNetworkResource(c, "internal-net")
	c.Assert(nr.Internal, checker.Equals, true)

	dockerCmd(c, "run", "--net", "internal-net", "-d", "--name", "c1",
		"-p", "8080:8080", "busybox", "nc", "-l", "-p", "8080")
	c.Assert(waitRun("c1"), check.IsNil)

	_, _, err := dockerCmdWithError("run", "--net=host", "busybox", "nc", "localhost", "8080")
	c.Assert(err, check.NotNil,
		check.Commentf("Port mapping on internal network is expected to fail"))

	// Connect container to another normal bridge network
	dockerCmd(c, "network", "create", "-d", "bridge", "foo-net")
	dockerCmd(c, "network", "connect", "foo-net", "c1")

	_, _, err = dockerCmdWithError("run", "--net=host", "busybox", "nc", "localhost", "8080")
	c.Assert(err, check.IsNil,
		check.Commentf("Port mapping on the new network is expected to succeed"))

}
