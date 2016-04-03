package main

import (
	"fmt"
	"io/ioutil"
	"net"
	"strings"

	"github.com/go-check/check"
)

func startServerContainer(c *check.C, msg string, port int) string {
	name := "server"
	cmd := []string{
		"-d",
		"-p", fmt.Sprintf("%d:%d", port, port),
		"busybox",
		"sh", "-c", fmt.Sprintf("echo %q | nc -lp %d", msg, port),
	}
	if err := waitForContainer(name, cmd...); err != nil {
		c.Fatalf("Failed to launch server container: %v", err)
	}
	return name
}

func getExternalAddress(c *check.C) net.IP {
	iface, err := net.InterfaceByName("eth0")
	if err != nil {
		c.Skip(fmt.Sprintf("Test not running with `make test`. Interface eth0 not found: %v", err))
	}

	ifaceAddrs, err := iface.Addrs()
	if err != nil || len(ifaceAddrs) == 0 {
		c.Fatalf("Error retrieving addresses for eth0: %v (%d addresses)", err, len(ifaceAddrs))
	}

	ifaceIP, _, err := net.ParseCIDR(ifaceAddrs[0].String())
	if err != nil {
		c.Fatalf("Error retrieving the up for eth0: %s", err)
	}

	return ifaceIP
}

func getContainerLogs(c *check.C, containerID string) string {
	out, _ := dockerCmd(c, "logs", containerID)
	return strings.Trim(out, "\r\n")
}

func getContainerStatus(c *check.C, containerID string) string {
	out, err := inspectField(containerID, "State.Running")
	c.Assert(err, check.IsNil)
	return out
}

func (s *DockerSuite) TestNetworkNat(c *check.C) {
	testRequires(c, SameHostDaemon, NativeExecDriver)
	msg := "it works"
	startServerContainer(c, msg, 8080)
	endpoint := getExternalAddress(c)
	conn, err := net.Dial("tcp", fmt.Sprintf("%s:%d", endpoint.String(), 8080))
	if err != nil {
		c.Fatalf("Failed to connect to container (%v)", err)
	}
	data, err := ioutil.ReadAll(conn)
	conn.Close()
	if err != nil {
		c.Fatal(err)
	}
	final := strings.TrimRight(string(data), "\n")
	if final != msg {
		c.Fatalf("Expected message %q but received %q", msg, final)
	}
}

func (s *DockerSuite) TestNetworkLocalhostTCPNat(c *check.C) {
	testRequires(c, SameHostDaemon, NativeExecDriver)
	var (
		msg = "hi yall"
	)
	startServerContainer(c, msg, 8081)
	conn, err := net.Dial("tcp", "localhost:8081")
	if err != nil {
		c.Fatalf("Failed to connect to container (%v)", err)
	}
	data, err := ioutil.ReadAll(conn)
	conn.Close()
	if err != nil {
		c.Fatal(err)
	}
	final := strings.TrimRight(string(data), "\n")
	if final != msg {
		c.Fatalf("Expected message %q but received %q", msg, final)
	}
}

func (s *DockerSuite) TestNetworkLoopbackNat(c *check.C) {
	testRequires(c, SameHostDaemon, NativeExecDriver)
	msg := "it works"
	startServerContainer(c, msg, 8080)
	endpoint := getExternalAddress(c)
	out, _ := dockerCmd(c, "run", "-t", "--net=container:server", "busybox",
		"sh", "-c", fmt.Sprintf("stty raw && nc -w 5 %s 8080", endpoint.String()))
	final := strings.TrimRight(string(out), "\n")
	if final != msg {
		c.Fatalf("Expected message %q but received %q", msg, final)
	}
}
