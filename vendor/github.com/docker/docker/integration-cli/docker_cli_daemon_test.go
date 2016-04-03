// +build daemon

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libtrust"
	"github.com/go-check/check"
)

func (s *DockerDaemonSuite) TestDaemonRestartWithRunningContainersPorts(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	if out, err := s.d.Cmd("run", "-d", "--name", "top1", "-p", "1234:80", "--restart", "always", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top1: err=%v\n%s", err, out)
	}
	// --restart=no by default
	if out, err := s.d.Cmd("run", "-d", "--name", "top2", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top2: err=%v\n%s", err, out)
	}

	testRun := func(m map[string]bool, prefix string) {
		var format string
		for cont, shouldRun := range m {
			out, err := s.d.Cmd("ps")
			if err != nil {
				c.Fatalf("Could not run ps: err=%v\n%q", err, out)
			}
			if shouldRun {
				format = "%scontainer %q is not running"
			} else {
				format = "%scontainer %q is running"
			}
			if shouldRun != strings.Contains(out, cont) {
				c.Fatalf(format, prefix, cont)
			}
		}
	}

	testRun(map[string]bool{"top1": true, "top2": true}, "")

	if err := s.d.Restart(); err != nil {
		c.Fatalf("Could not restart daemon: %v", err)
	}
	testRun(map[string]bool{"top1": true, "top2": false}, "After daemon restart: ")
}

func (s *DockerDaemonSuite) TestDaemonRestartWithVolumesRefs(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	if out, err := s.d.Cmd("run", "-d", "--name", "volrestarttest1", "-v", "/foo", "busybox"); err != nil {
		c.Fatal(err, out)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatal(err)
	}
	if _, err := s.d.Cmd("run", "-d", "--volumes-from", "volrestarttest1", "--name", "volrestarttest2", "busybox", "top"); err != nil {
		c.Fatal(err)
	}

	if out, err := s.d.Cmd("rm", "-fv", "volrestarttest2"); err != nil {
		c.Fatal(err, out)
	}

	out, err := s.d.Cmd("inspect", "-f", "{{json .Mounts}}", "volrestarttest1")
	c.Assert(err, check.IsNil)

	if _, err := inspectMountPointJSON(out, "/foo"); err != nil {
		c.Fatalf("Expected volume to exist: /foo, error: %v\n", err)
	}
}

func (s *DockerDaemonSuite) TestDaemonStartIptablesFalse(c *check.C) {
	if err := s.d.Start("--iptables=false"); err != nil {
		c.Fatalf("we should have been able to start the daemon with passing iptables=false: %v", err)
	}
}

// Issue #8444: If docker0 bridge is modified (intentionally or unintentionally) and
// no longer has an IP associated, we should gracefully handle that case and associate
// an IP with it rather than fail daemon start
func (s *DockerDaemonSuite) TestDaemonStartBridgeWithoutIPAssociation(c *check.C) {
	// rather than depending on brctl commands to verify docker0 is created and up
	// let's start the daemon and stop it, and then make a modification to run the
	// actual test
	if err := s.d.Start(); err != nil {
		c.Fatalf("Could not start daemon: %v", err)
	}
	if err := s.d.Stop(); err != nil {
		c.Fatalf("Could not stop daemon: %v", err)
	}

	// now we will remove the ip from docker0 and then try starting the daemon
	ipCmd := exec.Command("ip", "addr", "flush", "dev", "docker0")
	stdout, stderr, _, err := runCommandWithStdoutStderr(ipCmd)
	if err != nil {
		c.Fatalf("failed to remove docker0 IP association: %v, stdout: %q, stderr: %q", err, stdout, stderr)
	}

	if err := s.d.Start(); err != nil {
		warning := "**WARNING: Docker bridge network in bad state--delete docker0 bridge interface to fix"
		c.Fatalf("Could not start daemon when docker0 has no IP address: %v\n%s", err, warning)
	}
}

func (s *DockerDaemonSuite) TestDaemonIptablesClean(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	if out, err := s.d.Cmd("run", "-d", "--name", "top", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top: %s, %v", out, err)
	}

	// get output from iptables with container running
	ipTablesSearchString := "tcp dpt:80"
	ipTablesCmd := exec.Command("iptables", "-nvL")
	out, _, err := runCommandWithOutput(ipTablesCmd)
	if err != nil {
		c.Fatalf("Could not run iptables -nvL: %s, %v", out, err)
	}

	if !strings.Contains(out, ipTablesSearchString) {
		c.Fatalf("iptables output should have contained %q, but was %q", ipTablesSearchString, out)
	}

	if err := s.d.Stop(); err != nil {
		c.Fatalf("Could not stop daemon: %v", err)
	}

	// get output from iptables after restart
	ipTablesCmd = exec.Command("iptables", "-nvL")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	if err != nil {
		c.Fatalf("Could not run iptables -nvL: %s, %v", out, err)
	}

	if strings.Contains(out, ipTablesSearchString) {
		c.Fatalf("iptables output should not have contained %q, but was %q", ipTablesSearchString, out)
	}
}

func (s *DockerDaemonSuite) TestDaemonIptablesCreate(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	if out, err := s.d.Cmd("run", "-d", "--name", "top", "--restart=always", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top: %s, %v", out, err)
	}

	// get output from iptables with container running
	ipTablesSearchString := "tcp dpt:80"
	ipTablesCmd := exec.Command("iptables", "-nvL")
	out, _, err := runCommandWithOutput(ipTablesCmd)
	if err != nil {
		c.Fatalf("Could not run iptables -nvL: %s, %v", out, err)
	}

	if !strings.Contains(out, ipTablesSearchString) {
		c.Fatalf("iptables output should have contained %q, but was %q", ipTablesSearchString, out)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatalf("Could not restart daemon: %v", err)
	}

	// make sure the container is not running
	runningOut, err := s.d.Cmd("inspect", "--format='{{.State.Running}}'", "top")
	if err != nil {
		c.Fatalf("Could not inspect on container: %s, %v", out, err)
	}
	if strings.TrimSpace(runningOut) != "true" {
		c.Fatalf("Container should have been restarted after daemon restart. Status running should have been true but was: %q", strings.TrimSpace(runningOut))
	}

	// get output from iptables after restart
	ipTablesCmd = exec.Command("iptables", "-nvL")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	if err != nil {
		c.Fatalf("Could not run iptables -nvL: %s, %v", out, err)
	}

	if !strings.Contains(out, ipTablesSearchString) {
		c.Fatalf("iptables output after restart should have contained %q, but was %q", ipTablesSearchString, out)
	}
}

// TestDaemonIPv6Enabled checks that when the daemon is started with --ipv6=true that the docker0 bridge
// has the fe80::1 address and that a container is assigned a link-local address
func (s *DockerSuite) TestDaemonIPv6Enabled(c *check.C) {
	testRequires(c, IPv6)

	if err := setupV6(); err != nil {
		c.Fatal("Could not set up host for IPv6 tests")
	}

	d := NewDaemon(c)

	if err := d.StartWithBusybox("--ipv6"); err != nil {
		c.Fatal(err)
	}
	defer d.Stop()

	iface, err := net.InterfaceByName("docker0")
	if err != nil {
		c.Fatalf("Error getting docker0 interface: %v", err)
	}

	addrs, err := iface.Addrs()
	if err != nil {
		c.Fatalf("Error getting addresses for docker0 interface: %v", err)
	}

	var found bool
	expected := "fe80::1/64"

	for i := range addrs {
		if addrs[i].String() == expected {
			found = true
		}
	}

	if !found {
		c.Fatalf("Bridge does not have an IPv6 Address")
	}

	if out, err := d.Cmd("run", "-itd", "--name=ipv6test", "busybox:latest"); err != nil {
		c.Fatalf("Could not run container: %s, %v", out, err)
	}

	out, err := d.Cmd("inspect", "--format", "'{{.NetworkSettings.LinkLocalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip == nil {
		c.Fatalf("Container should have a link-local IPv6 address")
	}

	out, err = d.Cmd("inspect", "--format", "'{{.NetworkSettings.GlobalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip != nil {
		c.Fatalf("Container should not have a global IPv6 address: %v", out)
	}

	if err := teardownV6(); err != nil {
		c.Fatal("Could not perform teardown for IPv6 tests")
	}

}

// TestDaemonIPv6FixedCIDR checks that when the daemon is started with --ipv6=true and a fixed CIDR
// that running containers are given a link-local and global IPv6 address
func (s *DockerSuite) TestDaemonIPv6FixedCIDR(c *check.C) {
	testRequires(c, IPv6)

	if err := setupV6(); err != nil {
		c.Fatal("Could not set up host for IPv6 tests")
	}

	d := NewDaemon(c)

	if err := d.StartWithBusybox("--ipv6", "--fixed-cidr-v6='2001:db8:1::/64'"); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}
	defer d.Stop()

	if out, err := d.Cmd("run", "-itd", "--name=ipv6test", "busybox:latest"); err != nil {
		c.Fatalf("Could not run container: %s, %v", out, err)
	}

	out, err := d.Cmd("inspect", "--format", "'{{.NetworkSettings.LinkLocalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip == nil {
		c.Fatalf("Container should have a link-local IPv6 address")
	}

	out, err = d.Cmd("inspect", "--format", "'{{.NetworkSettings.GlobalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip == nil {
		c.Fatalf("Container should have a global IPv6 address")
	}
	if err := teardownV6(); err != nil {
		c.Fatal("Could not perform teardown for IPv6 tests")
	}
}

func (s *DockerDaemonSuite) TestDaemonLogLevelWrong(c *check.C) {
	c.Assert(s.d.Start("--log-level=bogus"), check.NotNil, check.Commentf("Daemon shouldn't start with wrong log level"))
}

func (s *DockerDaemonSuite) TestDaemonLogLevelDebug(c *check.C) {
	if err := s.d.Start("--log-level=debug"); err != nil {
		c.Fatal(err)
	}
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Missing level="debug" in log file:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonLogLevelFatal(c *check.C) {
	// we creating new daemons to create new logFile
	if err := s.d.Start("--log-level=fatal"); err != nil {
		c.Fatal(err)
	}
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should not have level="debug" in log file:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagD(c *check.C) {
	if err := s.d.Start("-D"); err != nil {
		c.Fatal(err)
	}
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should have level="debug" in log file using -D:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagDebug(c *check.C) {
	if err := s.d.Start("--debug"); err != nil {
		c.Fatal(err)
	}
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should have level="debug" in log file using --debug:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagDebugLogLevelFatal(c *check.C) {
	if err := s.d.Start("--debug", "--log-level=fatal"); err != nil {
		c.Fatal(err)
	}
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should have level="debug" in log file when using both --debug and --log-level=fatal:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonAllocatesListeningPort(c *check.C) {
	listeningPorts := [][]string{
		{"0.0.0.0", "0.0.0.0", "5678"},
		{"127.0.0.1", "127.0.0.1", "1234"},
		{"localhost", "127.0.0.1", "1235"},
	}

	cmdArgs := []string{}
	for _, hostDirective := range listeningPorts {
		cmdArgs = append(cmdArgs, "--host", fmt.Sprintf("tcp://%s:%s", hostDirective[0], hostDirective[2]))
	}

	if err := s.d.StartWithBusybox(cmdArgs...); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	for _, hostDirective := range listeningPorts {
		output, err := s.d.Cmd("run", "-p", fmt.Sprintf("%s:%s:80", hostDirective[1], hostDirective[2]), "busybox", "true")
		if err == nil {
			c.Fatalf("Container should not start, expected port already allocated error: %q", output)
		} else if !strings.Contains(output, "port is already allocated") {
			c.Fatalf("Expected port is already allocated error: %q", output)
		}
	}
}

func (s *DockerDaemonSuite) TestDaemonKeyGeneration(c *check.C) {
	// TODO: skip or update for Windows daemon
	os.Remove("/etc/docker/key.json")
	if err := s.d.Start(); err != nil {
		c.Fatalf("Could not start daemon: %v", err)
	}
	s.d.Stop()

	k, err := libtrust.LoadKeyFile("/etc/docker/key.json")
	if err != nil {
		c.Fatalf("Error opening key file")
	}
	kid := k.KeyID()
	// Test Key ID is a valid fingerprint (e.g. QQXN:JY5W:TBXI:MK3X:GX6P:PD5D:F56N:NHCS:LVRZ:JA46:R24J:XEFF)
	if len(kid) != 59 {
		c.Fatalf("Bad key ID: %s", kid)
	}
}

func (s *DockerDaemonSuite) TestDaemonKeyMigration(c *check.C) {
	// TODO: skip or update for Windows daemon
	os.Remove("/etc/docker/key.json")
	k1, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		c.Fatalf("Error generating private key: %s", err)
	}
	if err := os.MkdirAll(filepath.Join(os.Getenv("HOME"), ".docker"), 0755); err != nil {
		c.Fatalf("Error creating .docker directory: %s", err)
	}
	if err := libtrust.SaveKey(filepath.Join(os.Getenv("HOME"), ".docker", "key.json"), k1); err != nil {
		c.Fatalf("Error saving private key: %s", err)
	}

	if err := s.d.Start(); err != nil {
		c.Fatalf("Could not start daemon: %v", err)
	}
	s.d.Stop()

	k2, err := libtrust.LoadKeyFile("/etc/docker/key.json")
	if err != nil {
		c.Fatalf("Error opening key file")
	}
	if k1.KeyID() != k2.KeyID() {
		c.Fatalf("Key not migrated")
	}
}

// GH#11320 - verify that the daemon exits on failure properly
// Note that this explicitly tests the conflict of {-b,--bridge} and {--bip} options as the means
// to get a daemon init failure; no other tests for -b/--bip conflict are therefore required
func (s *DockerDaemonSuite) TestDaemonExitOnFailure(c *check.C) {
	//attempt to start daemon with incorrect flags (we know -b and --bip conflict)
	if err := s.d.Start("--bridge", "nosuchbridge", "--bip", "1.1.1.1"); err != nil {
		//verify we got the right error
		if !strings.Contains(err.Error(), "Daemon exited and never started") {
			c.Fatalf("Expected daemon not to start, got %v", err)
		}
		// look in the log and make sure we got the message that daemon is shutting down
		runCmd := exec.Command("grep", "Error starting daemon", s.d.LogfileName())
		if out, _, err := runCommandWithOutput(runCmd); err != nil {
			c.Fatalf("Expected 'Error starting daemon' message; but doesn't exist in log: %q, err: %v", out, err)
		}
	} else {
		//if we didn't get an error and the daemon is running, this is a failure
		c.Fatal("Conflicting options should cause the daemon to error out with a failure")
	}
}

func (s *DockerDaemonSuite) TestDaemonBridgeExternal(c *check.C) {
	d := s.d
	err := d.Start("--bridge", "nosuchbridge")
	c.Assert(err, check.NotNil, check.Commentf("--bridge option with an invalid bridge should cause the daemon to fail"))
	defer d.Restart()

	bridgeName := "external-bridge"
	bridgeIp := "192.169.1.1/24"
	_, bridgeIPNet, _ := net.ParseCIDR(bridgeIp)

	out, err := createInterface(c, "bridge", bridgeName, bridgeIp)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, bridgeName)

	err = d.StartWithBusybox("--bridge", bridgeName)
	c.Assert(err, check.IsNil)

	ipTablesSearchString := bridgeIPNet.String()
	ipTablesCmd := exec.Command("iptables", "-t", "nat", "-nvL")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	c.Assert(err, check.IsNil)

	c.Assert(strings.Contains(out, ipTablesSearchString), check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q",
			ipTablesSearchString, out))

	_, err = d.Cmd("run", "-d", "--name", "ExtContainer", "busybox", "top")
	c.Assert(err, check.IsNil)

	containerIp := d.findContainerIP("ExtContainer")
	ip := net.ParseIP(containerIp)
	c.Assert(bridgeIPNet.Contains(ip), check.Equals, true,
		check.Commentf("Container IP-Address must be in the same subnet range : %s",
			containerIp))
}

func createInterface(c *check.C, ifType string, ifName string, ipNet string) (string, error) {
	args := []string{"link", "add", "name", ifName, "type", ifType}
	ipLinkCmd := exec.Command("ip", args...)
	out, _, err := runCommandWithOutput(ipLinkCmd)
	if err != nil {
		return out, err
	}

	ifCfgCmd := exec.Command("ifconfig", ifName, ipNet, "up")
	out, _, err = runCommandWithOutput(ifCfgCmd)
	return out, err
}

func deleteInterface(c *check.C, ifName string) {
	ifCmd := exec.Command("ip", "link", "delete", ifName)
	out, _, err := runCommandWithOutput(ifCmd)
	c.Assert(err, check.IsNil, check.Commentf(out))

	flushCmd := exec.Command("iptables", "-t", "nat", "--flush")
	out, _, err = runCommandWithOutput(flushCmd)
	c.Assert(err, check.IsNil, check.Commentf(out))

	flushCmd = exec.Command("iptables", "--flush")
	out, _, err = runCommandWithOutput(flushCmd)
	c.Assert(err, check.IsNil, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestDaemonBridgeIP(c *check.C) {
	// TestDaemonBridgeIP Steps
	// 1. Delete the existing docker0 Bridge
	// 2. Set --bip daemon configuration and start the new Docker Daemon
	// 3. Check if the bip config has taken effect using ifconfig and iptables commands
	// 4. Launch a Container and make sure the IP-Address is in the expected subnet
	// 5. Delete the docker0 Bridge
	// 6. Restart the Docker Daemon (via defered action)
	//    This Restart takes care of bringing docker0 interface back to auto-assigned IP

	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIp := "192.169.1.1/24"
	ip, bridgeIPNet, _ := net.ParseCIDR(bridgeIp)

	err := d.StartWithBusybox("--bip", bridgeIp)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	ifconfigSearchString := ip.String()
	ifconfigCmd := exec.Command("ifconfig", defaultNetworkBridge)
	out, _, _, err := runCommandWithStdoutStderr(ifconfigCmd)
	c.Assert(err, check.IsNil)

	c.Assert(strings.Contains(out, ifconfigSearchString), check.Equals, true,
		check.Commentf("ifconfig output should have contained %q, but was %q",
			ifconfigSearchString, out))

	ipTablesSearchString := bridgeIPNet.String()
	ipTablesCmd := exec.Command("iptables", "-t", "nat", "-nvL")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	c.Assert(err, check.IsNil)

	c.Assert(strings.Contains(out, ipTablesSearchString), check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q",
			ipTablesSearchString, out))

	out, err = d.Cmd("run", "-d", "--name", "test", "busybox", "top")
	c.Assert(err, check.IsNil)

	containerIp := d.findContainerIP("test")
	ip = net.ParseIP(containerIp)
	c.Assert(bridgeIPNet.Contains(ip), check.Equals, true,
		check.Commentf("Container IP-Address must be in the same subnet range : %s",
			containerIp))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithBridgeIPChange(c *check.C) {
	if err := s.d.Start(); err != nil {
		c.Fatalf("Could not start daemon: %v", err)
	}
	defer s.d.Restart()
	if err := s.d.Stop(); err != nil {
		c.Fatalf("Could not stop daemon: %v", err)
	}

	// now we will change the docker0's IP and then try starting the daemon
	bridgeIP := "192.169.100.1/24"
	_, bridgeIPNet, _ := net.ParseCIDR(bridgeIP)

	ipCmd := exec.Command("ifconfig", "docker0", bridgeIP)
	stdout, stderr, _, err := runCommandWithStdoutStderr(ipCmd)
	if err != nil {
		c.Fatalf("failed to change docker0's IP association: %v, stdout: %q, stderr: %q", err, stdout, stderr)
	}

	if err := s.d.Start("--bip", bridgeIP); err != nil {
		c.Fatalf("Could not start daemon: %v", err)
	}

	//check if the iptables contains new bridgeIP MASQUERADE rule
	ipTablesSearchString := bridgeIPNet.String()
	ipTablesCmd := exec.Command("iptables", "-t", "nat", "-nvL")
	out, _, err := runCommandWithOutput(ipTablesCmd)
	if err != nil {
		c.Fatalf("Could not run iptables -nvL: %s, %v", out, err)
	}
	if !strings.Contains(out, ipTablesSearchString) {
		c.Fatalf("iptables output should have contained new MASQUERADE rule with IP %q, but was %q", ipTablesSearchString, out)
	}
}

func (s *DockerDaemonSuite) TestDaemonBridgeFixedCidr(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIp := "192.169.1.1/24"

	out, err := createInterface(c, "bridge", bridgeName, bridgeIp)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, bridgeName)

	args := []string{"--bridge", bridgeName, "--fixed-cidr", "192.169.1.0/30"}
	err = d.StartWithBusybox(args...)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	for i := 0; i < 4; i++ {
		cName := "Container" + strconv.Itoa(i)
		out, err := d.Cmd("run", "-d", "--name", cName, "busybox", "top")
		if err != nil {
			c.Assert(strings.Contains(out, "no available ip addresses"), check.Equals, true,
				check.Commentf("Could not run a Container : %s %s", err.Error(), out))
		}
	}
}

func (s *DockerDaemonSuite) TestDaemonDefaultGatewayIPv4Implicit(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIp := "192.169.1.1"
	bridgeIpNet := fmt.Sprintf("%s/24", bridgeIp)

	err := d.StartWithBusybox("--bip", bridgeIpNet)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	expectedMessage := fmt.Sprintf("default via %s dev", bridgeIp)
	out, err := d.Cmd("run", "busybox", "ip", "-4", "route", "list", "0/0")
	c.Assert(strings.Contains(out, expectedMessage), check.Equals, true,
		check.Commentf("Implicit default gateway should be bridge IP %s, but default route was '%s'",
			bridgeIp, strings.TrimSpace(out)))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonDefaultGatewayIPv4Explicit(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIp := "192.169.1.1"
	bridgeIpNet := fmt.Sprintf("%s/24", bridgeIp)
	gatewayIp := "192.169.1.254"

	err := d.StartWithBusybox("--bip", bridgeIpNet, "--default-gateway", gatewayIp)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	expectedMessage := fmt.Sprintf("default via %s dev", gatewayIp)
	out, err := d.Cmd("run", "busybox", "ip", "-4", "route", "list", "0/0")
	c.Assert(strings.Contains(out, expectedMessage), check.Equals, true,
		check.Commentf("Explicit default gateway should be %s, but default route was '%s'",
			gatewayIp, strings.TrimSpace(out)))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonIP(c *check.C) {
	d := s.d

	ipStr := "192.170.1.1/24"
	ip, _, _ := net.ParseCIDR(ipStr)
	args := []string{"--ip", ip.String()}
	err := d.StartWithBusybox(args...)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	out, err := d.Cmd("run", "-d", "-p", "8000:8000", "busybox", "top")
	c.Assert(err, check.NotNil,
		check.Commentf("Running a container must fail with an invalid --ip option"))
	c.Assert(strings.Contains(out, "Error starting userland proxy"), check.Equals, true)

	ifName := "dummy"
	out, err = createInterface(c, "dummy", ifName, ipStr)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, ifName)

	_, err = d.Cmd("run", "-d", "-p", "8000:8000", "busybox", "top")
	c.Assert(err, check.IsNil)

	ipTablesCmd := exec.Command("iptables", "-t", "nat", "-nvL")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	c.Assert(err, check.IsNil)

	regex := fmt.Sprintf("DNAT.*%s.*dpt:8000", ip.String())
	matched, _ := regexp.MatchString(regex, out)
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, out))
}

func (s *DockerDaemonSuite) TestDaemonICCPing(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIp := "192.169.1.1/24"

	out, err := createInterface(c, "bridge", bridgeName, bridgeIp)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, bridgeName)

	args := []string{"--bridge", bridgeName, "--icc=false"}
	err = d.StartWithBusybox(args...)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	ipTablesCmd := exec.Command("iptables", "-nvL", "FORWARD")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	c.Assert(err, check.IsNil)

	regex := fmt.Sprintf("DROP.*all.*%s.*%s", bridgeName, bridgeName)
	matched, _ := regexp.MatchString(regex, out)
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, out))

	// Pinging another container must fail with --icc=false
	pingContainers(c, d, true)

	ipStr := "192.171.1.1/24"
	ip, _, _ := net.ParseCIDR(ipStr)
	ifName := "icc-dummy"

	createInterface(c, "dummy", ifName, ipStr)

	// But, Pinging external or a Host interface must succeed
	pingCmd := fmt.Sprintf("ping -c 1 %s -W 1", ip.String())
	runArgs := []string{"--rm", "busybox", "sh", "-c", pingCmd}
	_, err = d.Cmd("run", runArgs...)
	c.Assert(err, check.IsNil)
}

func (s *DockerDaemonSuite) TestDaemonICCLinkExpose(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIp := "192.169.1.1/24"

	out, err := createInterface(c, "bridge", bridgeName, bridgeIp)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, bridgeName)

	args := []string{"--bridge", bridgeName, "--icc=false"}
	err = d.StartWithBusybox(args...)
	c.Assert(err, check.IsNil)
	defer d.Restart()

	ipTablesCmd := exec.Command("iptables", "-nvL", "FORWARD")
	out, _, err = runCommandWithOutput(ipTablesCmd)
	c.Assert(err, check.IsNil)

	regex := fmt.Sprintf("DROP.*all.*%s.*%s", bridgeName, bridgeName)
	matched, _ := regexp.MatchString(regex, out)
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, out))

	out, err = d.Cmd("run", "-d", "--expose", "4567", "--name", "icc1", "busybox", "nc", "-l", "-p", "4567")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = d.Cmd("run", "--link", "icc1:icc1", "busybox", "nc", "icc1", "4567")
	c.Assert(err, check.IsNil, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestDaemonLinksIpTablesRulesWhenLinkAndUnlink(c *check.C) {
	bridgeName := "external-bridge"
	bridgeIp := "192.169.1.1/24"

	out, err := createInterface(c, "bridge", bridgeName, bridgeIp)
	c.Assert(err, check.IsNil, check.Commentf(out))
	defer deleteInterface(c, bridgeName)

	args := []string{"--bridge", bridgeName, "--icc=false"}
	err = s.d.StartWithBusybox(args...)
	c.Assert(err, check.IsNil)
	defer s.d.Restart()

	_, err = s.d.Cmd("run", "-d", "--name", "child", "--publish", "8080:80", "busybox", "top")
	c.Assert(err, check.IsNil)
	_, err = s.d.Cmd("run", "-d", "--name", "parent", "--link", "child:http", "busybox", "top")
	c.Assert(err, check.IsNil)

	childIP := s.d.findContainerIP("child")
	parentIP := s.d.findContainerIP("parent")

	sourceRule := []string{"-i", bridgeName, "-o", bridgeName, "-p", "tcp", "-s", childIP, "--sport", "80", "-d", parentIP, "-j", "ACCEPT"}
	destinationRule := []string{"-i", bridgeName, "-o", bridgeName, "-p", "tcp", "-s", parentIP, "--dport", "80", "-d", childIP, "-j", "ACCEPT"}
	if !iptables.Exists("filter", "DOCKER", sourceRule...) || !iptables.Exists("filter", "DOCKER", destinationRule...) {
		c.Fatal("Iptables rules not found")
	}

	s.d.Cmd("rm", "--link", "parent/http")
	if iptables.Exists("filter", "DOCKER", sourceRule...) || iptables.Exists("filter", "DOCKER", destinationRule...) {
		c.Fatal("Iptables rules should be removed when unlink")
	}

	s.d.Cmd("kill", "child")
	s.d.Cmd("kill", "parent")
}

func (s *DockerDaemonSuite) TestDaemonUlimitDefaults(c *check.C) {
	testRequires(c, NativeExecDriver)

	if err := s.d.StartWithBusybox("--default-ulimit", "nofile=42:42", "--default-ulimit", "nproc=1024:1024"); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "--ulimit", "nproc=2048", "--name=test", "busybox", "/bin/sh", "-c", "echo $(ulimit -n); echo $(ulimit -p)")
	if err != nil {
		c.Fatal(out, err)
	}

	outArr := strings.Split(out, "\n")
	if len(outArr) < 2 {
		c.Fatalf("got unexpected output: %s", out)
	}
	nofile := strings.TrimSpace(outArr[0])
	nproc := strings.TrimSpace(outArr[1])

	if nofile != "42" {
		c.Fatalf("expected `ulimit -n` to be `42`, got: %s", nofile)
	}
	if nproc != "2048" {
		c.Fatalf("exepcted `ulimit -p` to be 2048, got: %s", nproc)
	}

	// Now restart daemon with a new default
	if err := s.d.Restart("--default-ulimit", "nofile=43"); err != nil {
		c.Fatal(err)
	}

	out, err = s.d.Cmd("start", "-a", "test")
	if err != nil {
		c.Fatal(err)
	}

	outArr = strings.Split(out, "\n")
	if len(outArr) < 2 {
		c.Fatalf("got unexpected output: %s", out)
	}
	nofile = strings.TrimSpace(outArr[0])
	nproc = strings.TrimSpace(outArr[1])

	if nofile != "43" {
		c.Fatalf("expected `ulimit -n` to be `43`, got: %s", nofile)
	}
	if nproc != "2048" {
		c.Fatalf("exepcted `ulimit -p` to be 2048, got: %s", nproc)
	}
}

// #11315
func (s *DockerDaemonSuite) TestDaemonRestartRenameContainer(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	if out, err := s.d.Cmd("run", "--name=test", "busybox"); err != nil {
		c.Fatal(err, out)
	}

	if out, err := s.d.Cmd("rename", "test", "test2"); err != nil {
		c.Fatal(err, out)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatal(err)
	}

	if out, err := s.d.Cmd("start", "test2"); err != nil {
		c.Fatal(err, out)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverDefault(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "-d", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id := strings.TrimSpace(out)

	if out, err := s.d.Cmd("wait", id); err != nil {
		c.Fatal(out, err)
	}
	logPath := filepath.Join(s.d.folder, "graph", "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err != nil {
		c.Fatal(err)
	}
	f, err := os.Open(logPath)
	if err != nil {
		c.Fatal(err)
	}
	var res struct {
		Log    string    `json:"log"`
		Stream string    `json:"stream"`
		Time   time.Time `json:"time"`
	}
	if err := json.NewDecoder(f).Decode(&res); err != nil {
		c.Fatal(err)
	}
	if res.Log != "testline\n" {
		c.Fatalf("Unexpected log line: %q, expected: %q", res.Log, "testline\n")
	}
	if res.Stream != "stdout" {
		c.Fatalf("Unexpected stream: %q, expected: %q", res.Stream, "stdout")
	}
	if !time.Now().After(res.Time) {
		c.Fatalf("Log time %v in future", res.Time)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverDefaultOverride(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "-d", "--log-driver=none", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id := strings.TrimSpace(out)

	if out, err := s.d.Cmd("wait", id); err != nil {
		c.Fatal(out, err)
	}
	logPath := filepath.Join(s.d.folder, "graph", "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err == nil || !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't exits, error on Stat: %s", logPath, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverNone(c *check.C) {
	if err := s.d.StartWithBusybox("--log-driver=none"); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "-d", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id := strings.TrimSpace(out)
	if out, err := s.d.Cmd("wait", id); err != nil {
		c.Fatal(out, err)
	}

	logPath := filepath.Join(s.d.folder, "graph", "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err == nil || !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't exits, error on Stat: %s", logPath, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverNoneOverride(c *check.C) {
	if err := s.d.StartWithBusybox("--log-driver=none"); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "-d", "--log-driver=json-file", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id := strings.TrimSpace(out)

	if out, err := s.d.Cmd("wait", id); err != nil {
		c.Fatal(out, err)
	}
	logPath := filepath.Join(s.d.folder, "graph", "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err != nil {
		c.Fatal(err)
	}
	f, err := os.Open(logPath)
	if err != nil {
		c.Fatal(err)
	}
	var res struct {
		Log    string    `json:"log"`
		Stream string    `json:"stream"`
		Time   time.Time `json:"time"`
	}
	if err := json.NewDecoder(f).Decode(&res); err != nil {
		c.Fatal(err)
	}
	if res.Log != "testline\n" {
		c.Fatalf("Unexpected log line: %q, expected: %q", res.Log, "testline\n")
	}
	if res.Stream != "stdout" {
		c.Fatalf("Unexpected stream: %q, expected: %q", res.Stream, "stdout")
	}
	if !time.Now().After(res.Time) {
		c.Fatalf("Log time %v in future", res.Time)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverNoneLogsError(c *check.C) {
	if err := s.d.StartWithBusybox("--log-driver=none"); err != nil {
		c.Fatal(err)
	}

	out, err := s.d.Cmd("run", "-d", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id := strings.TrimSpace(out)
	out, err = s.d.Cmd("logs", id)
	if err == nil {
		c.Fatalf("Logs should fail with \"none\" driver")
	}
	if !strings.Contains(out, `"logs" command is supported only for "json-file" logging driver`) {
		c.Fatalf("There should be error about non-json-file driver, got: %s", out)
	}
}

func (s *DockerDaemonSuite) TestDaemonDots(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}

	// Now create 4 containers
	if _, err := s.d.Cmd("create", "busybox"); err != nil {
		c.Fatalf("Error creating container: %q", err)
	}
	if _, err := s.d.Cmd("create", "busybox"); err != nil {
		c.Fatalf("Error creating container: %q", err)
	}
	if _, err := s.d.Cmd("create", "busybox"); err != nil {
		c.Fatalf("Error creating container: %q", err)
	}
	if _, err := s.d.Cmd("create", "busybox"); err != nil {
		c.Fatalf("Error creating container: %q", err)
	}

	s.d.Stop()

	s.d.Start("--log-level=debug")
	s.d.Stop()
	content, _ := ioutil.ReadFile(s.d.logFile.Name())
	if strings.Contains(string(content), "....") {
		c.Fatalf("Debug level should not have ....\n%s", string(content))
	}

	s.d.Start("--log-level=error")
	s.d.Stop()
	content, _ = ioutil.ReadFile(s.d.logFile.Name())
	if strings.Contains(string(content), "....") {
		c.Fatalf("Error level should not have ....\n%s", string(content))
	}

	s.d.Start("--log-level=info")
	s.d.Stop()
	content, _ = ioutil.ReadFile(s.d.logFile.Name())
	if !strings.Contains(string(content), "....") {
		c.Fatalf("Info level should have ....\n%s", string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonUnixSockCleanedUp(c *check.C) {
	dir, err := ioutil.TempDir("", "socket-cleanup-test")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(dir)

	sockPath := filepath.Join(dir, "docker.sock")
	if err := s.d.Start("--host", "unix://"+sockPath); err != nil {
		c.Fatal(err)
	}

	if _, err := os.Stat(sockPath); err != nil {
		c.Fatal("socket does not exist")
	}

	if err := s.d.Stop(); err != nil {
		c.Fatal(err)
	}

	if _, err := os.Stat(sockPath); err == nil || !os.IsNotExist(err) {
		c.Fatal("unix socket is not cleaned up")
	}
}

func (s *DockerDaemonSuite) TestDaemonWithWrongkey(c *check.C) {
	type Config struct {
		Crv string `json:"crv"`
		D   string `json:"d"`
		Kid string `json:"kid"`
		Kty string `json:"kty"`
		X   string `json:"x"`
		Y   string `json:"y"`
	}

	os.Remove("/etc/docker/key.json")
	if err := s.d.Start(); err != nil {
		c.Fatalf("Failed to start daemon: %v", err)
	}

	if err := s.d.Stop(); err != nil {
		c.Fatalf("Could not stop daemon: %v", err)
	}

	config := &Config{}
	bytes, err := ioutil.ReadFile("/etc/docker/key.json")
	if err != nil {
		c.Fatalf("Error reading key.json file: %s", err)
	}

	// byte[] to Data-Struct
	if err := json.Unmarshal(bytes, &config); err != nil {
		c.Fatalf("Error Unmarshal: %s", err)
	}

	//replace config.Kid with the fake value
	config.Kid = "VSAJ:FUYR:X3H2:B2VZ:KZ6U:CJD5:K7BX:ZXHY:UZXT:P4FT:MJWG:HRJ4"

	// NEW Data-Struct to byte[]
	newBytes, err := json.Marshal(&config)
	if err != nil {
		c.Fatalf("Error Marshal: %s", err)
	}

	// write back
	if err := ioutil.WriteFile("/etc/docker/key.json", newBytes, 0400); err != nil {
		c.Fatalf("Error ioutil.WriteFile: %s", err)
	}

	defer os.Remove("/etc/docker/key.json")

	if err := s.d.Start(); err == nil {
		c.Fatalf("It should not be successful to start daemon with wrong key: %v", err)
	}

	content, _ := ioutil.ReadFile(s.d.logFile.Name())

	if !strings.Contains(string(content), "Public Key ID does not match") {
		c.Fatal("Missing KeyID message from daemon logs")
	}
}

func (s *DockerDaemonSuite) TestDaemonRestartKillWait(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	out, err := s.d.Cmd("run", "-id", "busybox", "/bin/cat")
	if err != nil {
		c.Fatalf("Could not run /bin/cat: err=%v\n%s", err, out)
	}
	containerID := strings.TrimSpace(out)

	if out, err := s.d.Cmd("kill", containerID); err != nil {
		c.Fatalf("Could not kill %s: err=%v\n%s", containerID, err, out)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatalf("Could not restart daemon: %v", err)
	}

	errchan := make(chan error)
	go func() {
		if out, err := s.d.Cmd("wait", containerID); err != nil {
			errchan <- fmt.Errorf("%v:\n%s", err, out)
		}
		close(errchan)
	}()

	select {
	case <-time.After(5 * time.Second):
		c.Fatal("Waiting on a stopped (killed) container timed out")
	case err := <-errchan:
		if err != nil {
			c.Fatal(err)
		}
	}
}

// TestHttpsInfo connects via two-way authenticated HTTPS to the info endpoint
func (s *DockerDaemonSuite) TestHttpsInfo(c *check.C) {
	const (
		testDaemonHttpsAddr = "localhost:4271"
	)

	if err := s.d.Start("--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem", "-H", testDaemonHttpsAddr); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	//force tcp protocol
	host := fmt.Sprintf("tcp://%s", testDaemonHttpsAddr)
	daemonArgs := []string{"--host", host, "--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/client-cert.pem", "--tlskey", "fixtures/https/client-key.pem"}
	out, err := s.d.CmdWithArgs(daemonArgs, "info")
	if err != nil {
		c.Fatalf("Error Occurred: %s and output: %s", err, out)
	}
}

// TestHttpsInfoRogueCert connects via two-way authenticated HTTPS to the info endpoint
// by using a rogue client certificate and checks that it fails with the expected error.
func (s *DockerDaemonSuite) TestHttpsInfoRogueCert(c *check.C) {
	const (
		errBadCertificate   = "remote error: bad certificate"
		testDaemonHttpsAddr = "localhost:4271"
	)
	if err := s.d.Start("--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem", "-H", testDaemonHttpsAddr); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	//force tcp protocol
	host := fmt.Sprintf("tcp://%s", testDaemonHttpsAddr)
	daemonArgs := []string{"--host", host, "--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/client-rogue-cert.pem", "--tlskey", "fixtures/https/client-rogue-key.pem"}
	out, err := s.d.CmdWithArgs(daemonArgs, "info")
	if err == nil || !strings.Contains(out, errBadCertificate) {
		c.Fatalf("Expected err: %s, got instead: %s and output: %s", errBadCertificate, err, out)
	}
}

// TestHttpsInfoRogueServerCert connects via two-way authenticated HTTPS to the info endpoint
// which provides a rogue server certificate and checks that it fails with the expected error
func (s *DockerDaemonSuite) TestHttpsInfoRogueServerCert(c *check.C) {
	const (
		errCaUnknown             = "x509: certificate signed by unknown authority"
		testDaemonRogueHttpsAddr = "localhost:4272"
	)
	if err := s.d.Start("--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/server-rogue-cert.pem",
		"--tlskey", "fixtures/https/server-rogue-key.pem", "-H", testDaemonRogueHttpsAddr); err != nil {
		c.Fatalf("Could not start daemon with busybox: %v", err)
	}

	//force tcp protocol
	host := fmt.Sprintf("tcp://%s", testDaemonRogueHttpsAddr)
	daemonArgs := []string{"--host", host, "--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/client-rogue-cert.pem", "--tlskey", "fixtures/https/client-rogue-key.pem"}
	out, err := s.d.CmdWithArgs(daemonArgs, "info")
	if err == nil || !strings.Contains(out, errCaUnknown) {
		c.Fatalf("Expected err: %s, got instead: %s and output: %s", errCaUnknown, err, out)
	}
}

func pingContainers(c *check.C, d *Daemon, expectFailure bool) {
	var dargs []string
	if d != nil {
		dargs = []string{"--host", d.sock()}
	}

	args := append(dargs, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, args...)

	args = append(dargs, "run", "--rm", "--link", "container1:alias1", "busybox", "sh", "-c")
	pingCmd := "ping -c 1 %s -W 1"
	args = append(args, fmt.Sprintf(pingCmd, "alias1"))
	_, _, err := dockerCmdWithError(c, args...)

	if expectFailure {
		c.Assert(err, check.NotNil)
	} else {
		c.Assert(err, check.IsNil)
	}

	args = append(dargs, "rm", "-f", "container1")
	dockerCmd(c, args...)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithSocketAsVolume(c *check.C) {
	c.Assert(s.d.StartWithBusybox(), check.IsNil)

	socket := filepath.Join(s.d.folder, "docker.sock")

	out, err := s.d.Cmd("run", "-d", "-v", socket+":/sock", "busybox")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(s.d.Restart(), check.IsNil)
}

func (s *DockerDaemonSuite) TestCleanupMountsAfterCrash(c *check.C) {
	c.Assert(s.d.StartWithBusybox(), check.IsNil)

	out, err := s.d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	id := strings.TrimSpace(out)
	c.Assert(s.d.cmd.Process.Signal(os.Kill), check.IsNil)
	c.Assert(s.d.Start(), check.IsNil)
	mountOut, err := exec.Command("mount").CombinedOutput()
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, false, check.Commentf("Something mounted from older daemon start: %s", mountOut))
}

func (s *DockerDaemonSuite) TestRunContainerWithBridgeNone(c *check.C) {
	testRequires(c, NativeExecDriver)
	c.Assert(s.d.StartWithBusybox("-b", "none"), check.IsNil)

	out, err := s.d.Cmd("run", "--rm", "busybox", "ip", "l")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.Contains(out, "eth0"), check.Equals, false,
		check.Commentf("There shouldn't be eth0 in container in default(bridge) mode when bridge network is disabled: %s", out))

	out, err = s.d.Cmd("run", "--rm", "--net=bridge", "busybox", "ip", "l")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.Contains(out, "eth0"), check.Equals, false,
		check.Commentf("There shouldn't be eth0 in container in bridge mode when bridge network is disabled: %s", out))

	out, err = s.d.Cmd("run", "--rm", "--net=host", "busybox", "ip", "l")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.Contains(out, "eth0"), check.Equals, true,
		check.Commentf("There should be eth0 in container when --net=host when bridge network is disabled: %s", out))
}

func (s *DockerDaemonSuite) TestDaemonRestartWithContainerRunning(t *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		t.Fatal(err)
	}
	if out, err := s.d.Cmd("run", "-ti", "-d", "--name", "test", "busybox"); err != nil {
		t.Fatal(out, err)
	}

	if err := s.d.Restart(); err != nil {
		t.Fatal(err)
	}
	// Container 'test' should be removed without error
	if out, err := s.d.Cmd("rm", "test"); err != nil {
		t.Fatal(out, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonRestartCleanupNetns(c *check.C) {
	if err := s.d.StartWithBusybox(); err != nil {
		c.Fatal(err)
	}
	out, err := s.d.Cmd("run", "--name", "netns", "-d", "busybox", "top")
	if err != nil {
		c.Fatal(out, err)
	}
	if out, err := s.d.Cmd("stop", "netns"); err != nil {
		c.Fatal(out, err)
	}

	// Construct netns file name from container id
	out = strings.TrimSpace(out)
	nsFile := out[:12]

	// Test if the file still exists
	out, _, err = runCommandWithOutput(exec.Command("stat", "-c", "%n", "/var/run/docker/netns/"+nsFile))
	out = strings.TrimSpace(out)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(out, check.Equals, "/var/run/docker/netns/"+nsFile, check.Commentf("Output: %s", out))

	// Remove the container and restart the daemon
	if out, err := s.d.Cmd("rm", "netns"); err != nil {
		c.Fatal(out, err)
	}

	if err := s.d.Restart(); err != nil {
		c.Fatal(err)
	}

	// Test again and see now the netns file does not exist
	out, _, err = runCommandWithOutput(exec.Command("stat", "-c", "%n", "/var/run/docker/netns/"+nsFile))
	out = strings.TrimSpace(out)
	c.Assert(err, check.Not(check.IsNil), check.Commentf("Output: %s", out))
	// c.Assert(out, check.Equals, "", check.Commentf("Output: %s", out))
}

// tests regression detailed in #13964 where DOCKER_TLS_VERIFY env is ignored
func (s *DockerDaemonSuite) TestDaemonNoTlsCliTlsVerifyWithEnv(c *check.C) {
	host := "tcp://localhost:4271"
	c.Assert(s.d.Start("-H", host), check.IsNil)
	cmd := exec.Command(dockerBinary, "-H", host, "info")
	cmd.Env = []string{"DOCKER_TLS_VERIFY=1", "DOCKER_CERT_PATH=fixtures/https"}
	out, _, err := runCommandWithOutput(cmd)
	c.Assert(err, check.Not(check.IsNil), check.Commentf("%s", out))
	c.Assert(strings.Contains(out, "error occurred trying to connect"), check.Equals, true)

}

func setupV6() error {
	// Hack to get the right IPv6 address on docker0, which has already been created
	err := exec.Command("ip", "addr", "add", "fe80::1/64", "dev", "docker0").Run()
	if err != nil {
		return err
	}
	return nil
}

func teardownV6() error {
	err := exec.Command("ip", "addr", "del", "fe80::1/64", "dev", "docker0").Run()
	if err != nil {
		return err
	}
	return nil
}
