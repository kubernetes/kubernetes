// +build linux

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"crypto/tls"
	"crypto/x509"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/integration-cli/daemon"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/testutil"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	units "github.com/docker/go-units"
	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libtrust"
	"github.com/go-check/check"
	"github.com/kr/pty"
	"golang.org/x/sys/unix"
)

// TestLegacyDaemonCommand test starting docker daemon using "deprecated" docker daemon
// command. Remove this test when we remove this.
func (s *DockerDaemonSuite) TestLegacyDaemonCommand(c *check.C) {
	cmd := exec.Command(dockerBinary, "daemon", "--storage-driver=vfs", "--debug")
	err := cmd.Start()
	c.Assert(err, checker.IsNil, check.Commentf("could not start daemon using 'docker daemon'"))

	c.Assert(cmd.Process.Kill(), checker.IsNil)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithRunningContainersPorts(c *check.C) {
	s.d.StartWithBusybox(c)

	cli.Docker(
		cli.Args("run", "-d", "--name", "top1", "-p", "1234:80", "--restart", "always", "busybox:latest", "top"),
		cli.Daemon(s.d),
	).Assert(c, icmd.Success)

	cli.Docker(
		cli.Args("run", "-d", "--name", "top2", "-p", "80", "busybox:latest", "top"),
		cli.Daemon(s.d),
	).Assert(c, icmd.Success)

	testRun := func(m map[string]bool, prefix string) {
		var format string
		for cont, shouldRun := range m {
			out := cli.Docker(cli.Args("ps"), cli.Daemon(s.d)).Assert(c, icmd.Success).Combined()
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

	s.d.Restart(c)
	testRun(map[string]bool{"top1": true, "top2": false}, "After daemon restart: ")
}

func (s *DockerDaemonSuite) TestDaemonRestartWithVolumesRefs(c *check.C) {
	s.d.StartWithBusybox(c)

	if out, err := s.d.Cmd("run", "--name", "volrestarttest1", "-v", "/foo", "busybox"); err != nil {
		c.Fatal(err, out)
	}

	s.d.Restart(c)

	if out, err := s.d.Cmd("run", "-d", "--volumes-from", "volrestarttest1", "--name", "volrestarttest2", "busybox", "top"); err != nil {
		c.Fatal(err, out)
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

// #11008
func (s *DockerDaemonSuite) TestDaemonRestartUnlessStopped(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--name", "top1", "--restart", "always", "busybox:latest", "top")
	c.Assert(err, check.IsNil, check.Commentf("run top1: %v", out))

	out, err = s.d.Cmd("run", "-d", "--name", "top2", "--restart", "unless-stopped", "busybox:latest", "top")
	c.Assert(err, check.IsNil, check.Commentf("run top2: %v", out))

	testRun := func(m map[string]bool, prefix string) {
		var format string
		for name, shouldRun := range m {
			out, err := s.d.Cmd("ps")
			c.Assert(err, check.IsNil, check.Commentf("run ps: %v", out))
			if shouldRun {
				format = "%scontainer %q is not running"
			} else {
				format = "%scontainer %q is running"
			}
			c.Assert(strings.Contains(out, name), check.Equals, shouldRun, check.Commentf(format, prefix, name))
		}
	}

	// both running
	testRun(map[string]bool{"top1": true, "top2": true}, "")

	out, err = s.d.Cmd("stop", "top1")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("stop", "top2")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// both stopped
	testRun(map[string]bool{"top1": false, "top2": false}, "")

	s.d.Restart(c)

	// restart=always running
	testRun(map[string]bool{"top1": true, "top2": false}, "After daemon restart: ")

	out, err = s.d.Cmd("start", "top2")
	c.Assert(err, check.IsNil, check.Commentf("start top2: %v", out))

	s.d.Restart(c)

	// both running
	testRun(map[string]bool{"top1": true, "top2": true}, "After second daemon restart: ")

}

func (s *DockerDaemonSuite) TestDaemonRestartOnFailure(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--name", "test1", "--restart", "on-failure:3", "busybox:latest", "false")
	c.Assert(err, check.IsNil, check.Commentf("run top1: %v", out))

	// wait test1 to stop
	hostArgs := []string{"--host", s.d.Sock()}
	err = waitInspectWithArgs("test1", "{{.State.Running}} {{.State.Restarting}}", "false false", 10*time.Second, hostArgs...)
	c.Assert(err, checker.IsNil, check.Commentf("test1 should exit but not"))

	// record last start time
	out, err = s.d.Cmd("inspect", "-f={{.State.StartedAt}}", "test1")
	c.Assert(err, checker.IsNil, check.Commentf("out: %v", out))
	lastStartTime := out

	s.d.Restart(c)

	// test1 shouldn't restart at all
	err = waitInspectWithArgs("test1", "{{.State.Running}} {{.State.Restarting}}", "false false", 0, hostArgs...)
	c.Assert(err, checker.IsNil, check.Commentf("test1 should exit but not"))

	// make sure test1 isn't restarted when daemon restart
	// if "StartAt" time updates, means test1 was once restarted.
	out, err = s.d.Cmd("inspect", "-f={{.State.StartedAt}}", "test1")
	c.Assert(err, checker.IsNil, check.Commentf("out: %v", out))
	c.Assert(out, checker.Equals, lastStartTime, check.Commentf("test1 shouldn't start after daemon restarts"))
}

func (s *DockerDaemonSuite) TestDaemonStartIptablesFalse(c *check.C) {
	s.d.Start(c, "--iptables=false")
}

// Make sure we cannot shrink base device at daemon restart.
func (s *DockerDaemonSuite) TestDaemonRestartWithInvalidBasesize(c *check.C) {
	testRequires(c, Devicemapper)
	s.d.Start(c)

	oldBasesizeBytes := s.d.GetBaseDeviceSize(c)
	var newBasesizeBytes int64 = 1073741824 //1GB in bytes

	if newBasesizeBytes < oldBasesizeBytes {
		err := s.d.RestartWithError("--storage-opt", fmt.Sprintf("dm.basesize=%d", newBasesizeBytes))
		c.Assert(err, check.NotNil, check.Commentf("daemon should not have started as new base device size is less than existing base device size: %v", err))
		// 'err != nil' is expected behaviour, no new daemon started,
		// so no need to stop daemon.
		if err != nil {
			return
		}
	}
	s.d.Stop(c)
}

// Make sure we can grow base device at daemon restart.
func (s *DockerDaemonSuite) TestDaemonRestartWithIncreasedBasesize(c *check.C) {
	testRequires(c, Devicemapper)
	s.d.Start(c)

	oldBasesizeBytes := s.d.GetBaseDeviceSize(c)

	var newBasesizeBytes int64 = 53687091200 //50GB in bytes

	if newBasesizeBytes < oldBasesizeBytes {
		c.Skip(fmt.Sprintf("New base device size (%v) must be greater than (%s)", units.HumanSize(float64(newBasesizeBytes)), units.HumanSize(float64(oldBasesizeBytes))))
	}

	err := s.d.RestartWithError("--storage-opt", fmt.Sprintf("dm.basesize=%d", newBasesizeBytes))
	c.Assert(err, check.IsNil, check.Commentf("we should have been able to start the daemon with increased base device size: %v", err))

	basesizeAfterRestart := s.d.GetBaseDeviceSize(c)
	newBasesize, err := convertBasesize(newBasesizeBytes)
	c.Assert(err, check.IsNil, check.Commentf("Error in converting base device size: %v", err))
	c.Assert(newBasesize, check.Equals, basesizeAfterRestart, check.Commentf("Basesize passed is not equal to Basesize set"))
	s.d.Stop(c)
}

func convertBasesize(basesizeBytes int64) (int64, error) {
	basesize := units.HumanSize(float64(basesizeBytes))
	basesize = strings.Trim(basesize, " ")[:len(basesize)-3]
	basesizeFloat, err := strconv.ParseFloat(strings.Trim(basesize, " "), 64)
	if err != nil {
		return 0, err
	}
	return int64(basesizeFloat) * 1024 * 1024 * 1024, nil
}

// Issue #8444: If docker0 bridge is modified (intentionally or unintentionally) and
// no longer has an IP associated, we should gracefully handle that case and associate
// an IP with it rather than fail daemon start
func (s *DockerDaemonSuite) TestDaemonStartBridgeWithoutIPAssociation(c *check.C) {
	// rather than depending on brctl commands to verify docker0 is created and up
	// let's start the daemon and stop it, and then make a modification to run the
	// actual test
	s.d.Start(c)
	s.d.Stop(c)

	// now we will remove the ip from docker0 and then try starting the daemon
	icmd.RunCommand("ip", "addr", "flush", "dev", "docker0").Assert(c, icmd.Success)

	if err := s.d.StartWithError(); err != nil {
		warning := "**WARNING: Docker bridge network in bad state--delete docker0 bridge interface to fix"
		c.Fatalf("Could not start daemon when docker0 has no IP address: %v\n%s", err, warning)
	}
}

func (s *DockerDaemonSuite) TestDaemonIptablesClean(c *check.C) {
	s.d.StartWithBusybox(c)

	if out, err := s.d.Cmd("run", "-d", "--name", "top", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top: %s, %v", out, err)
	}

	ipTablesSearchString := "tcp dpt:80"

	// get output from iptables with container running
	verifyIPTablesContains(c, ipTablesSearchString)

	s.d.Stop(c)

	// get output from iptables after restart
	verifyIPTablesDoesNotContains(c, ipTablesSearchString)
}

func (s *DockerDaemonSuite) TestDaemonIptablesCreate(c *check.C) {
	s.d.StartWithBusybox(c)

	if out, err := s.d.Cmd("run", "-d", "--name", "top", "--restart=always", "-p", "80", "busybox:latest", "top"); err != nil {
		c.Fatalf("Could not run top: %s, %v", out, err)
	}

	// get output from iptables with container running
	ipTablesSearchString := "tcp dpt:80"
	verifyIPTablesContains(c, ipTablesSearchString)

	s.d.Restart(c)

	// make sure the container is not running
	runningOut, err := s.d.Cmd("inspect", "--format={{.State.Running}}", "top")
	if err != nil {
		c.Fatalf("Could not inspect on container: %s, %v", runningOut, err)
	}
	if strings.TrimSpace(runningOut) != "true" {
		c.Fatalf("Container should have been restarted after daemon restart. Status running should have been true but was: %q", strings.TrimSpace(runningOut))
	}

	// get output from iptables after restart
	verifyIPTablesContains(c, ipTablesSearchString)
}

func verifyIPTablesContains(c *check.C, ipTablesSearchString string) {
	result := icmd.RunCommand("iptables", "-nvL")
	result.Assert(c, icmd.Success)
	if !strings.Contains(result.Combined(), ipTablesSearchString) {
		c.Fatalf("iptables output should have contained %q, but was %q", ipTablesSearchString, result.Combined())
	}
}

func verifyIPTablesDoesNotContains(c *check.C, ipTablesSearchString string) {
	result := icmd.RunCommand("iptables", "-nvL")
	result.Assert(c, icmd.Success)
	if strings.Contains(result.Combined(), ipTablesSearchString) {
		c.Fatalf("iptables output should not have contained %q, but was %q", ipTablesSearchString, result.Combined())
	}
}

// TestDaemonIPv6Enabled checks that when the daemon is started with --ipv6=true that the docker0 bridge
// has the fe80::1 address and that a container is assigned a link-local address
func (s *DockerDaemonSuite) TestDaemonIPv6Enabled(c *check.C) {
	testRequires(c, IPv6)

	setupV6(c)
	defer teardownV6(c)

	s.d.StartWithBusybox(c, "--ipv6")

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
			break
		}
	}

	if !found {
		c.Fatalf("Bridge does not have an IPv6 Address")
	}

	if out, err := s.d.Cmd("run", "-itd", "--name=ipv6test", "busybox:latest"); err != nil {
		c.Fatalf("Could not run container: %s, %v", out, err)
	}

	out, err := s.d.Cmd("inspect", "--format", "'{{.NetworkSettings.Networks.bridge.LinkLocalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip == nil {
		c.Fatalf("Container should have a link-local IPv6 address")
	}

	out, err = s.d.Cmd("inspect", "--format", "'{{.NetworkSettings.Networks.bridge.GlobalIPv6Address}}'", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}

	if ip := net.ParseIP(out); ip != nil {
		c.Fatalf("Container should not have a global IPv6 address: %v", out)
	}
}

// TestDaemonIPv6FixedCIDR checks that when the daemon is started with --ipv6=true and a fixed CIDR
// that running containers are given a link-local and global IPv6 address
func (s *DockerDaemonSuite) TestDaemonIPv6FixedCIDR(c *check.C) {
	// IPv6 setup is messing with local bridge address.
	testRequires(c, SameHostDaemon)
	// Delete the docker0 bridge if its left around from previous daemon. It has to be recreated with
	// ipv6 enabled
	deleteInterface(c, "docker0")

	s.d.StartWithBusybox(c, "--ipv6", "--fixed-cidr-v6=2001:db8:2::/64", "--default-gateway-v6=2001:db8:2::100")

	out, err := s.d.Cmd("run", "-itd", "--name=ipv6test", "busybox:latest")
	c.Assert(err, checker.IsNil, check.Commentf("Could not run container: %s, %v", out, err))

	out, err = s.d.Cmd("inspect", "--format", "{{.NetworkSettings.Networks.bridge.GlobalIPv6Address}}", "ipv6test")
	out = strings.Trim(out, " \r\n'")

	c.Assert(err, checker.IsNil, check.Commentf(out))

	ip := net.ParseIP(out)
	c.Assert(ip, checker.NotNil, check.Commentf("Container should have a global IPv6 address"))

	out, err = s.d.Cmd("inspect", "--format", "{{.NetworkSettings.Networks.bridge.IPv6Gateway}}", "ipv6test")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	c.Assert(strings.Trim(out, " \r\n'"), checker.Equals, "2001:db8:2::100", check.Commentf("Container should have a global IPv6 gateway"))
}

// TestDaemonIPv6FixedCIDRAndMac checks that when the daemon is started with ipv6 fixed CIDR
// the running containers are given an IPv6 address derived from the MAC address and the ipv6 fixed CIDR
func (s *DockerDaemonSuite) TestDaemonIPv6FixedCIDRAndMac(c *check.C) {
	// IPv6 setup is messing with local bridge address.
	testRequires(c, SameHostDaemon)
	// Delete the docker0 bridge if its left around from previous daemon. It has to be recreated with
	// ipv6 enabled
	deleteInterface(c, "docker0")

	s.d.StartWithBusybox(c, "--ipv6", "--fixed-cidr-v6=2001:db8:1::/64")

	out, err := s.d.Cmd("run", "-itd", "--name=ipv6test", "--mac-address", "AA:BB:CC:DD:EE:FF", "busybox")
	c.Assert(err, checker.IsNil)

	out, err = s.d.Cmd("inspect", "--format", "{{.NetworkSettings.Networks.bridge.GlobalIPv6Address}}", "ipv6test")
	c.Assert(err, checker.IsNil)
	c.Assert(strings.Trim(out, " \r\n'"), checker.Equals, "2001:db8:1::aabb:ccdd:eeff")
}

// TestDaemonIPv6HostMode checks that when the running a container with
// network=host the host ipv6 addresses are not removed
func (s *DockerDaemonSuite) TestDaemonIPv6HostMode(c *check.C) {
	testRequires(c, SameHostDaemon)
	deleteInterface(c, "docker0")

	s.d.StartWithBusybox(c, "--ipv6", "--fixed-cidr-v6=2001:db8:2::/64")
	out, err := s.d.Cmd("run", "-itd", "--name=hostcnt", "--network=host", "busybox:latest")
	c.Assert(err, checker.IsNil, check.Commentf("Could not run container: %s, %v", out, err))

	out, err = s.d.Cmd("exec", "hostcnt", "ip", "-6", "addr", "show", "docker0")
	out = strings.Trim(out, " \r\n'")

	c.Assert(out, checker.Contains, "2001:db8:2::1")
}

func (s *DockerDaemonSuite) TestDaemonLogLevelWrong(c *check.C) {
	c.Assert(s.d.StartWithError("--log-level=bogus"), check.NotNil, check.Commentf("Daemon shouldn't start with wrong log level"))
}

func (s *DockerDaemonSuite) TestDaemonLogLevelDebug(c *check.C) {
	s.d.Start(c, "--log-level=debug")
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Missing level="debug" in log file:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonLogLevelFatal(c *check.C) {
	// we creating new daemons to create new logFile
	s.d.Start(c, "--log-level=fatal")
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	if strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should not have level="debug" in log file:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagD(c *check.C) {
	s.d.Start(c, "-D")
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should have level="debug" in log file using -D:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagDebug(c *check.C) {
	s.d.Start(c, "--debug")
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	if !strings.Contains(string(content), `level=debug`) {
		c.Fatalf(`Should have level="debug" in log file using --debug:\n%s`, string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonFlagDebugLogLevelFatal(c *check.C) {
	s.d.Start(c, "--debug", "--log-level=fatal")
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
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

	cmdArgs := make([]string, 0, len(listeningPorts)*2)
	for _, hostDirective := range listeningPorts {
		cmdArgs = append(cmdArgs, "--host", fmt.Sprintf("tcp://%s:%s", hostDirective[0], hostDirective[2]))
	}

	s.d.StartWithBusybox(c, cmdArgs...)

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
	s.d.Start(c)
	s.d.Stop(c)

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

// GH#11320 - verify that the daemon exits on failure properly
// Note that this explicitly tests the conflict of {-b,--bridge} and {--bip} options as the means
// to get a daemon init failure; no other tests for -b/--bip conflict are therefore required
func (s *DockerDaemonSuite) TestDaemonExitOnFailure(c *check.C) {
	//attempt to start daemon with incorrect flags (we know -b and --bip conflict)
	if err := s.d.StartWithError("--bridge", "nosuchbridge", "--bip", "1.1.1.1"); err != nil {
		//verify we got the right error
		if !strings.Contains(err.Error(), "Daemon exited") {
			c.Fatalf("Expected daemon not to start, got %v", err)
		}
		// look in the log and make sure we got the message that daemon is shutting down
		icmd.RunCommand("grep", "Error starting daemon", s.d.LogFileName()).Assert(c, icmd.Success)
	} else {
		//if we didn't get an error and the daemon is running, this is a failure
		c.Fatal("Conflicting options should cause the daemon to error out with a failure")
	}
}

func (s *DockerDaemonSuite) TestDaemonBridgeExternal(c *check.C) {
	d := s.d
	err := d.StartWithError("--bridge", "nosuchbridge")
	c.Assert(err, check.NotNil, check.Commentf("--bridge option with an invalid bridge should cause the daemon to fail"))
	defer d.Restart(c)

	bridgeName := "external-bridge"
	bridgeIP := "192.169.1.1/24"
	_, bridgeIPNet, _ := net.ParseCIDR(bridgeIP)

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	d.StartWithBusybox(c, "--bridge", bridgeName)

	ipTablesSearchString := bridgeIPNet.String()
	icmd.RunCommand("iptables", "-t", "nat", "-nvL").Assert(c, icmd.Expected{
		Out: ipTablesSearchString,
	})

	_, err = d.Cmd("run", "-d", "--name", "ExtContainer", "busybox", "top")
	c.Assert(err, check.IsNil)

	containerIP, err := d.FindContainerIP("ExtContainer")
	c.Assert(err, checker.IsNil)
	ip := net.ParseIP(containerIP)
	c.Assert(bridgeIPNet.Contains(ip), check.Equals, true,
		check.Commentf("Container IP-Address must be in the same subnet range : %s",
			containerIP))
}

func (s *DockerDaemonSuite) TestDaemonBridgeNone(c *check.C) {
	// start with bridge none
	d := s.d
	d.StartWithBusybox(c, "--bridge", "none")
	defer d.Restart(c)

	// verify docker0 iface is not there
	icmd.RunCommand("ifconfig", "docker0").Assert(c, icmd.Expected{
		ExitCode: 1,
		Error:    "exit status 1",
		Err:      "Device not found",
	})

	// verify default "bridge" network is not there
	out, err := d.Cmd("network", "inspect", "bridge")
	c.Assert(err, check.NotNil, check.Commentf("\"bridge\" network should not be present if daemon started with --bridge=none"))
	c.Assert(strings.Contains(out, "No such network"), check.Equals, true)
}

func createInterface(c *check.C, ifType string, ifName string, ipNet string) {
	icmd.RunCommand("ip", "link", "add", "name", ifName, "type", ifType).Assert(c, icmd.Success)
	icmd.RunCommand("ifconfig", ifName, ipNet, "up").Assert(c, icmd.Success)
}

func deleteInterface(c *check.C, ifName string) {
	icmd.RunCommand("ip", "link", "delete", ifName).Assert(c, icmd.Success)
	icmd.RunCommand("iptables", "-t", "nat", "--flush").Assert(c, icmd.Success)
	icmd.RunCommand("iptables", "--flush").Assert(c, icmd.Success)
}

func (s *DockerDaemonSuite) TestDaemonBridgeIP(c *check.C) {
	// TestDaemonBridgeIP Steps
	// 1. Delete the existing docker0 Bridge
	// 2. Set --bip daemon configuration and start the new Docker Daemon
	// 3. Check if the bip config has taken effect using ifconfig and iptables commands
	// 4. Launch a Container and make sure the IP-Address is in the expected subnet
	// 5. Delete the docker0 Bridge
	// 6. Restart the Docker Daemon (via deferred action)
	//    This Restart takes care of bringing docker0 interface back to auto-assigned IP

	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIP := "192.169.1.1/24"
	ip, bridgeIPNet, _ := net.ParseCIDR(bridgeIP)

	d.StartWithBusybox(c, "--bip", bridgeIP)
	defer d.Restart(c)

	ifconfigSearchString := ip.String()
	icmd.RunCommand("ifconfig", defaultNetworkBridge).Assert(c, icmd.Expected{
		Out: ifconfigSearchString,
	})

	ipTablesSearchString := bridgeIPNet.String()
	icmd.RunCommand("iptables", "-t", "nat", "-nvL").Assert(c, icmd.Expected{
		Out: ipTablesSearchString,
	})

	_, err := d.Cmd("run", "-d", "--name", "test", "busybox", "top")
	c.Assert(err, check.IsNil)

	containerIP, err := d.FindContainerIP("test")
	c.Assert(err, checker.IsNil)
	ip = net.ParseIP(containerIP)
	c.Assert(bridgeIPNet.Contains(ip), check.Equals, true,
		check.Commentf("Container IP-Address must be in the same subnet range : %s",
			containerIP))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithBridgeIPChange(c *check.C) {
	s.d.Start(c)
	defer s.d.Restart(c)
	s.d.Stop(c)

	// now we will change the docker0's IP and then try starting the daemon
	bridgeIP := "192.169.100.1/24"
	_, bridgeIPNet, _ := net.ParseCIDR(bridgeIP)

	icmd.RunCommand("ifconfig", "docker0", bridgeIP).Assert(c, icmd.Success)

	s.d.Start(c, "--bip", bridgeIP)

	//check if the iptables contains new bridgeIP MASQUERADE rule
	ipTablesSearchString := bridgeIPNet.String()
	icmd.RunCommand("iptables", "-t", "nat", "-nvL").Assert(c, icmd.Expected{
		Out: ipTablesSearchString,
	})
}

func (s *DockerDaemonSuite) TestDaemonBridgeFixedCidr(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIP := "192.169.1.1/24"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	args := []string{"--bridge", bridgeName, "--fixed-cidr", "192.169.1.0/30"}
	d.StartWithBusybox(c, args...)
	defer d.Restart(c)

	for i := 0; i < 4; i++ {
		cName := "Container" + strconv.Itoa(i)
		out, err := d.Cmd("run", "-d", "--name", cName, "busybox", "top")
		if err != nil {
			c.Assert(strings.Contains(out, "no available IPv4 addresses"), check.Equals, true,
				check.Commentf("Could not run a Container : %s %s", err.Error(), out))
		}
	}
}

func (s *DockerDaemonSuite) TestDaemonBridgeFixedCidr2(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIP := "10.2.2.1/16"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	d.StartWithBusybox(c, "--bip", bridgeIP, "--fixed-cidr", "10.2.2.0/24")
	defer s.d.Restart(c)

	out, err := d.Cmd("run", "-d", "--name", "bb", "busybox", "top")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	defer d.Cmd("stop", "bb")

	out, err = d.Cmd("exec", "bb", "/bin/sh", "-c", "ifconfig eth0 | awk '/inet addr/{print substr($2,6)}'")
	c.Assert(out, checker.Equals, "10.2.2.0\n")

	out, err = d.Cmd("run", "--rm", "busybox", "/bin/sh", "-c", "ifconfig eth0 | awk '/inet addr/{print substr($2,6)}'")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	c.Assert(out, checker.Equals, "10.2.2.2\n")
}

func (s *DockerDaemonSuite) TestDaemonBridgeFixedCIDREqualBridgeNetwork(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIP := "172.27.42.1/16"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	d.StartWithBusybox(c, "--bridge", bridgeName, "--fixed-cidr", bridgeIP)
	defer s.d.Restart(c)

	out, err := d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))
	cid1 := strings.TrimSpace(out)
	defer d.Cmd("stop", cid1)
}

func (s *DockerDaemonSuite) TestDaemonDefaultGatewayIPv4Implicit(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIP := "192.169.1.1"
	bridgeIPNet := fmt.Sprintf("%s/24", bridgeIP)

	d.StartWithBusybox(c, "--bip", bridgeIPNet)
	defer d.Restart(c)

	expectedMessage := fmt.Sprintf("default via %s dev", bridgeIP)
	out, err := d.Cmd("run", "busybox", "ip", "-4", "route", "list", "0/0")
	c.Assert(err, checker.IsNil)
	c.Assert(strings.Contains(out, expectedMessage), check.Equals, true,
		check.Commentf("Implicit default gateway should be bridge IP %s, but default route was '%s'",
			bridgeIP, strings.TrimSpace(out)))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonDefaultGatewayIPv4Explicit(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	d := s.d

	bridgeIP := "192.169.1.1"
	bridgeIPNet := fmt.Sprintf("%s/24", bridgeIP)
	gatewayIP := "192.169.1.254"

	d.StartWithBusybox(c, "--bip", bridgeIPNet, "--default-gateway", gatewayIP)
	defer d.Restart(c)

	expectedMessage := fmt.Sprintf("default via %s dev", gatewayIP)
	out, err := d.Cmd("run", "busybox", "ip", "-4", "route", "list", "0/0")
	c.Assert(err, checker.IsNil)
	c.Assert(strings.Contains(out, expectedMessage), check.Equals, true,
		check.Commentf("Explicit default gateway should be %s, but default route was '%s'",
			gatewayIP, strings.TrimSpace(out)))
	deleteInterface(c, defaultNetworkBridge)
}

func (s *DockerDaemonSuite) TestDaemonDefaultGatewayIPv4ExplicitOutsideContainerSubnet(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	// Program a custom default gateway outside of the container subnet, daemon should accept it and start
	s.d.StartWithBusybox(c, "--bip", "172.16.0.10/16", "--fixed-cidr", "172.16.1.0/24", "--default-gateway", "172.16.0.254")

	deleteInterface(c, defaultNetworkBridge)
	s.d.Restart(c)
}

func (s *DockerDaemonSuite) TestDaemonDefaultNetworkInvalidClusterConfig(c *check.C) {
	testRequires(c, DaemonIsLinux, SameHostDaemon)

	// Start daemon without docker0 bridge
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	discoveryBackend := "consul://consuladdr:consulport/some/path"
	s.d.Start(c, fmt.Sprintf("--cluster-store=%s", discoveryBackend))

	// Start daemon with docker0 bridge
	result := icmd.RunCommand("ifconfig", defaultNetworkBridge)
	c.Assert(result, icmd.Matches, icmd.Success)

	s.d.Restart(c, fmt.Sprintf("--cluster-store=%s", discoveryBackend))
}

func (s *DockerDaemonSuite) TestDaemonIP(c *check.C) {
	d := s.d

	ipStr := "192.170.1.1/24"
	ip, _, _ := net.ParseCIDR(ipStr)
	args := []string{"--ip", ip.String()}
	d.StartWithBusybox(c, args...)
	defer d.Restart(c)

	out, err := d.Cmd("run", "-d", "-p", "8000:8000", "busybox", "top")
	c.Assert(err, check.NotNil,
		check.Commentf("Running a container must fail with an invalid --ip option"))
	c.Assert(strings.Contains(out, "Error starting userland proxy"), check.Equals, true)

	ifName := "dummy"
	createInterface(c, "dummy", ifName, ipStr)
	defer deleteInterface(c, ifName)

	_, err = d.Cmd("run", "-d", "-p", "8000:8000", "busybox", "top")
	c.Assert(err, check.IsNil)

	result := icmd.RunCommand("iptables", "-t", "nat", "-nvL")
	result.Assert(c, icmd.Success)
	regex := fmt.Sprintf("DNAT.*%s.*dpt:8000", ip.String())
	matched, _ := regexp.MatchString(regex, result.Combined())
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, result.Combined()))
}

func (s *DockerDaemonSuite) TestDaemonICCPing(c *check.C) {
	testRequires(c, bridgeNfIptables)
	d := s.d

	bridgeName := "external-bridge"
	bridgeIP := "192.169.1.1/24"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	d.StartWithBusybox(c, "--bridge", bridgeName, "--icc=false")
	defer d.Restart(c)

	result := icmd.RunCommand("iptables", "-nvL", "FORWARD")
	result.Assert(c, icmd.Success)
	regex := fmt.Sprintf("DROP.*all.*%s.*%s", bridgeName, bridgeName)
	matched, _ := regexp.MatchString(regex, result.Combined())
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, result.Combined()))

	// Pinging another container must fail with --icc=false
	pingContainers(c, d, true)

	ipStr := "192.171.1.1/24"
	ip, _, _ := net.ParseCIDR(ipStr)
	ifName := "icc-dummy"

	createInterface(c, "dummy", ifName, ipStr)

	// But, Pinging external or a Host interface must succeed
	pingCmd := fmt.Sprintf("ping -c 1 %s -W 1", ip.String())
	runArgs := []string{"run", "--rm", "busybox", "sh", "-c", pingCmd}
	_, err := d.Cmd(runArgs...)
	c.Assert(err, check.IsNil)
}

func (s *DockerDaemonSuite) TestDaemonICCLinkExpose(c *check.C) {
	d := s.d

	bridgeName := "external-bridge"
	bridgeIP := "192.169.1.1/24"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	d.StartWithBusybox(c, "--bridge", bridgeName, "--icc=false")
	defer d.Restart(c)

	result := icmd.RunCommand("iptables", "-nvL", "FORWARD")
	result.Assert(c, icmd.Success)
	regex := fmt.Sprintf("DROP.*all.*%s.*%s", bridgeName, bridgeName)
	matched, _ := regexp.MatchString(regex, result.Combined())
	c.Assert(matched, check.Equals, true,
		check.Commentf("iptables output should have contained %q, but was %q", regex, result.Combined()))

	out, err := d.Cmd("run", "-d", "--expose", "4567", "--name", "icc1", "busybox", "nc", "-l", "-p", "4567")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = d.Cmd("run", "--link", "icc1:icc1", "busybox", "nc", "icc1", "4567")
	c.Assert(err, check.IsNil, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestDaemonLinksIpTablesRulesWhenLinkAndUnlink(c *check.C) {
	bridgeName := "external-bridge"
	bridgeIP := "192.169.1.1/24"

	createInterface(c, "bridge", bridgeName, bridgeIP)
	defer deleteInterface(c, bridgeName)

	s.d.StartWithBusybox(c, "--bridge", bridgeName, "--icc=false")
	defer s.d.Restart(c)

	_, err := s.d.Cmd("run", "-d", "--name", "child", "--publish", "8080:80", "busybox", "top")
	c.Assert(err, check.IsNil)
	_, err = s.d.Cmd("run", "-d", "--name", "parent", "--link", "child:http", "busybox", "top")
	c.Assert(err, check.IsNil)

	childIP, err := s.d.FindContainerIP("child")
	c.Assert(err, checker.IsNil)
	parentIP, err := s.d.FindContainerIP("parent")
	c.Assert(err, checker.IsNil)

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
	testRequires(c, DaemonIsLinux)

	s.d.StartWithBusybox(c, "--default-ulimit", "nofile=42:42", "--default-ulimit", "nproc=1024:1024")

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
		c.Fatalf("expected `ulimit -p` to be 2048, got: %s", nproc)
	}

	// Now restart daemon with a new default
	s.d.Restart(c, "--default-ulimit", "nofile=43")

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
		c.Fatalf("expected `ulimit -p` to be 2048, got: %s", nproc)
	}
}

// #11315
func (s *DockerDaemonSuite) TestDaemonRestartRenameContainer(c *check.C) {
	s.d.StartWithBusybox(c)

	if out, err := s.d.Cmd("run", "--name=test", "busybox"); err != nil {
		c.Fatal(err, out)
	}

	if out, err := s.d.Cmd("rename", "test", "test2"); err != nil {
		c.Fatal(err, out)
	}

	s.d.Restart(c)

	if out, err := s.d.Cmd("start", "test2"); err != nil {
		c.Fatal(err, out)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverDefault(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "--name=test", "busybox", "echo", "testline")
	c.Assert(err, check.IsNil, check.Commentf(out))
	id, err := s.d.GetIDByName("test")
	c.Assert(err, check.IsNil)

	logPath := filepath.Join(s.d.Root, "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err != nil {
		c.Fatal(err)
	}
	f, err := os.Open(logPath)
	if err != nil {
		c.Fatal(err)
	}
	defer f.Close()

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
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "--name=test", "--log-driver=none", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id, err := s.d.GetIDByName("test")
	c.Assert(err, check.IsNil)

	logPath := filepath.Join(s.d.Root, "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err == nil || !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't exits, error on Stat: %s", logPath, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverNone(c *check.C) {
	s.d.StartWithBusybox(c, "--log-driver=none")

	out, err := s.d.Cmd("run", "--name=test", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id, err := s.d.GetIDByName("test")
	c.Assert(err, check.IsNil)

	logPath := filepath.Join(s.d.Root, "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err == nil || !os.IsNotExist(err) {
		c.Fatalf("%s shouldn't exits, error on Stat: %s", logPath, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverNoneOverride(c *check.C) {
	s.d.StartWithBusybox(c, "--log-driver=none")

	out, err := s.d.Cmd("run", "--name=test", "--log-driver=json-file", "busybox", "echo", "testline")
	if err != nil {
		c.Fatal(out, err)
	}
	id, err := s.d.GetIDByName("test")
	c.Assert(err, check.IsNil)

	logPath := filepath.Join(s.d.Root, "containers", id, id+"-json.log")

	if _, err := os.Stat(logPath); err != nil {
		c.Fatal(err)
	}
	f, err := os.Open(logPath)
	if err != nil {
		c.Fatal(err)
	}
	defer f.Close()

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
	s.d.StartWithBusybox(c, "--log-driver=none")

	out, err := s.d.Cmd("run", "--name=test", "busybox", "echo", "testline")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("logs", "test")
	c.Assert(err, check.NotNil, check.Commentf("Logs should fail with 'none' driver"))
	expected := `configured logging driver does not support reading`
	c.Assert(out, checker.Contains, expected)
}

func (s *DockerDaemonSuite) TestDaemonLoggingDriverShouldBeIgnoredForBuild(c *check.C) {
	s.d.StartWithBusybox(c, "--log-driver=splunk")

	out, err := s.d.Cmd("build")
	out, code, err := s.d.BuildImageWithOut("busyboxs", `
        FROM busybox
        RUN echo foo`, false)
	comment := check.Commentf("Failed to build image. output %s, exitCode %d, err %v", out, code, err)
	c.Assert(err, check.IsNil, comment)
	c.Assert(code, check.Equals, 0, comment)
	c.Assert(out, checker.Contains, "foo", comment)
}

func (s *DockerDaemonSuite) TestDaemonUnixSockCleanedUp(c *check.C) {
	dir, err := ioutil.TempDir("", "socket-cleanup-test")
	if err != nil {
		c.Fatal(err)
	}
	defer os.RemoveAll(dir)

	sockPath := filepath.Join(dir, "docker.sock")
	s.d.Start(c, "--host", "unix://"+sockPath)

	if _, err := os.Stat(sockPath); err != nil {
		c.Fatal("socket does not exist")
	}

	s.d.Stop(c)

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
	s.d.Start(c)
	s.d.Stop(c)

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

	if err := s.d.StartWithError(); err == nil {
		c.Fatalf("It should not be successful to start daemon with wrong key: %v", err)
	}

	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)

	if !strings.Contains(string(content), "Public Key ID does not match") {
		c.Fatalf("Missing KeyID message from daemon logs: %s", string(content))
	}
}

func (s *DockerDaemonSuite) TestDaemonRestartKillWait(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-id", "busybox", "/bin/cat")
	if err != nil {
		c.Fatalf("Could not run /bin/cat: err=%v\n%s", err, out)
	}
	containerID := strings.TrimSpace(out)

	if out, err := s.d.Cmd("kill", containerID); err != nil {
		c.Fatalf("Could not kill %s: err=%v\n%s", containerID, err, out)
	}

	s.d.Restart(c)

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

// TestHTTPSInfo connects via two-way authenticated HTTPS to the info endpoint
func (s *DockerDaemonSuite) TestHTTPSInfo(c *check.C) {
	const (
		testDaemonHTTPSAddr = "tcp://localhost:4271"
	)

	s.d.Start(c,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem",
		"-H", testDaemonHTTPSAddr)

	args := []string{
		"--host", testDaemonHTTPSAddr,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/client-cert.pem",
		"--tlskey", "fixtures/https/client-key.pem",
		"info",
	}
	out, err := s.d.Cmd(args...)
	if err != nil {
		c.Fatalf("Error Occurred: %s and output: %s", err, out)
	}
}

// TestHTTPSRun connects via two-way authenticated HTTPS to the create, attach, start, and wait endpoints.
// https://github.com/docker/docker/issues/19280
func (s *DockerDaemonSuite) TestHTTPSRun(c *check.C) {
	const (
		testDaemonHTTPSAddr = "tcp://localhost:4271"
	)

	s.d.StartWithBusybox(c, "--tlsverify", "--tlscacert", "fixtures/https/ca.pem", "--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem", "-H", testDaemonHTTPSAddr)

	args := []string{
		"--host", testDaemonHTTPSAddr,
		"--tlsverify", "--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/client-cert.pem",
		"--tlskey", "fixtures/https/client-key.pem",
		"run", "busybox", "echo", "TLS response",
	}
	out, err := s.d.Cmd(args...)
	if err != nil {
		c.Fatalf("Error Occurred: %s and output: %s", err, out)
	}

	if !strings.Contains(out, "TLS response") {
		c.Fatalf("expected output to include `TLS response`, got %v", out)
	}
}

// TestTLSVerify verifies that --tlsverify=false turns on tls
func (s *DockerDaemonSuite) TestTLSVerify(c *check.C) {
	out, err := exec.Command(dockerdBinary, "--tlsverify=false").CombinedOutput()
	if err == nil || !strings.Contains(string(out), "Could not load X509 key pair") {
		c.Fatalf("Daemon should not have started due to missing certs: %v\n%s", err, string(out))
	}
}

// TestHTTPSInfoRogueCert connects via two-way authenticated HTTPS to the info endpoint
// by using a rogue client certificate and checks that it fails with the expected error.
func (s *DockerDaemonSuite) TestHTTPSInfoRogueCert(c *check.C) {
	const (
		errBadCertificate   = "bad certificate"
		testDaemonHTTPSAddr = "tcp://localhost:4271"
	)

	s.d.Start(c,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem",
		"-H", testDaemonHTTPSAddr)

	args := []string{
		"--host", testDaemonHTTPSAddr,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/client-rogue-cert.pem",
		"--tlskey", "fixtures/https/client-rogue-key.pem",
		"info",
	}
	out, err := s.d.Cmd(args...)
	if err == nil || !strings.Contains(out, errBadCertificate) {
		c.Fatalf("Expected err: %s, got instead: %s and output: %s", errBadCertificate, err, out)
	}
}

// TestHTTPSInfoRogueServerCert connects via two-way authenticated HTTPS to the info endpoint
// which provides a rogue server certificate and checks that it fails with the expected error
func (s *DockerDaemonSuite) TestHTTPSInfoRogueServerCert(c *check.C) {
	const (
		errCaUnknown             = "x509: certificate signed by unknown authority"
		testDaemonRogueHTTPSAddr = "tcp://localhost:4272"
	)
	s.d.Start(c,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/server-rogue-cert.pem",
		"--tlskey", "fixtures/https/server-rogue-key.pem",
		"-H", testDaemonRogueHTTPSAddr)

	args := []string{
		"--host", testDaemonRogueHTTPSAddr,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/client-rogue-cert.pem",
		"--tlskey", "fixtures/https/client-rogue-key.pem",
		"info",
	}
	out, err := s.d.Cmd(args...)
	if err == nil || !strings.Contains(out, errCaUnknown) {
		c.Fatalf("Expected err: %s, got instead: %s and output: %s", errCaUnknown, err, out)
	}
}

func pingContainers(c *check.C, d *daemon.Daemon, expectFailure bool) {
	var dargs []string
	if d != nil {
		dargs = []string{"--host", d.Sock()}
	}

	args := append(dargs, "run", "-d", "--name", "container1", "busybox", "top")
	dockerCmd(c, args...)

	args = append(dargs, "run", "--rm", "--link", "container1:alias1", "busybox", "sh", "-c")
	pingCmd := "ping -c 1 %s -W 1"
	args = append(args, fmt.Sprintf(pingCmd, "alias1"))
	_, _, err := dockerCmdWithError(args...)

	if expectFailure {
		c.Assert(err, check.NotNil)
	} else {
		c.Assert(err, check.IsNil)
	}

	args = append(dargs, "rm", "-f", "container1")
	dockerCmd(c, args...)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithSocketAsVolume(c *check.C) {
	s.d.StartWithBusybox(c)

	socket := filepath.Join(s.d.Folder, "docker.sock")

	out, err := s.d.Cmd("run", "--restart=always", "-v", socket+":/sock", "busybox")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	s.d.Restart(c)
}

// os.Kill should kill daemon ungracefully, leaving behind container mounts.
// A subsequent daemon restart should clean up said mounts.
func (s *DockerDaemonSuite) TestCleanupMountsAfterDaemonAndContainerKill(c *check.C) {
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{
		Experimental: testEnv.ExperimentalDaemon(),
	})
	d.StartWithBusybox(c)

	out, err := d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	id := strings.TrimSpace(out)
	c.Assert(d.Signal(os.Kill), check.IsNil)
	mountOut, err := ioutil.ReadFile("/proc/self/mountinfo")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))

	// container mounts should exist even after daemon has crashed.
	comment := check.Commentf("%s should stay mounted from older daemon start:\nDaemon root repository %s\n%s", id, d.Root, mountOut)
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, true, comment)

	// kill the container
	icmd.RunCommand(ctrBinary, "--address", "unix:///var/run/docker/libcontainerd/docker-containerd.sock", "containers", "kill", id).Assert(c, icmd.Success)

	// restart daemon.
	d.Restart(c)

	// Now, container mounts should be gone.
	mountOut, err = ioutil.ReadFile("/proc/self/mountinfo")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))
	comment = check.Commentf("%s is still mounted from older daemon start:\nDaemon root repository %s\n%s", id, d.Root, mountOut)
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, false, comment)

	d.Stop(c)
}

// os.Interrupt should perform a graceful daemon shutdown and hence cleanup mounts.
func (s *DockerDaemonSuite) TestCleanupMountsAfterGracefulShutdown(c *check.C) {
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{
		Experimental: testEnv.ExperimentalDaemon(),
	})
	d.StartWithBusybox(c)

	out, err := d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	id := strings.TrimSpace(out)

	// Send SIGINT and daemon should clean up
	c.Assert(d.Signal(os.Interrupt), check.IsNil)
	// Wait for the daemon to stop.
	c.Assert(<-d.Wait, checker.IsNil)

	mountOut, err := ioutil.ReadFile("/proc/self/mountinfo")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))

	comment := check.Commentf("%s is still mounted from older daemon start:\nDaemon root repository %s\n%s", id, d.Root, mountOut)
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, false, comment)
}

func (s *DockerDaemonSuite) TestRunContainerWithBridgeNone(c *check.C) {
	testRequires(c, DaemonIsLinux, NotUserNamespace)
	s.d.StartWithBusybox(c, "-b", "none")

	out, err := s.d.Cmd("run", "--rm", "busybox", "ip", "l")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.Contains(out, "eth0"), check.Equals, false,
		check.Commentf("There shouldn't be eth0 in container in default(bridge) mode when bridge network is disabled: %s", out))

	out, err = s.d.Cmd("run", "--rm", "--net=bridge", "busybox", "ip", "l")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.Contains(out, "eth0"), check.Equals, false,
		check.Commentf("There shouldn't be eth0 in container in bridge mode when bridge network is disabled: %s", out))
	// the extra grep and awk clean up the output of `ip` to only list the number and name of
	// interfaces, allowing for different versions of ip (e.g. inside and outside the container) to
	// be used while still verifying that the interface list is the exact same
	cmd := exec.Command("sh", "-c", "ip l | grep -E '^[0-9]+:' | awk -F: ' { print $1\":\"$2 } '")
	stdout := bytes.NewBuffer(nil)
	cmd.Stdout = stdout
	if err := cmd.Run(); err != nil {
		c.Fatal("Failed to get host network interface")
	}
	out, err = s.d.Cmd("run", "--rm", "--net=host", "busybox", "sh", "-c", "ip l | grep -E '^[0-9]+:' | awk -F: ' { print $1\":\"$2 } '")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(out, check.Equals, fmt.Sprintf("%s", stdout),
		check.Commentf("The network interfaces in container should be the same with host when --net=host when bridge network is disabled: %s", out))
}

func (s *DockerDaemonSuite) TestDaemonRestartWithContainerRunning(t *check.C) {
	s.d.StartWithBusybox(t)
	if out, err := s.d.Cmd("run", "-d", "--name", "test", "busybox", "top"); err != nil {
		t.Fatal(out, err)
	}

	s.d.Restart(t)
	// Container 'test' should be removed without error
	if out, err := s.d.Cmd("rm", "test"); err != nil {
		t.Fatal(out, err)
	}
}

func (s *DockerDaemonSuite) TestDaemonRestartCleanupNetns(c *check.C) {
	s.d.StartWithBusybox(c)
	out, err := s.d.Cmd("run", "--name", "netns", "-d", "busybox", "top")
	if err != nil {
		c.Fatal(out, err)
	}

	// Get sandbox key via inspect
	out, err = s.d.Cmd("inspect", "--format", "'{{.NetworkSettings.SandboxKey}}'", "netns")
	if err != nil {
		c.Fatalf("Error inspecting container: %s, %v", out, err)
	}
	fileName := strings.Trim(out, " \r\n'")

	if out, err := s.d.Cmd("stop", "netns"); err != nil {
		c.Fatal(out, err)
	}

	// Test if the file still exists
	icmd.RunCommand("stat", "-c", "%n", fileName).Assert(c, icmd.Expected{
		Out: fileName,
	})

	// Remove the container and restart the daemon
	if out, err := s.d.Cmd("rm", "netns"); err != nil {
		c.Fatal(out, err)
	}

	s.d.Restart(c)

	// Test again and see now the netns file does not exist
	icmd.RunCommand("stat", "-c", "%n", fileName).Assert(c, icmd.Expected{
		Err:      "No such file or directory",
		ExitCode: 1,
	})
}

// tests regression detailed in #13964 where DOCKER_TLS_VERIFY env is ignored
func (s *DockerDaemonSuite) TestDaemonTLSVerifyIssue13964(c *check.C) {
	host := "tcp://localhost:4271"
	s.d.Start(c, "-H", host)
	icmd.RunCmd(icmd.Cmd{
		Command: []string{dockerBinary, "-H", host, "info"},
		Env:     []string{"DOCKER_TLS_VERIFY=1", "DOCKER_CERT_PATH=fixtures/https"},
	}).Assert(c, icmd.Expected{
		ExitCode: 1,
		Err:      "error during connect",
	})
}

func setupV6(c *check.C) {
	// Hack to get the right IPv6 address on docker0, which has already been created
	result := icmd.RunCommand("ip", "addr", "add", "fe80::1/64", "dev", "docker0")
	result.Assert(c, icmd.Success)
}

func teardownV6(c *check.C) {
	result := icmd.RunCommand("ip", "addr", "del", "fe80::1/64", "dev", "docker0")
	result.Assert(c, icmd.Success)
}

func (s *DockerDaemonSuite) TestDaemonRestartWithContainerWithRestartPolicyAlways(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--restart", "always", "busybox", "top")
	c.Assert(err, check.IsNil)
	id := strings.TrimSpace(out)

	_, err = s.d.Cmd("stop", id)
	c.Assert(err, check.IsNil)
	_, err = s.d.Cmd("wait", id)
	c.Assert(err, check.IsNil)

	out, err = s.d.Cmd("ps", "-q")
	c.Assert(err, check.IsNil)
	c.Assert(out, check.Equals, "")

	s.d.Restart(c)

	out, err = s.d.Cmd("ps", "-q")
	c.Assert(err, check.IsNil)
	c.Assert(strings.TrimSpace(out), check.Equals, id[:12])
}

func (s *DockerDaemonSuite) TestDaemonWideLogConfig(c *check.C) {
	s.d.StartWithBusybox(c, "--log-opt=max-size=1k")
	name := "logtest"
	out, err := s.d.Cmd("run", "-d", "--log-opt=max-file=5", "--name", name, "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s, err: %v", out, err))

	out, err = s.d.Cmd("inspect", "-f", "{{ .HostConfig.LogConfig.Config }}", name)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(out, checker.Contains, "max-size:1k")
	c.Assert(out, checker.Contains, "max-file:5")

	out, err = s.d.Cmd("inspect", "-f", "{{ .HostConfig.LogConfig.Type }}", name)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.TrimSpace(out), checker.Equals, "json-file")
}

func (s *DockerDaemonSuite) TestDaemonRestartWithPausedContainer(c *check.C) {
	s.d.StartWithBusybox(c)
	if out, err := s.d.Cmd("run", "-i", "-d", "--name", "test", "busybox", "top"); err != nil {
		c.Fatal(err, out)
	}
	if out, err := s.d.Cmd("pause", "test"); err != nil {
		c.Fatal(err, out)
	}
	s.d.Restart(c)

	errchan := make(chan error)
	go func() {
		out, err := s.d.Cmd("start", "test")
		if err != nil {
			errchan <- fmt.Errorf("%v:\n%s", err, out)
		}
		name := strings.TrimSpace(out)
		if name != "test" {
			errchan <- fmt.Errorf("Paused container start error on docker daemon restart, expected 'test' but got '%s'", name)
		}
		close(errchan)
	}()

	select {
	case <-time.After(5 * time.Second):
		c.Fatal("Waiting on start a container timed out")
	case err := <-errchan:
		if err != nil {
			c.Fatal(err)
		}
	}
}

func (s *DockerDaemonSuite) TestDaemonRestartRmVolumeInUse(c *check.C) {
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("create", "-v", "test:/foo", "busybox")
	c.Assert(err, check.IsNil, check.Commentf(out))

	s.d.Restart(c)

	out, err = s.d.Cmd("volume", "rm", "test")
	c.Assert(err, check.NotNil, check.Commentf("should not be able to remove in use volume after daemon restart"))
	c.Assert(out, checker.Contains, "in use")
}

func (s *DockerDaemonSuite) TestDaemonRestartLocalVolumes(c *check.C) {
	s.d.Start(c)

	_, err := s.d.Cmd("volume", "create", "test")
	c.Assert(err, check.IsNil)
	s.d.Restart(c)

	_, err = s.d.Cmd("volume", "inspect", "test")
	c.Assert(err, check.IsNil)
}

// FIXME(vdemeester) should be a unit test
func (s *DockerDaemonSuite) TestDaemonCorruptedLogDriverAddress(c *check.C) {
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{
		Experimental: testEnv.ExperimentalDaemon(),
	})
	c.Assert(d.StartWithError("--log-driver=syslog", "--log-opt", "syslog-address=corrupted:42"), check.NotNil)
	expected := "Failed to set log opts: syslog-address should be in form proto://address"
	icmd.RunCommand("grep", expected, d.LogFileName()).Assert(c, icmd.Success)
}

// FIXME(vdemeester) should be a unit test
func (s *DockerDaemonSuite) TestDaemonCorruptedFluentdAddress(c *check.C) {
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{
		Experimental: testEnv.ExperimentalDaemon(),
	})
	c.Assert(d.StartWithError("--log-driver=fluentd", "--log-opt", "fluentd-address=corrupted:c"), check.NotNil)
	expected := "Failed to set log opts: invalid fluentd-address corrupted:c: "
	icmd.RunCommand("grep", expected, d.LogFileName()).Assert(c, icmd.Success)
}

// FIXME(vdemeester) Use a new daemon instance instead of the Suite one
func (s *DockerDaemonSuite) TestDaemonStartWithoutHost(c *check.C) {
	s.d.UseDefaultHost = true
	defer func() {
		s.d.UseDefaultHost = false
	}()
	s.d.Start(c)
}

// FIXME(vdemeester) Use a new daemon instance instead of the Suite one
func (s *DockerDaemonSuite) TestDaemonStartWithDefaultTLSHost(c *check.C) {
	s.d.UseDefaultTLSHost = true
	defer func() {
		s.d.UseDefaultTLSHost = false
	}()
	s.d.Start(c,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/server-cert.pem",
		"--tlskey", "fixtures/https/server-key.pem")

	// The client with --tlsverify should also use default host localhost:2376
	tmpHost := os.Getenv("DOCKER_HOST")
	defer func() {
		os.Setenv("DOCKER_HOST", tmpHost)
	}()

	os.Setenv("DOCKER_HOST", "")

	out, _ := dockerCmd(
		c,
		"--tlsverify",
		"--tlscacert", "fixtures/https/ca.pem",
		"--tlscert", "fixtures/https/client-cert.pem",
		"--tlskey", "fixtures/https/client-key.pem",
		"version",
	)
	if !strings.Contains(out, "Server") {
		c.Fatalf("docker version should return information of server side")
	}

	// ensure when connecting to the server that only a single acceptable CA is requested
	contents, err := ioutil.ReadFile("fixtures/https/ca.pem")
	c.Assert(err, checker.IsNil)
	rootCert, err := helpers.ParseCertificatePEM(contents)
	c.Assert(err, checker.IsNil)
	rootPool := x509.NewCertPool()
	rootPool.AddCert(rootCert)

	var certRequestInfo *tls.CertificateRequestInfo
	conn, err := tls.Dial("tcp", fmt.Sprintf("%s:%d", opts.DefaultHTTPHost, opts.DefaultTLSHTTPPort), &tls.Config{
		RootCAs: rootPool,
		GetClientCertificate: func(cri *tls.CertificateRequestInfo) (*tls.Certificate, error) {
			certRequestInfo = cri
			cert, err := tls.LoadX509KeyPair("fixtures/https/client-cert.pem", "fixtures/https/client-key.pem")
			if err != nil {
				return nil, err
			}
			return &cert, nil
		},
	})
	c.Assert(err, checker.IsNil)
	conn.Close()

	c.Assert(certRequestInfo, checker.NotNil)
	c.Assert(certRequestInfo.AcceptableCAs, checker.HasLen, 1)
	c.Assert(certRequestInfo.AcceptableCAs[0], checker.DeepEquals, rootCert.RawSubject)
}

func (s *DockerDaemonSuite) TestBridgeIPIsExcludedFromAllocatorPool(c *check.C) {
	defaultNetworkBridge := "docker0"
	deleteInterface(c, defaultNetworkBridge)

	bridgeIP := "192.169.1.1"
	bridgeRange := bridgeIP + "/30"

	s.d.StartWithBusybox(c, "--bip", bridgeRange)
	defer s.d.Restart(c)

	var cont int
	for {
		contName := fmt.Sprintf("container%d", cont)
		_, err := s.d.Cmd("run", "--name", contName, "-d", "busybox", "/bin/sleep", "2")
		if err != nil {
			// pool exhausted
			break
		}
		ip, err := s.d.Cmd("inspect", "--format", "'{{.NetworkSettings.IPAddress}}'", contName)
		c.Assert(err, check.IsNil)

		c.Assert(ip, check.Not(check.Equals), bridgeIP)
		cont++
	}
}

// Test daemon for no space left on device error
func (s *DockerDaemonSuite) TestDaemonNoSpaceLeftOnDeviceError(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux, Network)

	testDir, err := ioutil.TempDir("", "no-space-left-on-device-test")
	c.Assert(err, checker.IsNil)
	defer os.RemoveAll(testDir)
	c.Assert(mount.MakeRShared(testDir), checker.IsNil)
	defer mount.Unmount(testDir)

	// create a 2MiB image and mount it as graph root
	// Why in a container? Because `mount` sometimes behaves weirdly and often fails outright on this test in debian:jessie (which is what the test suite runs under if run from the Makefile)
	dockerCmd(c, "run", "--rm", "-v", testDir+":/test", "busybox", "sh", "-c", "dd of=/test/testfs.img bs=1M seek=3 count=0")
	icmd.RunCommand("mkfs.ext4", "-F", filepath.Join(testDir, "testfs.img")).Assert(c, icmd.Success)

	result := icmd.RunCommand("losetup", "-f", "--show", filepath.Join(testDir, "testfs.img"))
	result.Assert(c, icmd.Success)
	loopname := strings.TrimSpace(string(result.Combined()))
	defer exec.Command("losetup", "-d", loopname).Run()

	dockerCmd(c, "run", "--privileged", "--rm", "-v", testDir+":/test:shared", "busybox", "sh", "-c", fmt.Sprintf("mkdir -p /test/test-mount && mount -t ext4 -no loop,rw %v /test/test-mount", loopname))
	defer mount.Unmount(filepath.Join(testDir, "test-mount"))

	s.d.Start(c, "--data-root", filepath.Join(testDir, "test-mount"))
	defer s.d.Stop(c)

	// pull a repository large enough to fill the mount point
	pullOut, err := s.d.Cmd("pull", "registry:2")
	c.Assert(err, checker.NotNil, check.Commentf(pullOut))
	c.Assert(pullOut, checker.Contains, "no space left on device")
}

// Test daemon restart with container links + auto restart
func (s *DockerDaemonSuite) TestDaemonRestartContainerLinksRestart(c *check.C) {
	s.d.StartWithBusybox(c)

	parent1Args := []string{}
	parent2Args := []string{}
	wg := sync.WaitGroup{}
	maxChildren := 10
	chErr := make(chan error, maxChildren)

	for i := 0; i < maxChildren; i++ {
		wg.Add(1)
		name := fmt.Sprintf("test%d", i)

		if i < maxChildren/2 {
			parent1Args = append(parent1Args, []string{"--link", name}...)
		} else {
			parent2Args = append(parent2Args, []string{"--link", name}...)
		}

		go func() {
			_, err := s.d.Cmd("run", "-d", "--name", name, "--restart=always", "busybox", "top")
			chErr <- err
			wg.Done()
		}()
	}

	wg.Wait()
	close(chErr)
	for err := range chErr {
		c.Assert(err, check.IsNil)
	}

	parent1Args = append([]string{"run", "-d"}, parent1Args...)
	parent1Args = append(parent1Args, []string{"--name=parent1", "--restart=always", "busybox", "top"}...)
	parent2Args = append([]string{"run", "-d"}, parent2Args...)
	parent2Args = append(parent2Args, []string{"--name=parent2", "--restart=always", "busybox", "top"}...)

	_, err := s.d.Cmd(parent1Args...)
	c.Assert(err, check.IsNil)
	_, err = s.d.Cmd(parent2Args...)
	c.Assert(err, check.IsNil)

	s.d.Stop(c)
	// clear the log file -- we don't need any of it but may for the next part
	// can ignore the error here, this is just a cleanup
	os.Truncate(s.d.LogFileName(), 0)
	s.d.Start(c)

	for _, num := range []string{"1", "2"} {
		out, err := s.d.Cmd("inspect", "-f", "{{ .State.Running }}", "parent"+num)
		c.Assert(err, check.IsNil)
		if strings.TrimSpace(out) != "true" {
			log, _ := ioutil.ReadFile(s.d.LogFileName())
			c.Fatalf("parent container is not running\n%s", string(log))
		}
	}
}

func (s *DockerDaemonSuite) TestDaemonCgroupParent(c *check.C) {
	testRequires(c, DaemonIsLinux)

	cgroupParent := "test"
	name := "cgroup-test"

	s.d.StartWithBusybox(c, "--cgroup-parent", cgroupParent)
	defer s.d.Restart(c)

	out, err := s.d.Cmd("run", "--name", name, "busybox", "cat", "/proc/self/cgroup")
	c.Assert(err, checker.IsNil)
	cgroupPaths := testutil.ParseCgroupPaths(string(out))
	c.Assert(len(cgroupPaths), checker.Not(checker.Equals), 0, check.Commentf("unexpected output - %q", string(out)))
	out, err = s.d.Cmd("inspect", "-f", "{{.Id}}", name)
	c.Assert(err, checker.IsNil)
	id := strings.TrimSpace(string(out))
	expectedCgroup := path.Join(cgroupParent, id)
	found := false
	for _, path := range cgroupPaths {
		if strings.HasSuffix(path, expectedCgroup) {
			found = true
			break
		}
	}
	c.Assert(found, checker.True, check.Commentf("Cgroup path for container (%s) doesn't found in cgroups file: %s", expectedCgroup, cgroupPaths))
}

func (s *DockerDaemonSuite) TestDaemonRestartWithLinks(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows does not support links
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--name=test", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("run", "--name=test2", "--link", "test:abc", "busybox", "sh", "-c", "ping -c 1 -w 1 abc")
	c.Assert(err, check.IsNil, check.Commentf(out))

	s.d.Restart(c)

	// should fail since test is not running yet
	out, err = s.d.Cmd("start", "test2")
	c.Assert(err, check.NotNil, check.Commentf(out))

	out, err = s.d.Cmd("start", "test")
	c.Assert(err, check.IsNil, check.Commentf(out))
	out, err = s.d.Cmd("start", "-a", "test2")
	c.Assert(err, check.IsNil, check.Commentf(out))
	c.Assert(strings.Contains(out, "1 packets transmitted, 1 packets received"), check.Equals, true, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestDaemonRestartWithNames(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows does not support links
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("create", "--name=test", "busybox")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("run", "-d", "--name=test2", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))
	test2ID := strings.TrimSpace(out)

	out, err = s.d.Cmd("run", "-d", "--name=test3", "--link", "test2:abc", "busybox", "top")
	test3ID := strings.TrimSpace(out)

	s.d.Restart(c)

	out, err = s.d.Cmd("create", "--name=test", "busybox")
	c.Assert(err, check.NotNil, check.Commentf("expected error trying to create container with duplicate name"))
	// this one is no longer needed, removing simplifies the remainder of the test
	out, err = s.d.Cmd("rm", "-f", "test")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("ps", "-a", "--no-trunc")
	c.Assert(err, check.IsNil, check.Commentf(out))

	lines := strings.Split(strings.TrimSpace(out), "\n")[1:]

	test2validated := false
	test3validated := false
	for _, line := range lines {
		fields := strings.Fields(line)
		names := fields[len(fields)-1]
		switch fields[0] {
		case test2ID:
			c.Assert(names, check.Equals, "test2,test3/abc")
			test2validated = true
		case test3ID:
			c.Assert(names, check.Equals, "test3")
			test3validated = true
		}
	}

	c.Assert(test2validated, check.Equals, true)
	c.Assert(test3validated, check.Equals, true)
}

// TestDaemonRestartWithKilledRunningContainer requires live restore of running containers
func (s *DockerDaemonSuite) TestDaemonRestartWithKilledRunningContainer(t *check.C) {
	// TODO(mlaventure): Not sure what would the exit code be on windows
	testRequires(t, DaemonIsLinux)
	s.d.StartWithBusybox(t)

	cid, err := s.d.Cmd("run", "-d", "--name", "test", "busybox", "top")
	defer s.d.Stop(t)
	if err != nil {
		t.Fatal(cid, err)
	}
	cid = strings.TrimSpace(cid)

	pid, err := s.d.Cmd("inspect", "-f", "{{.State.Pid}}", cid)
	t.Assert(err, check.IsNil)
	pid = strings.TrimSpace(pid)

	// Kill the daemon
	if err := s.d.Kill(); err != nil {
		t.Fatal(err)
	}

	// kill the container
	icmd.RunCommand(ctrBinary, "--address", "unix:///var/run/docker/libcontainerd/docker-containerd.sock", "containers", "kill", cid).Assert(t, icmd.Success)

	// Give time to containerd to process the command if we don't
	// the exit event might be received after we do the inspect
	result := icmd.RunCommand("kill", "-0", pid)
	for result.ExitCode == 0 {
		time.Sleep(1 * time.Second)
		// FIXME(vdemeester) should we check it doesn't error out ?
		result = icmd.RunCommand("kill", "-0", pid)
	}

	// restart the daemon
	s.d.Start(t)

	// Check that we've got the correct exit code
	out, err := s.d.Cmd("inspect", "-f", "{{.State.ExitCode}}", cid)
	t.Assert(err, check.IsNil)

	out = strings.TrimSpace(out)
	if out != "143" {
		t.Fatalf("Expected exit code '%s' got '%s' for container '%s'\n", "143", out, cid)
	}

}

// os.Kill should kill daemon ungracefully, leaving behind live containers.
// The live containers should be known to the restarted daemon. Stopping
// them now, should remove the mounts.
func (s *DockerDaemonSuite) TestCleanupMountsAfterDaemonCrash(c *check.C) {
	testRequires(c, DaemonIsLinux)
	s.d.StartWithBusybox(c, "--live-restore")

	out, err := s.d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	id := strings.TrimSpace(out)

	c.Assert(s.d.Signal(os.Kill), check.IsNil)
	mountOut, err := ioutil.ReadFile("/proc/self/mountinfo")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))

	// container mounts should exist even after daemon has crashed.
	comment := check.Commentf("%s should stay mounted from older daemon start:\nDaemon root repository %s\n%s", id, s.d.Root, mountOut)
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, true, comment)

	// restart daemon.
	s.d.Start(c, "--live-restore")

	// container should be running.
	out, err = s.d.Cmd("inspect", "--format={{.State.Running}}", id)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	out = strings.TrimSpace(out)
	if out != "true" {
		c.Fatalf("Container %s expected to stay alive after daemon restart", id)
	}

	// 'docker stop' should work.
	out, err = s.d.Cmd("stop", id)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))

	// Now, container mounts should be gone.
	mountOut, err = ioutil.ReadFile("/proc/self/mountinfo")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", mountOut))
	comment = check.Commentf("%s is still mounted from older daemon start:\nDaemon root repository %s\n%s", id, s.d.Root, mountOut)
	c.Assert(strings.Contains(string(mountOut), id), check.Equals, false, comment)
}

// TestDaemonRestartWithUnpausedRunningContainer requires live restore of running containers.
func (s *DockerDaemonSuite) TestDaemonRestartWithUnpausedRunningContainer(t *check.C) {
	// TODO(mlaventure): Not sure what would the exit code be on windows
	testRequires(t, DaemonIsLinux)
	s.d.StartWithBusybox(t, "--live-restore")

	cid, err := s.d.Cmd("run", "-d", "--name", "test", "busybox", "top")
	defer s.d.Stop(t)
	if err != nil {
		t.Fatal(cid, err)
	}
	cid = strings.TrimSpace(cid)

	pid, err := s.d.Cmd("inspect", "-f", "{{.State.Pid}}", cid)
	t.Assert(err, check.IsNil)

	// pause the container
	if _, err := s.d.Cmd("pause", cid); err != nil {
		t.Fatal(cid, err)
	}

	// Kill the daemon
	if err := s.d.Kill(); err != nil {
		t.Fatal(err)
	}

	// resume the container
	result := icmd.RunCommand(
		ctrBinary,
		"--address", "unix:///var/run/docker/libcontainerd/docker-containerd.sock",
		"containers", "resume", cid)
	t.Assert(result, icmd.Matches, icmd.Success)

	// Give time to containerd to process the command if we don't
	// the resume event might be received after we do the inspect
	waitAndAssert(t, defaultReconciliationTimeout, func(*check.C) (interface{}, check.CommentInterface) {
		result := icmd.RunCommand("kill", "-0", strings.TrimSpace(pid))
		return result.ExitCode, nil
	}, checker.Equals, 0)

	// restart the daemon
	s.d.Start(t, "--live-restore")

	// Check that we've got the correct status
	out, err := s.d.Cmd("inspect", "-f", "{{.State.Status}}", cid)
	t.Assert(err, check.IsNil)

	out = strings.TrimSpace(out)
	if out != "running" {
		t.Fatalf("Expected exit code '%s' got '%s' for container '%s'\n", "running", out, cid)
	}
	if _, err := s.d.Cmd("kill", cid); err != nil {
		t.Fatal(err)
	}
}

// TestRunLinksChanged checks that creating a new container with the same name does not update links
// this ensures that the old, pre gh#16032 functionality continues on
func (s *DockerDaemonSuite) TestRunLinksChanged(c *check.C) {
	testRequires(c, DaemonIsLinux) // Windows does not support links
	s.d.StartWithBusybox(c)

	out, err := s.d.Cmd("run", "-d", "--name=test", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("run", "--name=test2", "--link=test:abc", "busybox", "sh", "-c", "ping -c 1 abc")
	c.Assert(err, check.IsNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "1 packets transmitted, 1 packets received")

	out, err = s.d.Cmd("rm", "-f", "test")
	c.Assert(err, check.IsNil, check.Commentf(out))

	out, err = s.d.Cmd("run", "-d", "--name=test", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))
	out, err = s.d.Cmd("start", "-a", "test2")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, check.Not(checker.Contains), "1 packets transmitted, 1 packets received")

	s.d.Restart(c)
	out, err = s.d.Cmd("start", "-a", "test2")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, check.Not(checker.Contains), "1 packets transmitted, 1 packets received")
}

func (s *DockerDaemonSuite) TestDaemonStartWithoutColors(c *check.C) {
	testRequires(c, DaemonIsLinux, NotPpc64le)

	infoLog := "\x1b[34mINFO\x1b"

	b := bytes.NewBuffer(nil)
	done := make(chan bool)

	p, tty, err := pty.Open()
	c.Assert(err, checker.IsNil)
	defer func() {
		tty.Close()
		p.Close()
	}()

	go func() {
		io.Copy(b, p)
		done <- true
	}()

	// Enable coloring explicitly
	s.d.StartWithLogFile(tty, "--raw-logs=false")
	s.d.Stop(c)
	// Wait for io.Copy() before checking output
	<-done
	c.Assert(b.String(), checker.Contains, infoLog)

	b.Reset()

	// "tty" is already closed in prev s.d.Stop(),
	// we have to close the other side "p" and open another pair of
	// pty for the next test.
	p.Close()
	p, tty, err = pty.Open()
	c.Assert(err, checker.IsNil)

	go func() {
		io.Copy(b, p)
		done <- true
	}()

	// Disable coloring explicitly
	s.d.StartWithLogFile(tty, "--raw-logs=true")
	s.d.Stop(c)
	// Wait for io.Copy() before checking output
	<-done
	c.Assert(b.String(), check.Not(check.Equals), "")
	c.Assert(b.String(), check.Not(checker.Contains), infoLog)
}

func (s *DockerDaemonSuite) TestDaemonDebugLog(c *check.C) {
	testRequires(c, DaemonIsLinux, NotPpc64le)

	debugLog := "\x1b[37mDEBU\x1b"

	p, tty, err := pty.Open()
	c.Assert(err, checker.IsNil)
	defer func() {
		tty.Close()
		p.Close()
	}()

	b := bytes.NewBuffer(nil)
	go io.Copy(b, p)

	s.d.StartWithLogFile(tty, "--debug")
	s.d.Stop(c)
	c.Assert(b.String(), checker.Contains, debugLog)
}

func (s *DockerDaemonSuite) TestDaemonDiscoveryBackendConfigReload(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	// daemon config file
	daemonConfig := `{ "debug" : false }`
	configFile, err := ioutil.TempFile("", "test-daemon-discovery-backend-config-reload-config")
	c.Assert(err, checker.IsNil, check.Commentf("could not create temp file for config reload"))
	configFilePath := configFile.Name()
	defer func() {
		configFile.Close()
		os.RemoveAll(configFile.Name())
	}()

	_, err = configFile.Write([]byte(daemonConfig))
	c.Assert(err, checker.IsNil)

	// --log-level needs to be set so that d.Start() doesn't add --debug causing
	// a conflict with the config
	s.d.Start(c, "--config-file", configFilePath, "--log-level=info")

	// daemon config file
	daemonConfig = `{
	      "cluster-store": "consul://consuladdr:consulport/some/path",
	      "cluster-advertise": "192.168.56.100:0",
	      "debug" : false
	}`

	err = configFile.Truncate(0)
	c.Assert(err, checker.IsNil)
	_, err = configFile.Seek(0, os.SEEK_SET)
	c.Assert(err, checker.IsNil)

	_, err = configFile.Write([]byte(daemonConfig))
	c.Assert(err, checker.IsNil)

	err = s.d.ReloadConfig()
	c.Assert(err, checker.IsNil, check.Commentf("error reloading daemon config"))

	out, err := s.d.Cmd("info")
	c.Assert(err, checker.IsNil)

	c.Assert(out, checker.Contains, fmt.Sprintf("Cluster Store: consul://consuladdr:consulport/some/path"))
	c.Assert(out, checker.Contains, fmt.Sprintf("Cluster Advertise: 192.168.56.100:0"))
}

// Test for #21956
func (s *DockerDaemonSuite) TestDaemonLogOptions(c *check.C) {
	s.d.StartWithBusybox(c, "--log-driver=syslog", "--log-opt=syslog-address=udp://127.0.0.1:514")

	out, err := s.d.Cmd("run", "-d", "--log-driver=json-file", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	out, err = s.d.Cmd("inspect", "--format='{{.HostConfig.LogConfig}}'", id)
	c.Assert(err, check.IsNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "{json-file map[]}")
}

// Test case for #20936, #22443
func (s *DockerDaemonSuite) TestDaemonMaxConcurrency(c *check.C) {
	s.d.Start(c, "--max-concurrent-uploads=6", "--max-concurrent-downloads=8")

	expectedMaxConcurrentUploads := `level=debug msg="Max Concurrent Uploads: 6"`
	expectedMaxConcurrentDownloads := `level=debug msg="Max Concurrent Downloads: 8"`
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)
}

// Test case for #20936, #22443
func (s *DockerDaemonSuite) TestDaemonMaxConcurrencyWithConfigFile(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	// daemon config file
	configFilePath := "test.json"
	configFile, err := os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	defer os.Remove(configFilePath)

	daemonConfig := `{ "max-concurrent-downloads" : 8 }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()
	s.d.Start(c, fmt.Sprintf("--config-file=%s", configFilePath))

	expectedMaxConcurrentUploads := `level=debug msg="Max Concurrent Uploads: 5"`
	expectedMaxConcurrentDownloads := `level=debug msg="Max Concurrent Downloads: 8"`
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)

	configFile, err = os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	daemonConfig = `{ "max-concurrent-uploads" : 7, "max-concurrent-downloads" : 9 }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()

	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)
	// unix.Kill(s.d.cmd.Process.Pid, unix.SIGHUP)

	time.Sleep(3 * time.Second)

	expectedMaxConcurrentUploads = `level=debug msg="Reset Max Concurrent Uploads: 7"`
	expectedMaxConcurrentDownloads = `level=debug msg="Reset Max Concurrent Downloads: 9"`
	content, err = s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)
}

// Test case for #20936, #22443
func (s *DockerDaemonSuite) TestDaemonMaxConcurrencyWithConfigFileReload(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	// daemon config file
	configFilePath := "test.json"
	configFile, err := os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	defer os.Remove(configFilePath)

	daemonConfig := `{ "max-concurrent-uploads" : null }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()
	s.d.Start(c, fmt.Sprintf("--config-file=%s", configFilePath))

	expectedMaxConcurrentUploads := `level=debug msg="Max Concurrent Uploads: 5"`
	expectedMaxConcurrentDownloads := `level=debug msg="Max Concurrent Downloads: 3"`
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)

	configFile, err = os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	daemonConfig = `{ "max-concurrent-uploads" : 1, "max-concurrent-downloads" : null }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()

	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)
	// unix.Kill(s.d.cmd.Process.Pid, unix.SIGHUP)

	time.Sleep(3 * time.Second)

	expectedMaxConcurrentUploads = `level=debug msg="Reset Max Concurrent Uploads: 1"`
	expectedMaxConcurrentDownloads = `level=debug msg="Reset Max Concurrent Downloads: 3"`
	content, err = s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)

	configFile, err = os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	daemonConfig = `{ "labels":["foo=bar"] }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()

	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)

	time.Sleep(3 * time.Second)

	expectedMaxConcurrentUploads = `level=debug msg="Reset Max Concurrent Uploads: 5"`
	expectedMaxConcurrentDownloads = `level=debug msg="Reset Max Concurrent Downloads: 3"`
	content, err = s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentUploads)
	c.Assert(string(content), checker.Contains, expectedMaxConcurrentDownloads)
}

func (s *DockerDaemonSuite) TestBuildOnDisabledBridgeNetworkDaemon(c *check.C) {
	s.d.StartWithBusybox(c, "-b=none", "--iptables=false")
	out, code, err := s.d.BuildImageWithOut("busyboxs",
		`FROM busybox
                RUN cat /etc/hosts`, false)
	comment := check.Commentf("Failed to build image. output %s, exitCode %d, err %v", out, code, err)
	c.Assert(err, check.IsNil, comment)
	c.Assert(code, check.Equals, 0, comment)
}

// Test case for #21976
func (s *DockerDaemonSuite) TestDaemonDNSFlagsInHostMode(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	s.d.StartWithBusybox(c, "--dns", "1.2.3.4", "--dns-search", "example.com", "--dns-opt", "timeout:3")

	expectedOutput := "nameserver 1.2.3.4"
	out, _ := s.d.Cmd("run", "--net=host", "busybox", "cat", "/etc/resolv.conf")
	c.Assert(out, checker.Contains, expectedOutput, check.Commentf("Expected '%s', but got %q", expectedOutput, out))
	expectedOutput = "search example.com"
	c.Assert(out, checker.Contains, expectedOutput, check.Commentf("Expected '%s', but got %q", expectedOutput, out))
	expectedOutput = "options timeout:3"
	c.Assert(out, checker.Contains, expectedOutput, check.Commentf("Expected '%s', but got %q", expectedOutput, out))
}

func (s *DockerDaemonSuite) TestRunWithRuntimeFromConfigFile(c *check.C) {
	conf, err := ioutil.TempFile("", "config-file-")
	c.Assert(err, check.IsNil)
	configName := conf.Name()
	conf.Close()
	defer os.Remove(configName)

	config := `
{
    "runtimes": {
        "oci": {
            "path": "docker-runc"
        },
        "vm": {
            "path": "/usr/local/bin/vm-manager",
            "runtimeArgs": [
                "--debug"
            ]
        }
    }
}
`
	ioutil.WriteFile(configName, []byte(config), 0644)
	s.d.StartWithBusybox(c, "--config-file", configName)

	// Run with default runtime
	out, err := s.d.Cmd("run", "--rm", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with default runtime explicitly
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with oci (same path as default) but keep it around
	out, err = s.d.Cmd("run", "--name", "oci-runtime-ls", "--runtime=oci", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with "vm"
	out, err = s.d.Cmd("run", "--rm", "--runtime=vm", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "/usr/local/bin/vm-manager: no such file or directory")

	// Reset config to only have the default
	config = `
{
    "runtimes": {
    }
}
`
	ioutil.WriteFile(configName, []byte(config), 0644)
	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)
	// Give daemon time to reload config
	<-time.After(1 * time.Second)

	// Run with default runtime
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with "oci"
	out, err = s.d.Cmd("run", "--rm", "--runtime=oci", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Unknown runtime specified oci")

	// Start previously created container with oci
	out, err = s.d.Cmd("start", "oci-runtime-ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Unknown runtime specified oci")

	// Check that we can't override the default runtime
	config = `
{
    "runtimes": {
        "runc": {
            "path": "my-runc"
        }
    }
}
`
	ioutil.WriteFile(configName, []byte(config), 0644)
	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)
	// Give daemon time to reload config
	<-time.After(1 * time.Second)

	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, `file configuration validation failed (runtime name 'runc' is reserved)`)

	// Check that we can select a default runtime
	config = `
{
    "default-runtime": "vm",
    "runtimes": {
        "oci": {
            "path": "docker-runc"
        },
        "vm": {
            "path": "/usr/local/bin/vm-manager",
            "runtimeArgs": [
                "--debug"
            ]
        }
    }
}
`
	ioutil.WriteFile(configName, []byte(config), 0644)
	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)
	// Give daemon time to reload config
	<-time.After(1 * time.Second)

	out, err = s.d.Cmd("run", "--rm", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "/usr/local/bin/vm-manager: no such file or directory")

	// Run with default runtime explicitly
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestRunWithRuntimeFromCommandLine(c *check.C) {
	s.d.StartWithBusybox(c, "--add-runtime", "oci=docker-runc", "--add-runtime", "vm=/usr/local/bin/vm-manager")

	// Run with default runtime
	out, err := s.d.Cmd("run", "--rm", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with default runtime explicitly
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with oci (same path as default) but keep it around
	out, err = s.d.Cmd("run", "--name", "oci-runtime-ls", "--runtime=oci", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with "vm"
	out, err = s.d.Cmd("run", "--rm", "--runtime=vm", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "/usr/local/bin/vm-manager: no such file or directory")

	// Start a daemon without any extra runtimes
	s.d.Stop(c)
	s.d.StartWithBusybox(c)

	// Run with default runtime
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))

	// Run with "oci"
	out, err = s.d.Cmd("run", "--rm", "--runtime=oci", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Unknown runtime specified oci")

	// Start previously created container with oci
	out, err = s.d.Cmd("start", "oci-runtime-ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Unknown runtime specified oci")

	// Check that we can't override the default runtime
	s.d.Stop(c)
	c.Assert(s.d.StartWithError("--add-runtime", "runc=my-runc"), checker.NotNil)

	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, `runtime name 'runc' is reserved`)

	// Check that we can select a default runtime
	s.d.Stop(c)
	s.d.StartWithBusybox(c, "--default-runtime=vm", "--add-runtime", "oci=docker-runc", "--add-runtime", "vm=/usr/local/bin/vm-manager")

	out, err = s.d.Cmd("run", "--rm", "busybox", "ls")
	c.Assert(err, check.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "/usr/local/bin/vm-manager: no such file or directory")

	// Run with default runtime explicitly
	out, err = s.d.Cmd("run", "--rm", "--runtime=runc", "busybox", "ls")
	c.Assert(err, check.IsNil, check.Commentf(out))
}

func (s *DockerDaemonSuite) TestDaemonRestartWithAutoRemoveContainer(c *check.C) {
	s.d.StartWithBusybox(c)

	// top1 will exist after daemon restarts
	out, err := s.d.Cmd("run", "-d", "--name", "top1", "busybox:latest", "top")
	c.Assert(err, checker.IsNil, check.Commentf("run top1: %v", out))
	// top2 will be removed after daemon restarts
	out, err = s.d.Cmd("run", "-d", "--rm", "--name", "top2", "busybox:latest", "top")
	c.Assert(err, checker.IsNil, check.Commentf("run top2: %v", out))

	out, err = s.d.Cmd("ps")
	c.Assert(out, checker.Contains, "top1", check.Commentf("top1 should be running"))
	c.Assert(out, checker.Contains, "top2", check.Commentf("top2 should be running"))

	// now restart daemon gracefully
	s.d.Restart(c)

	out, err = s.d.Cmd("ps", "-a")
	c.Assert(err, checker.IsNil, check.Commentf("out: %v", out))
	c.Assert(out, checker.Contains, "top1", check.Commentf("top1 should exist after daemon restarts"))
	c.Assert(out, checker.Not(checker.Contains), "top2", check.Commentf("top2 should be removed after daemon restarts"))
}

func (s *DockerDaemonSuite) TestDaemonRestartSaveContainerExitCode(c *check.C) {
	s.d.StartWithBusybox(c)

	containerName := "error-values"
	// Make a container with both a non 0 exit code and an error message
	// We explicitly disable `--init` for this test, because `--init` is enabled by default
	// on "experimental". Enabling `--init` results in a different behavior; because the "init"
	// process itself is PID1, the container does not fail on _startup_ (i.e., `docker-init` starting),
	// but directly after. The exit code of the container is still 127, but the Error Message is not
	// captured, so `.State.Error` is empty.
	// See the discussion on https://github.com/docker/docker/pull/30227#issuecomment-274161426,
	// and https://github.com/docker/docker/pull/26061#r78054578 for more information.
	out, err := s.d.Cmd("run", "--name", containerName, "--init=false", "busybox", "toto")
	c.Assert(err, checker.NotNil)

	// Check that those values were saved on disk
	out, err = s.d.Cmd("inspect", "-f", "{{.State.ExitCode}}", containerName)
	out = strings.TrimSpace(out)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Equals, "127")

	errMsg1, err := s.d.Cmd("inspect", "-f", "{{.State.Error}}", containerName)
	errMsg1 = strings.TrimSpace(errMsg1)
	c.Assert(err, checker.IsNil)
	c.Assert(errMsg1, checker.Contains, "executable file not found")

	// now restart daemon
	s.d.Restart(c)

	// Check that those values are still around
	out, err = s.d.Cmd("inspect", "-f", "{{.State.ExitCode}}", containerName)
	out = strings.TrimSpace(out)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Equals, "127")

	out, err = s.d.Cmd("inspect", "-f", "{{.State.Error}}", containerName)
	out = strings.TrimSpace(out)
	c.Assert(err, checker.IsNil)
	c.Assert(out, checker.Equals, errMsg1)
}

func (s *DockerDaemonSuite) TestDaemonBackcompatPre17Volumes(c *check.C) {
	testRequires(c, SameHostDaemon)
	d := s.d
	d.StartWithBusybox(c)

	// hack to be able to side-load a container config
	out, err := d.Cmd("create", "busybox:latest")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	id := strings.TrimSpace(out)

	out, err = d.Cmd("inspect", "--type=image", "--format={{.ID}}", "busybox:latest")
	c.Assert(err, checker.IsNil, check.Commentf(out))
	d.Stop(c)
	<-d.Wait

	imageID := strings.TrimSpace(out)
	volumeID := stringid.GenerateNonCryptoID()
	vfsPath := filepath.Join(d.Root, "vfs", "dir", volumeID)
	c.Assert(os.MkdirAll(vfsPath, 0755), checker.IsNil)

	config := []byte(`
		{
			"ID": "` + id + `",
			"Name": "hello",
			"Driver": "` + d.StorageDriver() + `",
			"Image": "` + imageID + `",
			"Config": {"Image": "busybox:latest"},
			"NetworkSettings": {},
			"Volumes": {
				"/bar":"/foo",
				"/foo": "` + vfsPath + `",
				"/quux":"/quux"
			},
			"VolumesRW": {
				"/bar": true,
				"/foo": true,
				"/quux": false
			}
		}
	`)

	configPath := filepath.Join(d.Root, "containers", id, "config.v2.json")
	c.Assert(ioutil.WriteFile(configPath, config, 600), checker.IsNil)
	d.Start(c)

	out, err = d.Cmd("inspect", "--type=container", "--format={{ json .Mounts }}", id)
	c.Assert(err, checker.IsNil, check.Commentf(out))
	type mount struct {
		Name        string
		Source      string
		Destination string
		Driver      string
		RW          bool
	}

	ls := []mount{}
	err = json.NewDecoder(strings.NewReader(out)).Decode(&ls)
	c.Assert(err, checker.IsNil)

	expected := []mount{
		{Source: "/foo", Destination: "/bar", RW: true},
		{Name: volumeID, Destination: "/foo", RW: true},
		{Source: "/quux", Destination: "/quux", RW: false},
	}
	c.Assert(ls, checker.HasLen, len(expected))

	for _, m := range ls {
		var matched bool
		for _, x := range expected {
			if m.Source == x.Source && m.Destination == x.Destination && m.RW == x.RW || m.Name != x.Name {
				matched = true
				break
			}
		}
		c.Assert(matched, checker.True, check.Commentf("did find match for %+v", m))
	}
}

func (s *DockerDaemonSuite) TestDaemonWithUserlandProxyPath(c *check.C) {
	testRequires(c, SameHostDaemon, DaemonIsLinux)

	dockerProxyPath, err := exec.LookPath("docker-proxy")
	c.Assert(err, checker.IsNil)
	tmpDir, err := ioutil.TempDir("", "test-docker-proxy")
	c.Assert(err, checker.IsNil)

	newProxyPath := filepath.Join(tmpDir, "docker-proxy")
	cmd := exec.Command("cp", dockerProxyPath, newProxyPath)
	c.Assert(cmd.Run(), checker.IsNil)

	// custom one
	s.d.StartWithBusybox(c, "--userland-proxy-path", newProxyPath)
	out, err := s.d.Cmd("run", "-p", "5000:5000", "busybox:latest", "true")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	// try with the original one
	s.d.Restart(c, "--userland-proxy-path", dockerProxyPath)
	out, err = s.d.Cmd("run", "-p", "5000:5000", "busybox:latest", "true")
	c.Assert(err, checker.IsNil, check.Commentf(out))

	// not exist
	s.d.Restart(c, "--userland-proxy-path", "/does/not/exist")
	out, err = s.d.Cmd("run", "-p", "5000:5000", "busybox:latest", "true")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "driver failed programming external connectivity on endpoint")
	c.Assert(out, checker.Contains, "/does/not/exist: no such file or directory")
}

// Test case for #22471
func (s *DockerDaemonSuite) TestDaemonShutdownTimeout(c *check.C) {
	testRequires(c, SameHostDaemon)
	s.d.StartWithBusybox(c, "--shutdown-timeout=3")

	_, err := s.d.Cmd("run", "-d", "busybox", "top")
	c.Assert(err, check.IsNil)

	c.Assert(s.d.Signal(unix.SIGINT), checker.IsNil)

	select {
	case <-s.d.Wait:
	case <-time.After(5 * time.Second):
	}

	expectedMessage := `level=debug msg="start clean shutdown of all containers with a 3 seconds timeout..."`
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMessage)
}

// Test case for #22471
func (s *DockerDaemonSuite) TestDaemonShutdownTimeoutWithConfigFile(c *check.C) {
	testRequires(c, SameHostDaemon)

	// daemon config file
	configFilePath := "test.json"
	configFile, err := os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	defer os.Remove(configFilePath)

	daemonConfig := `{ "shutdown-timeout" : 8 }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()
	s.d.Start(c, fmt.Sprintf("--config-file=%s", configFilePath))

	configFile, err = os.Create(configFilePath)
	c.Assert(err, checker.IsNil)
	daemonConfig = `{ "shutdown-timeout" : 5 }`
	fmt.Fprintf(configFile, "%s", daemonConfig)
	configFile.Close()

	c.Assert(s.d.Signal(unix.SIGHUP), checker.IsNil)

	select {
	case <-s.d.Wait:
	case <-time.After(3 * time.Second):
	}

	expectedMessage := `level=debug msg="Reset Shutdown Timeout: 5"`
	content, err := s.d.ReadLogFile()
	c.Assert(err, checker.IsNil)
	c.Assert(string(content), checker.Contains, expectedMessage)
}

// Test case for 29342
func (s *DockerDaemonSuite) TestExecWithUserAfterLiveRestore(c *check.C) {
	testRequires(c, DaemonIsLinux)
	s.d.StartWithBusybox(c, "--live-restore")

	out, err := s.d.Cmd("run", "-d", "--name=top", "busybox", "sh", "-c", "addgroup -S test && adduser -S -G test test -D -s /bin/sh && touch /adduser_end && top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))

	s.d.WaitRun("top")

	// Wait for shell command to be completed
	_, err = s.d.Cmd("exec", "top", "sh", "-c", `for i in $(seq 1 5); do if [ -e /adduser_end ]; then rm -f /adduser_end && break; else sleep 1 && false; fi; done`)
	c.Assert(err, check.IsNil, check.Commentf("Timeout waiting for shell command to be completed"))

	out1, err := s.d.Cmd("exec", "-u", "test", "top", "id")
	// uid=100(test) gid=101(test) groups=101(test)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out1))

	// restart daemon.
	s.d.Restart(c, "--live-restore")

	out2, err := s.d.Cmd("exec", "-u", "test", "top", "id")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out2))
	c.Assert(out2, check.Equals, out1, check.Commentf("Output: before restart '%s', after restart '%s'", out1, out2))

	out, err = s.d.Cmd("stop", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
}

func (s *DockerDaemonSuite) TestRemoveContainerAfterLiveRestore(c *check.C) {
	testRequires(c, DaemonIsLinux, overlayFSSupported, SameHostDaemon)
	s.d.StartWithBusybox(c, "--live-restore", "--storage-driver", "overlay")
	out, err := s.d.Cmd("run", "-d", "--name=top", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))

	s.d.WaitRun("top")

	// restart daemon.
	s.d.Restart(c, "--live-restore", "--storage-driver", "overlay")

	out, err = s.d.Cmd("stop", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))

	// test if the rootfs mountpoint still exist
	mountpoint, err := s.d.InspectField("top", ".GraphDriver.Data.MergedDir")
	c.Assert(err, check.IsNil)
	f, err := os.Open("/proc/self/mountinfo")
	c.Assert(err, check.IsNil)
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(line, mountpoint) {
			c.Fatalf("mountinfo should not include the mountpoint of stop container")
		}
	}

	out, err = s.d.Cmd("rm", "top")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
}

// #29598
func (s *DockerDaemonSuite) TestRestartPolicyWithLiveRestore(c *check.C) {
	testRequires(c, DaemonIsLinux, SameHostDaemon)
	s.d.StartWithBusybox(c, "--live-restore")

	out, err := s.d.Cmd("run", "-d", "--restart", "always", "busybox", "top")
	c.Assert(err, check.IsNil, check.Commentf("output: %s", out))
	id := strings.TrimSpace(out)

	type state struct {
		Running   bool
		StartedAt time.Time
	}
	out, err = s.d.Cmd("inspect", "-f", "{{json .State}}", id)
	c.Assert(err, checker.IsNil, check.Commentf("output: %s", out))

	var origState state
	err = json.Unmarshal([]byte(strings.TrimSpace(out)), &origState)
	c.Assert(err, checker.IsNil)

	s.d.Restart(c, "--live-restore")

	pid, err := s.d.Cmd("inspect", "-f", "{{.State.Pid}}", id)
	c.Assert(err, check.IsNil)
	pidint, err := strconv.Atoi(strings.TrimSpace(pid))
	c.Assert(err, check.IsNil)
	c.Assert(pidint, checker.GreaterThan, 0)
	c.Assert(unix.Kill(pidint, unix.SIGKILL), check.IsNil)

	ticker := time.NewTicker(50 * time.Millisecond)
	timeout := time.After(10 * time.Second)

	for range ticker.C {
		select {
		case <-timeout:
			c.Fatal("timeout waiting for container restart")
		default:
		}

		out, err := s.d.Cmd("inspect", "-f", "{{json .State}}", id)
		c.Assert(err, checker.IsNil, check.Commentf("output: %s", out))

		var newState state
		err = json.Unmarshal([]byte(strings.TrimSpace(out)), &newState)
		c.Assert(err, checker.IsNil)

		if !newState.Running {
			continue
		}
		if newState.StartedAt.After(origState.StartedAt) {
			break
		}
	}

	out, err = s.d.Cmd("stop", id)
	c.Assert(err, check.IsNil, check.Commentf("output: %s", out))
}

func (s *DockerDaemonSuite) TestShmSize(c *check.C) {
	testRequires(c, DaemonIsLinux)

	size := 67108864 * 2
	pattern := regexp.MustCompile(fmt.Sprintf("shm on /dev/shm type tmpfs(.*)size=%dk", size/1024))

	s.d.StartWithBusybox(c, "--default-shm-size", fmt.Sprintf("%v", size))

	name := "shm1"
	out, err := s.d.Cmd("run", "--name", name, "busybox", "mount")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(pattern.MatchString(out), checker.True)
	out, err = s.d.Cmd("inspect", "--format", "{{.HostConfig.ShmSize}}", name)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.TrimSpace(out), check.Equals, fmt.Sprintf("%v", size))
}

func (s *DockerDaemonSuite) TestShmSizeReload(c *check.C) {
	testRequires(c, DaemonIsLinux)

	configPath, err := ioutil.TempDir("", "test-daemon-shm-size-reload-config")
	c.Assert(err, checker.IsNil, check.Commentf("could not create temp file for config reload"))
	defer os.RemoveAll(configPath) // clean up
	configFile := filepath.Join(configPath, "config.json")

	size := 67108864 * 2
	configData := []byte(fmt.Sprintf(`{"default-shm-size": "%dM"}`, size/1024/1024))
	c.Assert(ioutil.WriteFile(configFile, configData, 0666), checker.IsNil, check.Commentf("could not write temp file for config reload"))
	pattern := regexp.MustCompile(fmt.Sprintf("shm on /dev/shm type tmpfs(.*)size=%dk", size/1024))

	s.d.StartWithBusybox(c, "--config-file", configFile)

	name := "shm1"
	out, err := s.d.Cmd("run", "--name", name, "busybox", "mount")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(pattern.MatchString(out), checker.True)
	out, err = s.d.Cmd("inspect", "--format", "{{.HostConfig.ShmSize}}", name)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.TrimSpace(out), check.Equals, fmt.Sprintf("%v", size))

	size = 67108864 * 3
	configData = []byte(fmt.Sprintf(`{"default-shm-size": "%dM"}`, size/1024/1024))
	c.Assert(ioutil.WriteFile(configFile, configData, 0666), checker.IsNil, check.Commentf("could not write temp file for config reload"))
	pattern = regexp.MustCompile(fmt.Sprintf("shm on /dev/shm type tmpfs(.*)size=%dk", size/1024))

	err = s.d.ReloadConfig()
	c.Assert(err, checker.IsNil, check.Commentf("error reloading daemon config"))

	name = "shm2"
	out, err = s.d.Cmd("run", "--name", name, "busybox", "mount")
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(pattern.MatchString(out), checker.True)
	out, err = s.d.Cmd("inspect", "--format", "{{.HostConfig.ShmSize}}", name)
	c.Assert(err, check.IsNil, check.Commentf("Output: %s", out))
	c.Assert(strings.TrimSpace(out), check.Equals, fmt.Sprintf("%v", size))
}

// TestFailedPluginRemove makes sure that a failed plugin remove does not block
// the daemon from starting
func (s *DockerDaemonSuite) TestFailedPluginRemove(c *check.C) {
	testRequires(c, DaemonIsLinux, IsAmd64, SameHostDaemon)
	d := daemon.New(c, dockerBinary, dockerdBinary, daemon.Config{})
	d.Start(c)
	cli, err := client.NewClient(d.Sock(), api.DefaultVersion, nil, nil)
	c.Assert(err, checker.IsNil)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	name := "test-plugin-rm-fail"
	out, err := cli.PluginInstall(ctx, name, types.PluginInstallOptions{
		Disabled:             true,
		AcceptAllPermissions: true,
		RemoteRef:            "cpuguy83/docker-logdriver-test",
	})
	c.Assert(err, checker.IsNil)
	defer out.Close()
	io.Copy(ioutil.Discard, out)

	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	p, _, err := cli.PluginInspectWithRaw(ctx, name)
	c.Assert(err, checker.IsNil)

	// simulate a bad/partial removal by removing the plugin config.
	configPath := filepath.Join(d.Root, "plugins", p.ID, "config.json")
	c.Assert(os.Remove(configPath), checker.IsNil)

	d.Restart(c)
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = cli.Ping(ctx)
	c.Assert(err, checker.IsNil)

	_, _, err = cli.PluginInspectWithRaw(ctx, name)
	// plugin should be gone since the config.json is gone
	c.Assert(err, checker.NotNil)
}
