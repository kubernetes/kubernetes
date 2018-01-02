// +build !windows

package main

import (
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/docker/docker/pkg/parsers/kernel"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
)

// ensure Kernel version is >= v3.9 for macvlan support
func macvlanKernelSupport() bool {
	return checkKernelMajorVersionGreaterOrEqualThen(3, 9)
}

// ensure Kernel version is >= v4.2 for ipvlan support
func ipvlanKernelSupport() bool {
	return checkKernelMajorVersionGreaterOrEqualThen(4, 2)
}

func checkKernelMajorVersionGreaterOrEqualThen(kernelVersion int, majorVersion int) bool {
	kv, err := kernel.GetKernelVersion()
	if err != nil {
		return false
	}
	if kv.Kernel < kernelVersion || (kv.Kernel == kernelVersion && kv.Major < majorVersion) {
		return false
	}
	return true
}

func (s *DockerNetworkSuite) TestDockerNetworkMacvlanPersistance(c *check.C) {
	// verify the driver automatically provisions the 802.1q link (dm-dummy0.60)
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)

	// master dummy interface 'dm' abbreviation represents 'docker macvlan'
	master := "dm-dummy0"
	// simulate the master link the vlan tagged subinterface parent link will use
	createMasterDummy(c, master)
	// create a network specifying the desired sub-interface name
	dockerCmd(c, "network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0.60", "dm-persist")
	assertNwIsAvailable(c, "dm-persist")
	// Restart docker daemon to test the config has persisted to disk
	s.d.Restart(c)
	// verify network is recreated from persistence
	assertNwIsAvailable(c, "dm-persist")
	// cleanup the master interface that also collects the slave dev
	deleteInterface(c, "dm-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanPersistance(c *check.C) {
	// verify the driver automatically provisions the 802.1q link (di-dummy0.70)
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	// master dummy interface 'di' notation represent 'docker ipvlan'
	master := "di-dummy0"
	// simulate the master link the vlan tagged subinterface parent link will use
	createMasterDummy(c, master)
	// create a network specifying the desired sub-interface name
	dockerCmd(c, "network", "create", "--driver=ipvlan", "-o", "parent=di-dummy0.70", "di-persist")
	assertNwIsAvailable(c, "di-persist")
	// Restart docker daemon to test the config has persisted to disk
	s.d.Restart(c)
	// verify network is recreated from persistence
	assertNwIsAvailable(c, "di-persist")
	// cleanup the master interface that also collects the slave dev
	deleteInterface(c, "di-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkMacvlanSubIntCreate(c *check.C) {
	// verify the driver automatically provisions the 802.1q link (dm-dummy0.50)
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	// master dummy interface 'dm' abbreviation represents 'docker macvlan'
	master := "dm-dummy0"
	// simulate the master link the vlan tagged subinterface parent link will use
	createMasterDummy(c, master)
	// create a network specifying the desired sub-interface name
	dockerCmd(c, "network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0.50", "dm-subinterface")
	assertNwIsAvailable(c, "dm-subinterface")
	// cleanup the master interface which also collects the slave dev
	deleteInterface(c, "dm-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanSubIntCreate(c *check.C) {
	// verify the driver automatically provisions the 802.1q link (di-dummy0.50)
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	// master dummy interface 'dm' abbreviation represents 'docker ipvlan'
	master := "di-dummy0"
	// simulate the master link the vlan tagged subinterface parent link will use
	createMasterDummy(c, master)
	// create a network specifying the desired sub-interface name
	dockerCmd(c, "network", "create", "--driver=ipvlan", "-o", "parent=di-dummy0.60", "di-subinterface")
	assertNwIsAvailable(c, "di-subinterface")
	// cleanup the master interface which also collects the slave dev
	deleteInterface(c, "di-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkMacvlanOverlapParent(c *check.C) {
	// verify the same parent interface cannot be used if already in use by an existing network
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	// master dummy interface 'dm' abbreviation represents 'docker macvlan'
	master := "dm-dummy0"
	createMasterDummy(c, master)
	createVlanInterface(c, master, "dm-dummy0.40", "40")
	// create a network using an existing parent interface
	dockerCmd(c, "network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0.40", "dm-subinterface")
	assertNwIsAvailable(c, "dm-subinterface")
	// attempt to create another network using the same parent iface that should fail
	out, _, err := dockerCmdWithError("network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0.40", "dm-parent-net-overlap")
	// verify that the overlap returns an error
	c.Assert(err, check.NotNil, check.Commentf(out))
	// cleanup the master interface which also collects the slave dev
	deleteInterface(c, "dm-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanOverlapParent(c *check.C) {
	// verify the same parent interface cannot be used if already in use by an existing network
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	// master dummy interface 'dm' abbreviation represents 'docker ipvlan'
	master := "di-dummy0"
	createMasterDummy(c, master)
	createVlanInterface(c, master, "di-dummy0.30", "30")
	// create a network using an existing parent interface
	dockerCmd(c, "network", "create", "--driver=ipvlan", "-o", "parent=di-dummy0.30", "di-subinterface")
	assertNwIsAvailable(c, "di-subinterface")
	// attempt to create another network using the same parent iface that should fail
	out, _, err := dockerCmdWithError("network", "create", "--driver=ipvlan", "-o", "parent=di-dummy0.30", "di-parent-net-overlap")
	// verify that the overlap returns an error
	c.Assert(err, check.NotNil, check.Commentf(out))
	// cleanup the master interface which also collects the slave dev
	deleteInterface(c, "di-dummy0")
}

func (s *DockerNetworkSuite) TestDockerNetworkMacvlanMultiSubnet(c *check.C) {
	// create a dual stack multi-subnet Macvlan bridge mode network and validate connectivity between four containers, two on each subnet
	testRequires(c, DaemonIsLinux, IPv6, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=macvlan", "--ipv6", "--subnet=172.28.100.0/24", "--subnet=172.28.102.0/24", "--gateway=172.28.102.254",
		"--subnet=2001:db8:abc2::/64", "--subnet=2001:db8:abc4::/64", "--gateway=2001:db8:abc4::254", "dualstackbridge")
	// Ensure the network was created
	assertNwIsAvailable(c, "dualstackbridge")
	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.100.0/24 and 2001:db8:abc2::/64
	dockerCmd(c, "run", "-d", "--net=dualstackbridge", "--name=first", "--ip", "172.28.100.20", "--ip6", "2001:db8:abc2::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackbridge", "--name=second", "--ip", "172.28.100.21", "--ip6", "2001:db8:abc2::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackbridge
	ip := inspectField(c, "first", "NetworkSettings.Networks.dualstackbridge.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackbridge
	ip6 := inspectField(c, "first", "NetworkSettings.Networks.dualstackbridge.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address second to first
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address second to first
	c.Skip("Temporarily skipping while investigating sporadic v6 CI issues")
	_, _, err = dockerCmdWithError("exec", "second", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.102.0/24 and 2001:db8:abc4::/64
	dockerCmd(c, "run", "-d", "--net=dualstackbridge", "--name=third", "--ip", "172.28.102.20", "--ip6", "2001:db8:abc4::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackbridge", "--name=fourth", "--ip", "172.28.102.21", "--ip6", "2001:db8:abc4::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackbridge
	ip = inspectField(c, "third", "NetworkSettings.Networks.dualstackbridge.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackbridge
	ip6 = inspectField(c, "third", "NetworkSettings.Networks.dualstackbridge.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// Inspect the v4 gateway to ensure the proper default GW was assigned
	ip4gw := inspectField(c, "first", "NetworkSettings.Networks.dualstackbridge.Gateway")
	c.Assert(strings.TrimSpace(ip4gw), check.Equals, "172.28.100.1")
	// Inspect the v6 gateway to ensure the proper default GW was assigned
	ip6gw := inspectField(c, "first", "NetworkSettings.Networks.dualstackbridge.IPv6Gateway")
	c.Assert(strings.TrimSpace(ip6gw), check.Equals, "2001:db8:abc2::1")

	// Inspect the v4 gateway to ensure the proper explicitly assigned default GW was assigned
	ip4gw = inspectField(c, "third", "NetworkSettings.Networks.dualstackbridge.Gateway")
	c.Assert(strings.TrimSpace(ip4gw), check.Equals, "172.28.102.254")
	// Inspect the v6 gateway to ensure the proper explicitly assigned default GW was assigned
	ip6gw = inspectField(c, "third", "NetworkSettings.Networks.dualstackbridge.IPv6Gateway")
	c.Assert(strings.TrimSpace(ip6gw), check.Equals, "2001:db8:abc4::254")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanL2MultiSubnet(c *check.C) {
	// create a dual stack multi-subnet Ipvlan L2 network and validate connectivity within the subnets, two on each subnet
	testRequires(c, DaemonIsLinux, IPv6, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=ipvlan", "--ipv6", "--subnet=172.28.200.0/24", "--subnet=172.28.202.0/24", "--gateway=172.28.202.254",
		"--subnet=2001:db8:abc8::/64", "--subnet=2001:db8:abc6::/64", "--gateway=2001:db8:abc6::254", "dualstackl2")
	// Ensure the network was created
	assertNwIsAvailable(c, "dualstackl2")
	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.200.0/24 and 2001:db8:abc8::/64
	dockerCmd(c, "run", "-d", "--net=dualstackl2", "--name=first", "--ip", "172.28.200.20", "--ip6", "2001:db8:abc8::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackl2", "--name=second", "--ip", "172.28.200.21", "--ip6", "2001:db8:abc8::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackl2
	ip := inspectField(c, "first", "NetworkSettings.Networks.dualstackl2.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackl2
	ip6 := inspectField(c, "first", "NetworkSettings.Networks.dualstackl2.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address second to first
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address second to first
	_, _, err = dockerCmdWithError("exec", "second", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.202.0/24 and 2001:db8:abc6::/64
	dockerCmd(c, "run", "-d", "--net=dualstackl2", "--name=third", "--ip", "172.28.202.20", "--ip6", "2001:db8:abc6::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackl2", "--name=fourth", "--ip", "172.28.202.21", "--ip6", "2001:db8:abc6::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackl2
	ip = inspectField(c, "third", "NetworkSettings.Networks.dualstackl2.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackl2
	ip6 = inspectField(c, "third", "NetworkSettings.Networks.dualstackl2.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// Inspect the v4 gateway to ensure the proper default GW was assigned
	ip4gw := inspectField(c, "first", "NetworkSettings.Networks.dualstackl2.Gateway")
	c.Assert(strings.TrimSpace(ip4gw), check.Equals, "172.28.200.1")
	// Inspect the v6 gateway to ensure the proper default GW was assigned
	ip6gw := inspectField(c, "first", "NetworkSettings.Networks.dualstackl2.IPv6Gateway")
	c.Assert(strings.TrimSpace(ip6gw), check.Equals, "2001:db8:abc8::1")

	// Inspect the v4 gateway to ensure the proper explicitly assigned default GW was assigned
	ip4gw = inspectField(c, "third", "NetworkSettings.Networks.dualstackl2.Gateway")
	c.Assert(strings.TrimSpace(ip4gw), check.Equals, "172.28.202.254")
	// Inspect the v6 gateway to ensure the proper explicitly assigned default GW was assigned
	ip6gw = inspectField(c, "third", "NetworkSettings.Networks.dualstackl2.IPv6Gateway")
	c.Assert(strings.TrimSpace(ip6gw), check.Equals, "2001:db8:abc6::254")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanL3MultiSubnet(c *check.C) {
	// create a dual stack multi-subnet Ipvlan L3 network and validate connectivity between all four containers per L3 mode
	testRequires(c, DaemonIsLinux, IPv6, ipvlanKernelSupport, NotUserNamespace, NotArm, IPv6, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=ipvlan", "--ipv6", "--subnet=172.28.10.0/24", "--subnet=172.28.12.0/24", "--gateway=172.28.12.254",
		"--subnet=2001:db8:abc9::/64", "--subnet=2001:db8:abc7::/64", "--gateway=2001:db8:abc7::254", "-o", "ipvlan_mode=l3", "dualstackl3")
	// Ensure the network was created
	assertNwIsAvailable(c, "dualstackl3")

	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.10.0/24 and 2001:db8:abc9::/64
	dockerCmd(c, "run", "-d", "--net=dualstackl3", "--name=first", "--ip", "172.28.10.20", "--ip6", "2001:db8:abc9::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackl3", "--name=second", "--ip", "172.28.10.21", "--ip6", "2001:db8:abc9::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackl3
	ip := inspectField(c, "first", "NetworkSettings.Networks.dualstackl3.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackl3
	ip6 := inspectField(c, "first", "NetworkSettings.Networks.dualstackl3.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address second to first
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address second to first
	_, _, err = dockerCmdWithError("exec", "second", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// start dual stack containers and verify the user specified --ip and --ip6 addresses on subnets 172.28.12.0/24 and 2001:db8:abc7::/64
	dockerCmd(c, "run", "-d", "--net=dualstackl3", "--name=third", "--ip", "172.28.12.20", "--ip6", "2001:db8:abc7::20", "busybox", "top")
	dockerCmd(c, "run", "-d", "--net=dualstackl3", "--name=fourth", "--ip", "172.28.12.21", "--ip6", "2001:db8:abc7::21", "busybox", "top")

	// Inspect and store the v4 address from specified container on the network dualstackl3
	ip = inspectField(c, "third", "NetworkSettings.Networks.dualstackl3.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackl3
	ip6 = inspectField(c, "third", "NetworkSettings.Networks.dualstackl3.GlobalIPv6Address")

	// verify ipv4 connectivity to the explicit --ipv address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	// verify ipv6 connectivity to the explicit --ipv6 address from third to fourth
	_, _, err = dockerCmdWithError("exec", "fourth", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// Inspect and store the v4 address from specified container on the network dualstackl3
	ip = inspectField(c, "second", "NetworkSettings.Networks.dualstackl3.IPAddress")
	// Inspect and store the v6 address from specified container on the network dualstackl3
	ip6 = inspectField(c, "second", "NetworkSettings.Networks.dualstackl3.GlobalIPv6Address")

	// Verify connectivity across disparate subnets which is unique to L3 mode only
	_, _, err = dockerCmdWithError("exec", "third", "ping", "-c", "1", strings.TrimSpace(ip))
	c.Assert(err, check.IsNil)
	_, _, err = dockerCmdWithError("exec", "third", "ping6", "-c", "1", strings.TrimSpace(ip6))
	c.Assert(err, check.IsNil)

	// Inspect the v4 gateway to ensure no next hop is assigned in L3 mode
	ip4gw := inspectField(c, "first", "NetworkSettings.Networks.dualstackl3.Gateway")
	c.Assert(strings.TrimSpace(ip4gw), check.Equals, "")
	// Inspect the v6 gateway to ensure the explicitly specified default GW is ignored per L3 mode enabled
	ip6gw := inspectField(c, "third", "NetworkSettings.Networks.dualstackl3.IPv6Gateway")
	c.Assert(strings.TrimSpace(ip6gw), check.Equals, "")
}

func (s *DockerNetworkSuite) TestDockerNetworkIpvlanAddressing(c *check.C) {
	// Ensure the default gateways, next-hops and default dev devices are properly set
	testRequires(c, DaemonIsLinux, IPv6, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=macvlan", "--ipv6", "--subnet=172.28.130.0/24",
		"--subnet=2001:db8:abca::/64", "--gateway=2001:db8:abca::254", "-o", "macvlan_mode=bridge", "dualstackbridge")
	assertNwIsAvailable(c, "dualstackbridge")
	dockerCmd(c, "run", "-d", "--net=dualstackbridge", "--name=first", "busybox", "top")
	// Validate macvlan bridge mode defaults gateway sets the default IPAM next-hop inferred from the subnet
	out, _, err := dockerCmdWithError("exec", "first", "ip", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default via 172.28.130.1 dev eth0")
	// Validate macvlan bridge mode sets the v6 gateway to the user specified default gateway/next-hop
	out, _, err = dockerCmdWithError("exec", "first", "ip", "-6", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default via 2001:db8:abca::254 dev eth0")

	// Verify ipvlan l2 mode sets the proper default gateway routes via netlink
	// for either an explicitly set route by the user or inferred via default IPAM
	dockerCmd(c, "network", "create", "--driver=ipvlan", "--ipv6", "--subnet=172.28.140.0/24", "--gateway=172.28.140.254",
		"--subnet=2001:db8:abcb::/64", "-o", "ipvlan_mode=l2", "dualstackl2")
	assertNwIsAvailable(c, "dualstackl2")
	dockerCmd(c, "run", "-d", "--net=dualstackl2", "--name=second", "busybox", "top")
	// Validate ipvlan l2 mode defaults gateway sets the default IPAM next-hop inferred from the subnet
	out, _, err = dockerCmdWithError("exec", "second", "ip", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default via 172.28.140.254 dev eth0")
	// Validate ipvlan l2 mode sets the v6 gateway to the user specified default gateway/next-hop
	out, _, err = dockerCmdWithError("exec", "second", "ip", "-6", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default via 2001:db8:abcb::1 dev eth0")

	// Validate ipvlan l3 mode sets the v4 gateway to dev eth0 and disregards any explicit or inferred next-hops
	dockerCmd(c, "network", "create", "--driver=ipvlan", "--ipv6", "--subnet=172.28.160.0/24", "--gateway=172.28.160.254",
		"--subnet=2001:db8:abcd::/64", "--gateway=2001:db8:abcd::254", "-o", "ipvlan_mode=l3", "dualstackl3")
	assertNwIsAvailable(c, "dualstackl3")
	dockerCmd(c, "run", "-d", "--net=dualstackl3", "--name=third", "busybox", "top")
	// Validate ipvlan l3 mode sets the v4 gateway to dev eth0 and disregards any explicit or inferred next-hops
	out, _, err = dockerCmdWithError("exec", "third", "ip", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default dev eth0")
	// Validate ipvlan l3 mode sets the v6 gateway to dev eth0 and disregards any explicit or inferred next-hops
	out, _, err = dockerCmdWithError("exec", "third", "ip", "-6", "route")
	c.Assert(err, check.IsNil)
	c.Assert(out, checker.Contains, "default dev eth0")
}

func (s *DockerSuite) TestDockerNetworkMacVlanBridgeNilParent(c *check.C) {
	// macvlan bridge mode - dummy parent interface is provisioned dynamically
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=macvlan", "dm-nil-parent")
	assertNwIsAvailable(c, "dm-nil-parent")

	// start two containers on the same subnet
	dockerCmd(c, "run", "-d", "--net=dm-nil-parent", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	dockerCmd(c, "run", "-d", "--net=dm-nil-parent", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// intra-network communications should succeed
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", "first")
	c.Assert(err, check.IsNil)
}

func (s *DockerSuite) TestDockerNetworkMacVlanBridgeInternalMode(c *check.C) {
	// macvlan bridge mode --internal containers can communicate inside the network but not externally
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	cli.DockerCmd(c, "network", "create", "--driver=macvlan", "--internal", "dm-internal")
	assertNwIsAvailable(c, "dm-internal")
	nr := getNetworkResource(c, "dm-internal")
	c.Assert(nr.Internal, checker.True)

	// start two containers on the same subnet
	cli.DockerCmd(c, "run", "-d", "--net=dm-internal", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	cli.DockerCmd(c, "run", "-d", "--net=dm-internal", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// access outside of the network should fail
	result := cli.Docker(cli.Args("exec", "first", "ping", "-c", "1", "-w", "1", "8.8.8.8"), cli.WithTimeout(time.Second))
	c.Assert(result, icmd.Matches, icmd.Expected{Timeout: true})

	// intra-network communications should succeed
	cli.DockerCmd(c, "exec", "second", "ping", "-c", "1", "first")
}

func (s *DockerSuite) TestDockerNetworkIpvlanL2NilParent(c *check.C) {
	// ipvlan l2 mode - dummy parent interface is provisioned dynamically
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=ipvlan", "di-nil-parent")
	assertNwIsAvailable(c, "di-nil-parent")

	// start two containers on the same subnet
	dockerCmd(c, "run", "-d", "--net=di-nil-parent", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	dockerCmd(c, "run", "-d", "--net=di-nil-parent", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// intra-network communications should succeed
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", "first")
	c.Assert(err, check.IsNil)
}

func (s *DockerSuite) TestDockerNetworkIpvlanL2InternalMode(c *check.C) {
	// ipvlan l2 mode --internal containers can communicate inside the network but not externally
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	cli.DockerCmd(c, "network", "create", "--driver=ipvlan", "--internal", "di-internal")
	assertNwIsAvailable(c, "di-internal")
	nr := getNetworkResource(c, "di-internal")
	c.Assert(nr.Internal, checker.True)

	// start two containers on the same subnet
	cli.DockerCmd(c, "run", "-d", "--net=di-internal", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	cli.DockerCmd(c, "run", "-d", "--net=di-internal", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// access outside of the network should fail
	result := cli.Docker(cli.Args("exec", "first", "ping", "-c", "1", "-w", "1", "8.8.8.8"), cli.WithTimeout(time.Second))
	c.Assert(result, icmd.Matches, icmd.Expected{Timeout: true})
	// intra-network communications should succeed
	cli.DockerCmd(c, "exec", "second", "ping", "-c", "1", "first")
}

func (s *DockerSuite) TestDockerNetworkIpvlanL3NilParent(c *check.C) {
	// ipvlan l3 mode - dummy parent interface is provisioned dynamically
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	dockerCmd(c, "network", "create", "--driver=ipvlan", "--subnet=172.28.230.0/24",
		"--subnet=172.28.220.0/24", "-o", "ipvlan_mode=l3", "di-nil-parent-l3")
	assertNwIsAvailable(c, "di-nil-parent-l3")

	// start two containers on separate subnets
	dockerCmd(c, "run", "-d", "--ip=172.28.220.10", "--net=di-nil-parent-l3", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	dockerCmd(c, "run", "-d", "--ip=172.28.230.10", "--net=di-nil-parent-l3", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// intra-network communications should succeed
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", "first")
	c.Assert(err, check.IsNil)
}

func (s *DockerSuite) TestDockerNetworkIpvlanL3InternalMode(c *check.C) {
	// ipvlan l3 mode --internal containers can communicate inside the network but not externally
	testRequires(c, DaemonIsLinux, ipvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	cli.DockerCmd(c, "network", "create", "--driver=ipvlan", "--subnet=172.28.230.0/24",
		"--subnet=172.28.220.0/24", "-o", "ipvlan_mode=l3", "--internal", "di-internal-l3")
	assertNwIsAvailable(c, "di-internal-l3")
	nr := getNetworkResource(c, "di-internal-l3")
	c.Assert(nr.Internal, checker.True)

	// start two containers on separate subnets
	cli.DockerCmd(c, "run", "-d", "--ip=172.28.220.10", "--net=di-internal-l3", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	cli.DockerCmd(c, "run", "-d", "--ip=172.28.230.10", "--net=di-internal-l3", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)

	// access outside of the network should fail
	result := cli.Docker(cli.Args("exec", "first", "ping", "-c", "1", "-w", "1", "8.8.8.8"), cli.WithTimeout(time.Second))
	c.Assert(result, icmd.Matches, icmd.Expected{Timeout: true})
	// intra-network communications should succeed
	cli.DockerCmd(c, "exec", "second", "ping", "-c", "1", "first")
}

func (s *DockerSuite) TestDockerNetworkMacVlanExistingParent(c *check.C) {
	// macvlan bridge mode - empty parent interface containers can reach each other internally but not externally
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	netName := "dm-parent-exists"
	createMasterDummy(c, "dm-dummy0")
	//out, err := createVlanInterface(c, "dm-parent", "dm-slave", "macvlan", "bridge")
	// create a network using an existing parent interface
	dockerCmd(c, "network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0", netName)
	assertNwIsAvailable(c, netName)
	// delete the network while preserving the parent link
	dockerCmd(c, "network", "rm", netName)
	assertNwNotAvailable(c, netName)
	// verify the network delete did not delete the predefined link
	linkExists(c, "dm-dummy0")
	deleteInterface(c, "dm-dummy0")
}

func (s *DockerSuite) TestDockerNetworkMacVlanSubinterface(c *check.C) {
	// macvlan bridge mode -  empty parent interface containers can reach each other internally but not externally
	testRequires(c, DaemonIsLinux, macvlanKernelSupport, NotUserNamespace, NotArm, ExperimentalDaemon)
	netName := "dm-subinterface"
	createMasterDummy(c, "dm-dummy0")
	createVlanInterface(c, "dm-dummy0", "dm-dummy0.20", "20")
	// create a network using an existing parent interface
	dockerCmd(c, "network", "create", "--driver=macvlan", "-o", "parent=dm-dummy0.20", netName)
	assertNwIsAvailable(c, netName)

	// start containers on 802.1q tagged '-o parent' sub-interface
	dockerCmd(c, "run", "-d", "--net=dm-subinterface", "--name=first", "busybox", "top")
	c.Assert(waitRun("first"), check.IsNil)
	dockerCmd(c, "run", "-d", "--net=dm-subinterface", "--name=second", "busybox", "top")
	c.Assert(waitRun("second"), check.IsNil)
	// verify containers can communicate
	_, _, err := dockerCmdWithError("exec", "second", "ping", "-c", "1", "first")
	c.Assert(err, check.IsNil)

	// remove the containers
	dockerCmd(c, "rm", "-f", "first")
	dockerCmd(c, "rm", "-f", "second")
	// delete the network while preserving the parent link
	dockerCmd(c, "network", "rm", netName)
	assertNwNotAvailable(c, netName)
	// verify the network delete did not delete the predefined sub-interface
	linkExists(c, "dm-dummy0.20")
	// delete the parent interface which also collects the slave
	deleteInterface(c, "dm-dummy0")
}

func createMasterDummy(c *check.C, master string) {
	// ip link add <dummy_name> type dummy
	icmd.RunCommand("ip", "link", "add", master, "type", "dummy").Assert(c, icmd.Success)
	icmd.RunCommand("ip", "link", "set", master, "up").Assert(c, icmd.Success)
}

func createVlanInterface(c *check.C, master, slave, id string) {
	// ip link add link <master> name <master>.<VID> type vlan id <VID>
	icmd.RunCommand("ip", "link", "add", "link", master, "name", slave, "type", "vlan", "id", id).Assert(c, icmd.Success)
	// ip link set <sub_interface_name> up
	icmd.RunCommand("ip", "link", "set", slave, "up").Assert(c, icmd.Success)
}

func linkExists(c *check.C, master string) {
	// verify the specified link exists, ip link show <link_name>
	icmd.RunCommand("ip", "link", "show", master).Assert(c, icmd.Success)
}
