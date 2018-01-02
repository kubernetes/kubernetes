package bridge

import (
	"net"
	"testing"

	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libnetwork/portmapper"
	"github.com/docker/libnetwork/testutils"
	"github.com/vishvananda/netlink"
)

const (
	iptablesTestBridgeIP = "192.168.42.1"
)

func TestProgramIPTable(t *testing.T) {
	// Create a test bridge with a basic bridge configuration (name + IPv4).
	defer testutils.SetupTestOSContext(t)()

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}

	createTestBridge(getBasicTestConfig(), &bridgeInterface{nlh: nh}, t)

	// Store various iptables chain rules we care for.
	rules := []struct {
		rule  iptRule
		descr string
	}{
		{iptRule{table: iptables.Filter, chain: "FORWARD", args: []string{"-d", "127.1.2.3", "-i", "lo", "-o", "lo", "-j", "DROP"}}, "Test Loopback"},
		{iptRule{table: iptables.Nat, chain: "POSTROUTING", preArgs: []string{"-t", "nat"}, args: []string{"-s", iptablesTestBridgeIP, "!", "-o", DefaultBridgeName, "-j", "MASQUERADE"}}, "NAT Test"},
		{iptRule{table: iptables.Filter, chain: "FORWARD", args: []string{"-o", DefaultBridgeName, "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"}}, "Test ACCEPT INCOMING"},
		{iptRule{table: iptables.Filter, chain: "FORWARD", args: []string{"-i", DefaultBridgeName, "!", "-o", DefaultBridgeName, "-j", "ACCEPT"}}, "Test ACCEPT NON_ICC OUTGOING"},
		{iptRule{table: iptables.Filter, chain: "FORWARD", args: []string{"-i", DefaultBridgeName, "-o", DefaultBridgeName, "-j", "ACCEPT"}}, "Test enable ICC"},
		{iptRule{table: iptables.Filter, chain: "FORWARD", args: []string{"-i", DefaultBridgeName, "-o", DefaultBridgeName, "-j", "DROP"}}, "Test disable ICC"},
	}

	// Assert the chain rules' insertion and removal.
	for _, c := range rules {
		assertIPTableChainProgramming(c.rule, c.descr, t)
	}
}

func TestSetupIPChains(t *testing.T) {
	// Create a test bridge with a basic bridge configuration (name + IPv4).
	defer testutils.SetupTestOSContext(t)()

	nh, err := netlink.NewHandle()
	if err != nil {
		t.Fatal(err)
	}

	driverconfig := &configuration{
		EnableIPTables: true,
	}
	d := &driver{
		config: driverconfig,
	}
	assertChainConfig(d, t)

	config := getBasicTestConfig()
	br := &bridgeInterface{nlh: nh}
	createTestBridge(config, br, t)

	assertBridgeConfig(config, br, d, t)

	config.EnableIPMasquerade = true
	assertBridgeConfig(config, br, d, t)

	config.EnableICC = true
	assertBridgeConfig(config, br, d, t)

	config.EnableIPMasquerade = false
	assertBridgeConfig(config, br, d, t)
}

func getBasicTestConfig() *networkConfiguration {
	config := &networkConfiguration{
		BridgeName:  DefaultBridgeName,
		AddressIPv4: &net.IPNet{IP: net.ParseIP(iptablesTestBridgeIP), Mask: net.CIDRMask(16, 32)}}
	return config
}

func createTestBridge(config *networkConfiguration, br *bridgeInterface, t *testing.T) {
	if err := setupDevice(config, br); err != nil {
		t.Fatalf("Failed to create the testing Bridge: %s", err.Error())
	}
	if err := setupBridgeIPv4(config, br); err != nil {
		t.Fatalf("Failed to bring up the testing Bridge: %s", err.Error())
	}
}

// Assert base function which pushes iptables chain rules on insertion and removal.
func assertIPTableChainProgramming(rule iptRule, descr string, t *testing.T) {
	// Add
	if err := programChainRule(rule, descr, true); err != nil {
		t.Fatalf("Failed to program iptable rule %s: %s", descr, err.Error())
	}
	if iptables.Exists(rule.table, rule.chain, rule.args...) == false {
		t.Fatalf("Failed to effectively program iptable rule: %s", descr)
	}

	// Remove
	if err := programChainRule(rule, descr, false); err != nil {
		t.Fatalf("Failed to remove iptable rule %s: %s", descr, err.Error())
	}
	if iptables.Exists(rule.table, rule.chain, rule.args...) == true {
		t.Fatalf("Failed to effectively remove iptable rule: %s", descr)
	}
}

// Assert function which create chains.
func assertChainConfig(d *driver, t *testing.T) {
	var err error

	d.natChain, d.filterChain, d.isolationChain, err = setupIPChains(d.config)
	if err != nil {
		t.Fatal(err)
	}
}

// Assert function which pushes chains based on bridge config parameters.
func assertBridgeConfig(config *networkConfiguration, br *bridgeInterface, d *driver, t *testing.T) {
	nw := bridgeNetwork{portMapper: portmapper.New(""),
		config: config}
	nw.driver = d

	// Attempt programming of ip tables.
	err := nw.setupIPTables(config, br)
	if err != nil {
		t.Fatalf("%v", err)
	}
}
