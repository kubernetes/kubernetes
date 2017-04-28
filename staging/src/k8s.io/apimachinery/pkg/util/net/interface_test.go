/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package net

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

func TestIsInterfaceUp(t *testing.T) {
	testCases := []struct {
		tcase    string
		intf     *net.Interface
		expected bool
	}{
		{"up", &net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}, true},
		{"down", &net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: 0}, false},
		{"no interface", nil, false},
	}
	for _, tc := range testCases {
		it := isInterfaceUp(tc.intf)
		if it != tc.expected {
			t.Errorf("case[%v]: expected %+v, got %+v .", tc.tcase, tc.expected, it)
		}
	}
}

func TestBuildingIPRouteCommand(t *testing.T) {
	expected := "{ /usr/sbin/ip route; } | /usr/bin/grep default"
	actual := buildRouteCommand(false)
	if actual != expected {
		t.Errorf("ip route command not correct for IPv4\n\texpected %q\n\tactual   %q", expected, actual)
	}

	expected = "{ /usr/sbin/ip route; /usr/sbin/ip -6 route; } | /usr/bin/grep default"
	actual = buildRouteCommand(true)
	if actual != expected {
		t.Errorf("ip route command not correct for IPv4\n\texpected %q\n\tactual   %q", expected, actual)
	}
}

func TestExceutingCommandToGetDefaultRoutes(t *testing.T) {
	expected := `default via 10.86.7.129 dev eth0
default via fe80::21f:caff:fea0:ec00
`
	// Relicate the compound command that is used for getting default routes,
	// only use echo commands, instead of 'ip route' commands, and simulate output.
	actual, err := getDefaultRoutes("{ echo \"default via 10.86.7.129 dev eth0\"; echo \"default via fe80::21f:caff:fea0:ec00\"; } | grep default")
	if err != nil {
		t.Errorf("unable to exec command to obtain default route info: %v", err)
	}
	if actual != expected {
		t.Errorf("did not get expected command output\n\texpected %q\n\tactual   %q", expected, actual)
	}
}

func TestFailureGettingDefaultRoutes(t *testing.T) {
	_, err := getDefaultRoutes("bogus-command-attempted")
	if err == nil {
		t.Errorf("expected command to fail to run")
	}
}

const no_default_route = ``
const one_ipv4_route = `default via 10.10.10.10 dev eth0
`
const ipv4_ipv6_routes = `default via 10.10.10.10 dev eth0
default via fe80::21f:caff:fea0:ec00 dev eth1  proto static  metric 100
`
const duplicate_routes = `default via 10.10.10.10 dev eth0
default via 10.10.10.10 dev eth0  proto static  metric 100
`
const bad_gw_address = `default via 10.10.10.10.10 dev eth0
`

func routesMatch(a, b Route) bool {
	return (a.Interface == a.Interface &&
		a.Gateway.Equal(b.Gateway) &&
		a.Destination.Equal(b.Destination))
}

func TestParsingDefaultRoutes(t *testing.T) {
	testCases := []struct {
		tcase     string
		rawRoutes string
		expected  []Route
	}{
		{"no default routes", no_default_route, []Route{}},
		{"ipv4 default route", one_ipv4_route,
			[]Route{{Interface: "eth0", Destination: ipv4_zero, Gateway: net.ParseIP("10.10.10.10")}}},
		{"ipv4/iv6 default routes", ipv4_ipv6_routes,
			[]Route{
				{Interface: "eth0", Destination: ipv4_zero, Gateway: net.ParseIP("10.10.10.10")},
				{Interface: "eth1", Destination: ipv6_zero, Gateway: net.ParseIP("fe80::21f:caff:fea0:ec00")},
			}},
		{"duplicate default routes", duplicate_routes,
			[]Route{
				{Interface: "eth0", Destination: ipv4_zero, Gateway: net.ParseIP("10.10.10.10")},
				{Interface: "eth0", Destination: ipv4_zero, Gateway: net.ParseIP("10.10.10.10")},
			}},
	}
	for _, tc := range testCases {
		routes, err := parseDefaultRoutes(tc.rawRoutes)
		if len(routes) != len(tc.expected) {
			t.Errorf("case[%s]: routes expected %d, got %d err : %v",
				tc.tcase, len(tc.expected), len(routes), err)
		}
		for i, route := range routes {
			if !routesMatch(route, tc.expected[i]) {
				t.Errorf("case[%s]: routes different: expected %+v, got %+v", tc.tcase, tc.expected[i], route)
			}
		}
	}
}

func TestParsingDefaultRouteFailures(t *testing.T) {
	_, err := parseDefaultRoutes(bad_gw_address)
	if err == nil {
		t.Errorf("should not have been able to parse bad GW address")
	}
}

type addrStruct struct{ val string }

func (a addrStruct) Network() string {
	return a.val
}
func (a addrStruct) String() string {
	return a.val
}

const (
	upInterface       = net.FlagUp | net.FlagBroadcast | net.FlagMulticast
	downInterface     = net.FlagBroadcast | net.FlagMulticast
	loopbackInterface = net.FlagUp | net.FlagLoopback
	p2pInterface      = net.FlagUp | net.FlagPointToPoint
)

func makeIntf(index int, name string, flags net.Flags) net.Interface {
	mac := net.HardwareAddr{0, 0x32, 0x7d, 0x69, 0xf7, byte(0x30 + index)}
	return net.Interface{
		Index:        index,
		MTU:          1500,
		Name:         name,
		HardwareAddr: mac,
		Flags:        flags}
}

// Unable to get interface(s)
type failGettingNetworkInterface struct {
}

func (_ failGettingNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return nil, fmt.Errorf("unable get Interface")
}
func (_ failGettingNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, nil
}
func (_ failGettingNetworkInterface) Interfaces() ([]net.Interface, error) {
	return nil, fmt.Errorf("mock failed getting all interfaces")
}

// No interfaces
type noNetworkInterface struct {
}

func (_ noNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return nil, fmt.Errorf("unable get Interface")
}
func (_ noNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, nil
}
func (_ noNetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{}, nil
}

// Interface is down
type downNetworkInterface struct {
}

func (_ downNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: downInterface}
	return &c, nil
}
func (_ downNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}, addrStruct{val: "10.254.71.145/17"}}
	return ifat, nil
}
func (_ downNetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", downInterface)}, nil
}

// Loopback interface
type loopbackNetworkInterface struct {
}

func (_ loopbackNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: loopbackInterface}
	return &c, nil
}
func (_ loopbackNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "::1/128"}, addrStruct{val: "127.0.0.1/8"}}
	return ifat, nil
}
func (_ loopbackNetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "lo", loopbackInterface)}, nil
}

// Point to point interface
type p2pNetworkInterface struct {
}

func (_ p2pNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: p2pInterface}
	return &c, nil
}
func (_ p2pNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "::1/128"}, addrStruct{val: "127.0.0.1/8"}}
	return ifat, nil
}
func (_ p2pNetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "lo", p2pInterface)}, nil
}

// Unable to get IP addresses for interface
type networkInterfaceFailGetAddrs struct {
}

func (_ networkInterfaceFailGetAddrs) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ networkInterfaceFailGetAddrs) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, fmt.Errorf("unable to get Addrs")
}
func (_ networkInterfaceFailGetAddrs) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", upInterface)}, nil
}

// No addresses for interfae
type networkInterfaceWithNoAddrs struct {
}

func (_ networkInterfaceWithNoAddrs) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ networkInterfaceWithNoAddrs) Addrs(intf *net.Interface) ([]net.Addr, error) {
	ifat := []net.Addr{}
	return ifat, nil
}
func (_ networkInterfaceWithNoAddrs) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", upInterface)}, nil
}

// Only with link local addresses
type networkInterfaceWithOnlyLinkLocals struct {
}

func (_ networkInterfaceWithOnlyLinkLocals) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ networkInterfaceWithOnlyLinkLocals) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "fe80::200/10"}, addrStruct{val: "169.254.162.166/16"}}
	return ifat, nil
}
func (_ networkInterfaceWithOnlyLinkLocals) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth0", upInterface)}, nil
}

// Invalid addresses for interface
type networkInterfaceWithInvalidAddr struct {
}

func (_ networkInterfaceWithInvalidAddr) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ networkInterfaceWithInvalidAddr) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "10.20.30.40.50/24"}}
	return ifat, nil
}
func (_ networkInterfaceWithInvalidAddr) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", upInterface)}, nil
}

// Valid addresses for interface
type validNetworkInterface struct {
}

func (_ validNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ validNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "2001::200/64"}, addrStruct{val: "10.254.71.145/17"}}
	return ifat, nil
}
func (_ validNetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", upInterface)}, nil
}

// Interface with only IPv6 address
type ipv6NetworkInterface struct {
}

func (_ ipv6NetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: upInterface}
	return &c, nil
}
func (_ ipv6NetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "2001::200/64"}}
	return ifat, nil
}
func (_ ipv6NetworkInterface) Interfaces() ([]net.Interface, error) {
	return []net.Interface{makeIntf(1, "eth3", upInterface)}, nil
}

func TestFindGlobalIP(t *testing.T) {
	testCases := []struct {
		tcase     string
		addr      []net.Addr
		allowIPv6 bool
		expected  net.IP
	}{
		{"no addresses", []net.Addr{}, true, nil},
		{"invalid CIDR", []net.Addr{addrStruct{val: "10.10.10.10.10/24"}}, false, nil},
		{"no IPv4", []net.Addr{addrStruct{val: "2001::5/64"}}, false, nil},
		{"loopback", []net.Addr{addrStruct{val: "127.0.0.1/24"}}, true, nil},
		{"LL unicast", []net.Addr{addrStruct{val: "fe80::dab1:90ff:fed4:ea38/64"}}, true, nil},
		{"ip4", []net.Addr{addrStruct{val: "10.20.30.40/24"}}, false, net.ParseIP("10.20.30.40")},
		{"ipv6", []net.Addr{addrStruct{val: "2001::5/64"}}, true, net.ParseIP("2001::5")},
		{"multiples", []net.Addr{addrStruct{val: "10.10.10.10/24"}, addrStruct{val: "20.20.20.20/24"}}, false, net.ParseIP("10.10.10.10")},
	}
	for _, tc := range testCases {
		ip, err := findGlobalIP(tc.addr, tc.allowIPv6)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%s]: expected %+v, got %+v with err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}

func TestGetGlobalIPFromInterface(t *testing.T) {
	testCases := []struct {
		tcase     string
		nwname    string
		nw        networkInterfacer
		allowIPv6 bool
		expected  net.IP
	}{
		{"no interface", "eth3", failGettingNetworkInterface{}, false, nil},
		{"not up", "eth3", downNetworkInterface{}, false, nil},
		{"get addr fail", "eth3", networkInterfaceFailGetAddrs{}, false, nil},
		{"no addresses", "eth3", networkInterfaceWithNoAddrs{}, false, nil},
		{"invalid address", "eth3", networkInterfaceWithInvalidAddr{}, false, nil},
		{"no match", "eth3", networkInterfaceWithOnlyLinkLocals{}, true, nil},
		{"ipv4", "eth3", validNetworkInterface{}, false, net.ParseIP("10.254.71.145")},
		{"ipv6", "eth3", validNetworkInterface{}, true, net.ParseIP("2001::200")},
	}
	for _, tc := range testCases {
		ip, err := getIPFromInterface(tc.nwname, tc.nw, tc.allowIPv6)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%s]: expected %+v, got %+v with err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}

type ExecCommandResults struct {
	stdOut   string
	exitCode int
}

// AddExecResult provides standard output and exit code for a mocked exec.Command()
func (e *ExecCommandResults) AddExecResult(stdout string, exitCode int) {
	e.stdOut = stdout
	e.exitCode = exitCode
}

// ExecCommand overrides exec.Command() for testing and will use mocked results.
// It runs the TestHelperProcess test case (only) from the test executable,
// under a separate process and provides mocked results as environment variables.
func (e *ExecCommandResults) ExecCommand(command string, args ...string) *exec.Cmd {
	cs := []string{"-test.run=TestHelperProcess", "--", command}
	cs = append(cs, args...)
	cmd := exec.Command(os.Args[0], cs...)

	stdout := fmt.Sprintf("EXEC_COMMAND_STDOUT=%q", e.stdOut)
	exitCode := fmt.Sprintf("EXEC_COMMAND_EXIT_CODE=%d", e.exitCode)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1", stdout, exitCode}
	return cmd
}

// TestHelperProcess will be invoked during testing to provide mock output and
// exit code for when exec.Command() is called by ExecCommand.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	stdout := os.Getenv("EXEC_COMMAND_STDOUT")
	exitCode, err := strconv.ParseInt(os.Getenv("EXEC_COMMAND_EXIT_CODE"), 10, 64)

	if err != nil {
		os.Exit(1)
	}

	fmt.Fprintf(os.Stdout, stdout)
	os.Exit(int(exitCode))
}

// TestChoosingInterfaceFromDefaultRoute tests the higher level
// chooseHostInterfaceFromRoute() method, mocking some of the
// lower levels so that happy path and error paths can be tested.
func TestChoosingInterfaceFromDefaultRoute(t *testing.T) {
	testCases := []struct {
		tcase       string
		includeIpv6 bool
		exitCode    int
		rawRoute    string
		nw          networkInterfacer
		expected    net.IP
		errStrFrag  string
	}{
		{"get route fail", false, 2, no_default_route, validNetworkInterface{}, nil, "x"},
		{"parse route fail", false, 0, bad_gw_address, validNetworkInterface{}, nil, "unable to parse gateway IP"},
		{"IP from i/f fail", false, 0, one_ipv4_route, failGettingNetworkInterface{}, nil, "unable get Interface"},
		{"ipv4", false, 0, one_ipv4_route, validNetworkInterface{}, net.ParseIP("10.254.71.145"), ""},
		{"ipv6", true, 0, ipv4_ipv6_routes, validNetworkInterface{}, net.ParseIP("2001::200"), ""},
		{"no match", true, 0, ipv4_ipv6_routes, networkInterfaceWithOnlyLinkLocals{}, nil, "unable to select an IP"},
	}
	execCommandHelper := &ExecCommandResults{}
	execCommand = execCommandHelper.ExecCommand
	defer func() { execCommand = exec.Command }()

	for _, tc := range testCases {
		execCommandHelper.AddExecResult(tc.rawRoute, tc.exitCode)
		ip, err := chooseHostInterfaceFromRoute(tc.includeIpv6, tc.nw)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%s]: expected %+v, got %+v with err : %v", tc.tcase, tc.expected, ip, err)
		}
		// On failures, make sure it failed for the right reason
		if err != nil && !strings.Contains(err.Error(), tc.errStrFrag) {
			t.Errorf("case[%s]: unable to find %q in error string %q", tc.tcase, tc.errStrFrag, err.Error())
		}
	}
}

func TestLoopbackOrPointToPointInterface(t *testing.T) {
	testCases := []struct {
		tcase    string
		nw       networkInterfacer
		expected bool
	}{
		{"normal I/F", validNetworkInterface{}, false},
		{"loopback I/F", loopbackNetworkInterface{}, true},
		{"p2p I/F", p2pNetworkInterface{}, true},
	}
	for _, tc := range testCases {
		intfs, _ := tc.nw.Interfaces()
		actual := isLoopbackOrPointToPoint(&intfs[0])
		if actual != tc.expected {
			t.Errorf("case[%s]: LB/P2P test of %+v expected %v, got %v", tc.tcase, intfs[0], actual, tc.expected)
		}
	}
}

func TestGetIPFromHostInterfaces(t *testing.T) {
	testCases := []struct {
		tcase      string
		allowIPv6  bool
		nw         networkInterfacer
		expected   net.IP
		errStrFrag string
	}{
		{"fail get I/Fs", false, failGettingNetworkInterface{}, nil, "failed getting all interfaces"},
		{"no interfaces", false, noNetworkInterface{}, nil, "no interfaces found"},
		{"I/F not up", false, downNetworkInterface{}, nil, "down interface"},
		{"loopback only", true, loopbackNetworkInterface{}, nil, "LB or P2P interface"},
		{"P2P I/F only", false, p2pNetworkInterface{}, nil, "LB or P2P interface"},
		{"fail get addrs", false, networkInterfaceFailGetAddrs{}, nil, "unable to get Addrs"},
		{"no addresses", false, networkInterfaceWithNoAddrs{}, nil, "no addresses"},
		{"invalid addr", false, networkInterfaceWithInvalidAddr{}, nil, "invalid CIDR"},
		{"LL unicast only", false, networkInterfaceWithOnlyLinkLocals{}, nil, "link local address"},
		{"no IPv4", false, ipv6NetworkInterface{}, nil, "non-IPv4 address"},
		{"IPv4", false, validNetworkInterface{}, net.ParseIP("10.254.71.145"), ""},
		{"IPv6", true, validNetworkInterface{}, net.ParseIP("2001::200"), ""},
	}

	for _, tc := range testCases {
		ip, err := chooseIPFromHostInterfaces(tc.allowIPv6, tc.nw)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%s]: expected %+v, got %+v with err : %v", tc.tcase, tc.expected, ip, err)
		}
		if err != nil && !strings.Contains(err.Error(), tc.errStrFrag) {
			t.Errorf("case[%s]: unable to find %q in error string %q", tc.tcase, tc.errStrFrag, err.Error())
		}
	}
}
