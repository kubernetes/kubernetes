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
	"io"
	"net"
	"strings"
	"testing"
)

const gatewayfirst = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                       
eth3	00000000	0100FE0A	0003	0	0	1024	00000000	0	0	0                                                                   
eth3	0000FE0A	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`
const gatewaylast = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT  
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0                                                                                                                     
eth3	0000FE0A	00000000	0001	0	0	0	0080FFFF	0	0	0       
eth3	00000000	0100FE0A	0003	0	0	1024	00000000	0	0	0                                                                 
`
const gatewaymiddle = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                                                                                     
eth3	0000FE0A	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0       
eth3	00000000	0100FE0A	0003	0	0	1024	00000000	0	0	0                                                                         
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`
const noInternetConnection = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                       
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0            
`
const nothing = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                            
`
const badDestination = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                       
eth3	00000000	0100FE0A	0003	0	0	1024	00000000	0	0	0                                                                   
eth3	0000FE0AA1	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`
const badGateway = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth3	00000000	0100FE0AA1	0003	0	0	1024	00000000	0	0	0                                                                   
eth3	0000FE0A	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`
const route_Invalidhex = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth3	00000000	0100FE0AA	0003	0	0	1024	00000000	0	0	0                                                                   
eth3	0000FE0A	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`

// Based on DigitalOcean COREOS
const gatewayfirstLinkLocal = `Iface   Destination     Gateway         Flags   RefCnt  Use     Metric  Mask            MTU     Window  IRTT                                                       
eth0    00000000        0120372D        0001    0       0       0       00000000        0       0       0                                                                               
eth0    00000000        00000000        0001    0       0       2048    00000000        0       0       0                                                                            
`

var (
	ipv4Route = Route{Interface: "eth3", Gateway: net.ParseIP("10.254.0.1")}
)

func TestGetRoutes(t *testing.T) {
	testCases := []struct {
		tcase      string
		route      string
		count      int
		expected   *Route
		errStrFrag string
	}{
		{"gatewayfirst", gatewayfirst, 1, &ipv4Route, ""},
		{"gatewaymiddle", gatewaymiddle, 1, &ipv4Route, ""},
		{"gatewaylast", gatewaylast, 1, &ipv4Route, ""},
		{"no routes", nothing, 0, nil, ""},
		{"badDestination", badDestination, 0, nil, "invalid IPv4"},
		{"badGateway", badGateway, 0, nil, "invalid IPv4"},
		{"route_Invalidhex", route_Invalidhex, 0, nil, "odd length hex string"},
		{"no default routes", noInternetConnection, 0, nil, ""},
	}
	for _, tc := range testCases {
		r := strings.NewReader(tc.route)
		routes, err := getRoutes(r)
		if err != nil {
			if !strings.Contains(err.Error(), tc.errStrFrag) {
				t.Errorf("case[%s]: Error string %q does not contain %q", tc.tcase, err, tc.errStrFrag)
			}
		} else if tc.errStrFrag != "" {
			t.Errorf("case[%s]: Error %q expected, but not seen", tc.tcase, tc.errStrFrag)
		} else {
			if tc.count != len(routes) {
				t.Errorf("case[%s]: expected %d routes, have %v", tc.tcase, tc.count, routes)
			} else if tc.count == 1 && !tc.expected.Gateway.Equal(routes[0].Gateway) {
				t.Errorf("case[%s]: expected %v, got %v .err : %v", tc.tcase, tc.expected, routes, err)
			}
		}
	}
}

func TestParseIP(t *testing.T) {
	testCases := []struct {
		tcase    string
		ip       string
		success  bool
		expected net.IP
	}{
		{"empty", "", false, nil},
		{"too short", "AA", false, nil},
		{"too long", "0011223344", false, nil},
		{"invalid", "invalid!", false, nil},
		{"zero", "00000000", true, net.IP{0, 0, 0, 0}},
		{"ffff", "FFFFFFFF", true, net.IP{0xff, 0xff, 0xff, 0xff}},
		{"valid", "12345678", true, net.IP{120, 86, 52, 18}},
	}
	for _, tc := range testCases {
		ip, err := parseIP(tc.ip)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%v]: expected %q, got %q . err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}

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
			t.Errorf("case[%v]: expected %v, got %v .", tc.tcase, tc.expected, it)
		}
	}
}

type addrStruct struct{ val string }

func (a addrStruct) Network() string {
	return a.val
}
func (a addrStruct) String() string {
	return a.val
}

func TestFinalIP(t *testing.T) {
	testCases := []struct {
		tcase    string
		addr     []net.Addr
		family   AddressFamily
		expected net.IP
	}{
		{"no ipv4", []net.Addr{addrStruct{val: "2001::5/64"}}, familyIPv4, nil},
		{"no ipv6", []net.Addr{addrStruct{val: "10.128.0.4/32"}}, familyIPv6, nil},
		{"invalidV4CIDR", []net.Addr{addrStruct{val: "10.20.30.40.50/24"}}, familyIPv4, nil},
		{"invalidV6CIDR", []net.Addr{addrStruct{val: "fe80::2f7:67fff:fe6e:2956/64"}}, familyIPv6, nil},
		{"loopback", []net.Addr{addrStruct{val: "127.0.0.1/24"}}, familyIPv4, nil},
		{"loopbackv6", []net.Addr{addrStruct{val: "::1/128"}}, familyIPv6, nil},
		{"link local v4", []net.Addr{addrStruct{val: "169.254.1.10/16"}}, familyIPv4, nil},
		{"link local v6", []net.Addr{addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}}, familyIPv6, nil},
		{"ip4", []net.Addr{addrStruct{val: "10.254.12.132/17"}}, familyIPv4, net.ParseIP("10.254.12.132")},
		{"ip6", []net.Addr{addrStruct{val: "2001::5/64"}}, familyIPv6, net.ParseIP("2001::5")},

		{"no addresses", []net.Addr{}, familyIPv4, nil},
	}
	for _, tc := range testCases {
		ip, err := getFinalIP(tc.addr, tc.family)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%v]: expected %v, got %v .err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}

func TestAddrs(t *testing.T) {
	var nw networkInterfacer = validNetworkInterface{}
	intf := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: 0}
	addrs, err := nw.Addrs(&intf)
	if err != nil {
		t.Errorf("expected no error got : %v", err)
	}
	if len(addrs) != 2 {
		t.Errorf("expected addrs: 2 got null")
	}
}

type validNetworkInterface struct {
}

func (_ validNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ validNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}, addrStruct{val: "10.254.71.145/17"}}
	return ifat, nil
}

// Interface with only IPv6 address
type ipv6NetworkInterface struct {
}

func (_ ipv6NetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ ipv6NetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "2001::200/64"}}
	return ifat, nil
}

type validNetworkInterfaceWithOnlyLinkLocals struct {
}

func (_ validNetworkInterfaceWithOnlyLinkLocals) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ validNetworkInterfaceWithOnlyLinkLocals) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "169.254.162.166/16"}, addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}}
	return ifat, nil
}

type noNetworkInterface struct {
}

func (_ noNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return nil, fmt.Errorf("no such network interface")
}
func (_ noNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, nil
}

// Interface is down
type downNetworkInterface struct {
}

func (_ downNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: 0}
	return &c, nil
}
func (_ downNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{
		addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}, addrStruct{val: "10.254.71.145/17"}}
	return ifat, nil
}

type networkInterfacewithNoAddrs struct {
}

func (_ networkInterfacewithNoAddrs) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ networkInterfacewithNoAddrs) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, fmt.Errorf("unable get Addrs")
}

// Invalid addresses for interface
type networkInterfaceWithInvalidAddr struct {
}

func (_ networkInterfaceWithInvalidAddr) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ networkInterfaceWithInvalidAddr) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "10.20.30.40.50/24"}}
	return ifat, nil
}

func TestGetIPFromInterface(t *testing.T) {
	testCases := []struct {
		tcase      string
		nwname     string
		family     AddressFamily
		nw         networkInterfacer
		expected   net.IP
		errStrFrag string
	}{
		{"ipv4", "eth3", familyIPv4, validNetworkInterface{}, net.ParseIP("10.254.71.145"), ""},
		{"ipv6", "eth3", familyIPv6, ipv6NetworkInterface{}, net.ParseIP("2001::200"), ""},
		{"no ipv4", "eth3", familyIPv4, ipv6NetworkInterface{}, nil, ""},
		{"no ipv6", "eth3", familyIPv6, validNetworkInterface{}, nil, ""},
		{"I/F down", "eth3", familyIPv4, downNetworkInterface{}, nil, ""},
		{"get addrs fail", "eth3", familyIPv4, noNetworkInterface{}, nil, "no such network interface"},
		{"fail to get IP", "eth3", familyIPv4, networkInterfaceWithInvalidAddr{}, nil, "invalid CIDR"},
	}
	for _, tc := range testCases {
		ip, err := getIPFromInterface(tc.nwname, tc.family, tc.nw)
		if err != nil {
			if !strings.Contains(err.Error(), tc.errStrFrag) {
				t.Errorf("case[%s]: Error string %q does not contain %q", tc.tcase, err, tc.errStrFrag)
			}
		} else if tc.errStrFrag != "" {
			t.Errorf("case[%s]: Error %q expected, but not seen", tc.tcase, tc.errStrFrag)
		} else if !ip.Equal(tc.expected) {
			t.Errorf("case[%v]: expected %v, got %+v .err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}

func TestChooseHostInterfaceFromRoute(t *testing.T) {
	testCases := []struct {
		tcase    string
		inFile   io.Reader
		nw       networkInterfacer
		expected net.IP
	}{
		{"ipv4", strings.NewReader(gatewayfirst), validNetworkInterface{}, net.ParseIP("10.254.71.145")},
		{"ipv6", strings.NewReader(gatewaymiddle), ipv6NetworkInterface{}, net.ParseIP("2001::200")},
		{"no non-link-local ip", strings.NewReader(gatewaymiddle), validNetworkInterfaceWithOnlyLinkLocals{}, nil},
		{"no routes", strings.NewReader(nothing), validNetworkInterface{}, nil},
		{"no route file", nil, validNetworkInterface{}, nil},
		{"no interfaces", nil, noNetworkInterface{}, nil},
		{"no interface addrs", strings.NewReader(gatewaymiddle), networkInterfacewithNoAddrs{}, nil},
	}
	for _, tc := range testCases {
		ip, err := chooseHostInterfaceFromRoute(tc.inFile, tc.nw)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%v]: expected %v, got %+v .err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}
