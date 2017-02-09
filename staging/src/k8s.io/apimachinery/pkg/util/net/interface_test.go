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
const gatewayfirstIpv6_1 = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                       
eth3	00000000	0100FE0A	0003	0	0	1024	00000000	0	0	0                                                                   
eth3	0000FE0AA1	00000000	0001	0	0	0	0080FFFF	0	0	0                                                                      
docker0	000011AC	00000000	0001	0	0	0	0000FFFF	0	0	0                                                                            
virbr0	007AA8C0	00000000	0001	0	0	0	00FFFFFF	0	0	0
`
const gatewayfirstIpv6_2 = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT                                                       
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

func TestGetRoutes(t *testing.T) {
	testCases := []struct {
		tcase    string
		route    string
		expected int
	}{
		{"gatewayfirst", gatewayfirst, 4},
		{"gatewaymiddle", gatewaymiddle, 4},
		{"gatewaylast", gatewaylast, 4},
		{"nothing", nothing, 0},
		{"gatewayfirstIpv6_1", gatewayfirstIpv6_1, 0},
		{"gatewayfirstIpv6_2", gatewayfirstIpv6_2, 0},
		{"route_Invalidhex", route_Invalidhex, 0},
	}
	for _, tc := range testCases {
		r := strings.NewReader(tc.route)
		routes, err := getRoutes(r)
		if len(routes) != tc.expected {
			t.Errorf("case[%v]: expected %v, got %v .err : %v", tc.tcase, tc.expected, len(routes), err)
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
		intf     net.Interface
		expected bool
	}{
		{"up", net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}, true},
		{"down", net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: 0}, false},
		{"nothing", net.Interface{}, false},
	}
	for _, tc := range testCases {
		it := isInterfaceUp(&tc.intf)
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
		expected net.IP
	}{
		{"ipv6", []net.Addr{addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}}, nil},
		{"invalidCIDR", []net.Addr{addrStruct{val: "fe80::2f7:67fff:fe6e:2956/64"}}, nil},
		{"loopback", []net.Addr{addrStruct{val: "127.0.0.1/24"}}, nil},
		{"ip4", []net.Addr{addrStruct{val: "10.254.12.132/17"}}, net.ParseIP("10.254.12.132")},

		{"nothing", []net.Addr{}, nil},
	}
	for _, tc := range testCases {
		ip, err := getFinalIP(tc.addr)
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

type validNetworkInterfaceWithLinkLocal struct {
}

func (_ validNetworkInterfaceWithLinkLocal) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ validNetworkInterfaceWithLinkLocal) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "169.254.162.166/16"}, addrStruct{val: "45.55.47.146/19"}}
	return ifat, nil
}

type validNetworkInterfacewithIpv6Only struct {
}

func (_ validNetworkInterfacewithIpv6Only) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ validNetworkInterfacewithIpv6Only) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "fe80::2f7:6fff:fe6e:2956/64"}}
	return ifat, nil
}

type noNetworkInterface struct {
}

func (_ noNetworkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return nil, fmt.Errorf("unable get Interface")
}
func (_ noNetworkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return nil, nil
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

type networkInterfacewithIpv6addrs struct {
}

func (_ networkInterfacewithIpv6addrs) InterfaceByName(intfName string) (*net.Interface, error) {
	c := net.Interface{Index: 0, MTU: 0, Name: "eth3", HardwareAddr: nil, Flags: net.FlagUp}
	return &c, nil
}
func (_ networkInterfacewithIpv6addrs) Addrs(intf *net.Interface) ([]net.Addr, error) {
	var ifat []net.Addr
	ifat = []net.Addr{addrStruct{val: "fe80::2f7:6ffff:fe6e:2956/64"}}
	return ifat, nil
}

func TestGetIPFromInterface(t *testing.T) {
	testCases := []struct {
		tcase    string
		nwname   string
		nw       networkInterfacer
		expected net.IP
	}{
		{"valid", "eth3", validNetworkInterface{}, net.ParseIP("10.254.71.145")},
		{"ipv6", "eth3", validNetworkInterfacewithIpv6Only{}, nil},
		{"nothing", "eth3", noNetworkInterface{}, nil},
	}
	for _, tc := range testCases {
		ip, err := getIPFromInterface(tc.nwname, tc.nw)
		if !ip.Equal(tc.expected) {
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
		{"valid_routefirst", strings.NewReader(gatewayfirst), validNetworkInterface{}, net.ParseIP("10.254.71.145")},
		{"valid_routelast", strings.NewReader(gatewaylast), validNetworkInterface{}, net.ParseIP("10.254.71.145")},
		{"valid_routemiddle", strings.NewReader(gatewaymiddle), validNetworkInterface{}, net.ParseIP("10.254.71.145")},
		{"valid_routemiddle_ipv6", strings.NewReader(gatewaymiddle), validNetworkInterfacewithIpv6Only{}, nil},
		{"no internet connection", strings.NewReader(noInternetConnection), validNetworkInterface{}, nil},
		{"no non-link-local ip", strings.NewReader(gatewayfirstLinkLocal), validNetworkInterfaceWithLinkLocal{}, net.ParseIP("45.55.47.146")},
		{"no route", strings.NewReader(nothing), validNetworkInterface{}, nil},
		{"no route file", nil, validNetworkInterface{}, nil},
		{"no interfaces", nil, noNetworkInterface{}, nil},
		{"no interface Addrs", strings.NewReader(gatewaymiddle), networkInterfacewithNoAddrs{}, nil},
		{"Invalid Addrs", strings.NewReader(gatewaymiddle), networkInterfacewithIpv6addrs{}, nil},
	}
	for _, tc := range testCases {
		ip, err := chooseHostInterfaceFromRoute(tc.inFile, tc.nw)
		if !ip.Equal(tc.expected) {
			t.Errorf("case[%v]: expected %v, got %+v .err : %v", tc.tcase, tc.expected, ip, err)
		}
	}
}
