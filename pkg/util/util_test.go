/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"reflect"
	"strings"
	"testing"

	"github.com/ghodss/yaml"
)

func TestUntil(t *testing.T) {
	ch := make(chan struct{})
	close(ch)
	Until(func() {
		t.Fatal("should not have been invoked")
	}, 0, ch)

	ch = make(chan struct{})
	called := make(chan struct{})
	go func() {
		Until(func() {
			called <- struct{}{}
		}, 0, ch)
		close(called)
	}()
	<-called
	close(ch)
	<-called
}

func TestHandleCrash(t *testing.T) {
	count := 0
	expect := 10
	for i := 0; i < expect; i = i + 1 {
		defer HandleCrash()
		if i%2 == 0 {
			panic("Test Panic")
		}
		count = count + 1
	}
	if count != expect {
		t.Errorf("Expected %d iterations, found %d", expect, count)
	}
}

func TestCustomHandleCrash(t *testing.T) {
	old := PanicHandlers
	defer func() { PanicHandlers = old }()
	var result interface{}
	PanicHandlers = []func(interface{}){
		func(r interface{}) {
			result = r
		},
	}
	func() {
		defer HandleCrash()
		panic("test")
	}()
	if result != "test" {
		t.Errorf("did not receive custom handler")
	}
}

func TestCustomHandleError(t *testing.T) {
	old := ErrorHandlers
	defer func() { ErrorHandlers = old }()
	var result error
	ErrorHandlers = []func(error){
		func(err error) {
			result = err
		},
	}
	err := fmt.Errorf("test")
	HandleError(err)
	if result != err {
		t.Errorf("did not receive custom handler")
	}
}

func TestNewIntOrStringFromInt(t *testing.T) {
	i := NewIntOrStringFromInt(93)
	if i.Kind != IntstrInt || i.IntVal != 93 {
		t.Errorf("Expected IntVal=93, got %+v", i)
	}
}

func TestNewIntOrStringFromString(t *testing.T) {
	i := NewIntOrStringFromString("76")
	if i.Kind != IntstrString || i.StrVal != "76" {
		t.Errorf("Expected StrVal=\"76\", got %+v", i)
	}
}

type IntOrStringHolder struct {
	IOrS IntOrString `json:"val"`
}

func TestIntOrStringUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input  string
		result IntOrString
	}{
		{"val: 123\n", IntOrString{Kind: IntstrInt, IntVal: 123}},
		{"val: \"123\"\n", IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		var result IntOrStringHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected: %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestIntOrStringMarshalYAML(t *testing.T) {
	cases := []struct {
		input  IntOrString
		result string
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}, "val: 123\n"},
		{IntOrString{Kind: IntstrString, StrVal: "123"}, "val: \"123\"\n"},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestIntOrStringUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result IntOrString
	}{
		{"{\"val\": 123}", IntOrString{Kind: IntstrInt, IntVal: 123}},
		{"{\"val\": \"123\"}", IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		var result IntOrStringHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestIntOrStringMarshalJSON(t *testing.T) {
	cases := []struct {
		input  IntOrString
		result string
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}, "{\"val\":123}"},
		{IntOrString{Kind: IntstrString, StrVal: "123"}, "{\"val\":\"123\"}"},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestIntOrStringMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input IntOrString
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}},
		{IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("1: Failed to marshal input: '%v': %v", input, err)
		}

		var result IntOrStringHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("2: Failed to unmarshal '%+v': %v", string(jsonMarshalled), err)
		}

		if !reflect.DeepEqual(input, result) {
			t.Errorf("3: Failed to marshal input '%+v': got %+v", input, result)
		}
	}
}

func TestStringDiff(t *testing.T) {
	diff := StringDiff("aaabb", "aaacc")
	expect := "aaa\n\nA: bb\n\nB: cc\n\n"
	if diff != expect {
		t.Errorf("diff returned %v", diff)
	}
}

func TestCompileRegex(t *testing.T) {
	uncompiledRegexes := []string{"endsWithMe$", "^startingWithMe"}
	regexes, err := CompileRegexps(uncompiledRegexes)

	if err != nil {
		t.Errorf("Failed to compile legal regexes: '%v': %v", uncompiledRegexes, err)
	}
	if len(regexes) != len(uncompiledRegexes) {
		t.Errorf("Wrong number of regexes returned: '%v': %v", uncompiledRegexes, regexes)
	}

	if !regexes[0].MatchString("Something that endsWithMe") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if regexes[0].MatchString("Something that doesn't endsWithMe.") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if !regexes[1].MatchString("startingWithMe is very important") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
	if regexes[1].MatchString("not startingWithMe should fail") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
}

func TestAllPtrFieldsNil(t *testing.T) {
	testCases := []struct {
		obj      interface{}
		expected bool
	}{
		{struct{}{}, true},
		{struct{ Foo int }{12345}, true},
		{&struct{ Foo int }{12345}, true},
		{struct{ Foo *int }{nil}, true},
		{&struct{ Foo *int }{nil}, true},
		{struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{&struct {
			Foo int
			Bar *int
		}{12345, nil}, true},
		{struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{&struct {
			Foo *int
			Bar *int
		}{nil, nil}, true},
		{struct{ Foo *int }{new(int)}, false},
		{&struct{ Foo *int }{new(int)}, false},
		{struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{&struct {
			Foo *int
			Bar *int
		}{nil, new(int)}, false},
		{(*struct{})(nil), true},
	}
	for i, tc := range testCases {
		if AllPtrFieldsNil(tc.obj) != tc.expected {
			t.Errorf("case[%d]: expected %t, got %t", i, tc.expected, !tc.expected)
		}
	}
}

func TestSplitQualifiedName(t *testing.T) {
	testCases := []struct {
		input  string
		output []string
	}{
		{"kubernetes.io/blah", []string{"kubernetes.io", "blah"}},
		{"blah", []string{"", "blah"}},
		{"kubernetes.io/blah/blah", []string{"kubernetes.io", "blah"}},
	}
	for i, tc := range testCases {
		namespace, name := SplitQualifiedName(tc.input)
		if namespace != tc.output[0] || name != tc.output[1] {
			t.Errorf("case[%d]: expected (%q, %q), got (%q, %q)", i, tc.output[0], tc.output[1], namespace, name)
		}
	}
}

func TestJoinQualifiedName(t *testing.T) {
	testCases := []struct {
		input  []string
		output string
	}{
		{[]string{"kubernetes.io", "blah"}, "kubernetes.io/blah"},
		{[]string{"blah", ""}, "blah"},
		{[]string{"kubernetes.io", "blah"}, "kubernetes.io/blah"},
	}
	for i, tc := range testCases {
		res := JoinQualifiedName(tc.input[0], tc.input[1])
		if res != tc.output {
			t.Errorf("case[%d]: expected %q, got %q", i, tc.output, res)
		}
	}
}

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
