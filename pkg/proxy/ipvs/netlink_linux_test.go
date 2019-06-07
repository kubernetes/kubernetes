/*
Copyright 2017 The Kubernetes Authors.

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

package ipvs

import (
	"fmt"
	"github.com/vishvananda/netlink"
	"k8s.io/apimachinery/pkg/util/sets"
	"net"
	"syscall"
	"testing"
)

var linkindex map[string]int
var routes []netlink.Route

func getFakeLinkIndex(h *netlinkHandle, dev string) (int, error) {
	if "" == dev {
		return -1, nil
	}
	link, ok := linkindex[dev]
	if !ok {
		return -1, fmt.Errorf("error getting device %s", dev)
	}
	return link, nil
}
func ipNetEqual(ipn1 *net.IPNet, ipn2 *net.IPNet) bool {
	if ipn1 == ipn2 {
		return true
	}
	if ipn1 == nil || ipn2 == nil {
		return false
	}
	m1, _ := ipn1.Mask.Size()
	m2, _ := ipn2.Mask.Size()
	return m1 == m2 && ipn1.IP.Equal(ipn2.IP)
}
func fakeRouteListFiltered(h *netlinkHandle, family int, filter *netlink.Route, filterMask uint64) ([]netlink.Route, error) {
	var res []netlink.Route
	for _, route := range routes {

		if filter != nil {
			switch {
			case filterMask&netlink.RT_FILTER_TABLE != 0 && filter.Table != syscall.RT_TABLE_UNSPEC && route.Table != filter.Table:
				continue
			case filterMask&netlink.RT_FILTER_PROTOCOL != 0 && route.Protocol != filter.Protocol:
				continue
			case filterMask&netlink.RT_FILTER_SCOPE != 0 && route.Scope != filter.Scope:
				continue
			case filterMask&netlink.RT_FILTER_TYPE != 0 && route.Type != filter.Type:
				continue
			case filterMask&netlink.RT_FILTER_TOS != 0 && route.Tos != filter.Tos:
				continue
			case filterMask&netlink.RT_FILTER_OIF != 0 && route.LinkIndex != filter.LinkIndex:
				continue
			case filterMask&netlink.RT_FILTER_IIF != 0 && route.ILinkIndex != filter.ILinkIndex:
				continue
			case filterMask&netlink.RT_FILTER_GW != 0 && !route.Gw.Equal(filter.Gw):
				continue
			case filterMask&netlink.RT_FILTER_SRC != 0 && !route.Src.Equal(filter.Src):
				continue
			case filterMask&netlink.RT_FILTER_DST != 0:
				if filter.MPLSDst == nil || route.MPLSDst == nil || (*filter.MPLSDst) != (*route.MPLSDst) {
					if !ipNetEqual(route.Dst, filter.Dst) {
						continue
					}
				}
			}
		}
		res = append(res, route)
	}
	return res, nil
}

func TestGetLocalAddresses(t *testing.T) {
	//h := fakenetlinkHandle{netlinkHandle{netlink.Handle{}, false},make(map[string]int),nil}
	h := NewNetLinkHandle(false)
	getLinkIndex = getFakeLinkIndex
	routeListFiltered = fakeRouteListFiltered

	linkindex = map[string]int{"lo": 1, "eth1": 2, "docker0": 3}

	routes = []netlink.Route{
		{LinkIndex: 1, Scope: syscall.RT_SCOPE_HOST, Dst: &net.IPNet{IP: net.IP{127, 0, 0, 0}, Mask: net.IPMask{255, 0, 0, 0}}, Src: net.IP{127, 0, 0, 1}, Protocol: 2, Table: 255, Type: 2},
		{LinkIndex: 1, Scope: syscall.RT_SCOPE_HOST, Dst: &net.IPNet{IP: net.IP{127, 0, 0, 1}, Mask: net.IPMask{255, 255, 255, 255}}, Src: net.IP{127, 0, 0, 1}, Protocol: 2, Table: 255, Type: 2},
		{LinkIndex: 2, Scope: syscall.RT_SCOPE_HOST, Dst: &net.IPNet{IP: net.IP{10, 47, 184, 126}, Mask: net.IPMask{255, 255, 255, 255}}, Src: net.IP{10, 47, 184, 126}, Protocol: 2, Table: 255, Type: 2},
		{LinkIndex: 3, Scope: syscall.RT_SCOPE_HOST, Dst: &net.IPNet{IP: net.IP{172, 17, 0, 1}, Mask: net.IPMask{255, 255, 255, 255}}, Src: net.IP{172, 17, 0, 1}, Protocol: 2, Table: 255, Type: 2},
	}
	testCases := []struct {
		expected  sets.String
		dev       string
		filterDev string
	}{
		{sets.String{"10.47.184.126": {}}, "eth1", ""},
		{sets.String{"172.17.0.1": {}}, "docker0", ""},
		{sets.String{"127.0.0.1": {}}, "lo", ""},
		{sets.String{"127.0.0.1": {}, "172.17.0.1": {}}, "", "eth1"},
		{sets.String{}, "eth1", "eth1"},
		{sets.String{"127.0.0.1": {}, "172.17.0.1": {}, "10.47.184.126": {}}, "", ""},
	}
	for _, testCase := range testCases {
		if localaddr, err := h.GetLocalAddresses(testCase.dev, testCase.filterDev); err != nil {
			t.Errorf("Unexpected mismatch, expected: %v, got: %v", testCase.expected, err)
		} else if !localaddr.Equal(testCase.expected) {
			t.Errorf("Unexpected mismatch, expected: %v, got: %v", testCase.expected, localaddr)
		}
	}

}
