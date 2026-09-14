//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"net"
	"reflect"
	"sort"
	"testing"

	"github.com/vishvananda/netlink"
	netutils "k8s.io/utils/net"
)

// addr builds a netlink.Addr suitable for use in filterAddrsExcept tests.
func addr(ip string, mask, linkIndex int) netlink.Addr {
	parsed := netutils.ParseIPSloppy(ip)
	bits := 32
	if parsed.To4() == nil {
		bits = 128
	}
	return netlink.Addr{
		IPNet:     &net.IPNet{IP: parsed, Mask: net.CIDRMask(mask, bits)},
		LinkIndex: linkIndex,
	}
}

func collectIPs(addrs []net.Addr) []string {
	out := make([]string, 0, len(addrs))
	for _, a := range addrs {
		ipnet, ok := a.(*net.IPNet)
		if !ok {
			continue
		}
		out = append(out, ipnet.IP.String())
	}
	sort.Strings(out)
	return out
}

func TestFilterAddrsExcept(t *testing.T) {
	tests := []struct {
		name     string
		addrs    []netlink.Addr
		devIndex int
		expected []string
	}{
		{
			name: "filters addresses on dev by LinkIndex",
			addrs: []netlink.Addr{
				addr("192.168.1.10", 24, 2),
				addr("10.233.0.1", 32, 10),
				addr("10.233.0.2", 32, 10),
				addr("10.233.0.3", 32, 10),
				addr("fd00::1", 128, 2),
			},
			devIndex: 10,
			expected: []string{"192.168.1.10", "fd00::1"},
		},
		{
			name: "keeps everything when no address belongs to dev",
			addrs: []netlink.Addr{
				addr("192.168.1.10", 24, 2),
				addr("10.0.0.1", 24, 3),
			},
			devIndex: 10,
			expected: []string{"10.0.0.1", "192.168.1.10"},
		},
		{
			name: "drops nil IPNet defensively",
			addrs: []netlink.Addr{
				{IPNet: nil, LinkIndex: 2},
				addr("192.168.1.10", 24, 2),
			},
			devIndex: 10,
			expected: []string{"192.168.1.10"},
		},
		{
			name:     "empty input",
			addrs:    nil,
			devIndex: 10,
			expected: []string{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := collectIPs(filterAddrsExcept(tc.addrs, tc.devIndex))
			want := tc.expected
			if want == nil {
				want = []string{}
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("filterAddrsExcept(_, %d) = %v, want %v", tc.devIndex, got, want)
			}
		})
	}
}
