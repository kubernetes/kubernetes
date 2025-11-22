//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

package conntrack

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
	"k8s.io/apimachinery/pkg/util/sets"
)

type fakeHandler struct {
	tableType             netlink.ConntrackTableType
	ipFamily              netlink.InetFamily
	filter                *conntrackFilter
	calls                 int
	throwEINTROnFirstCall bool
}

func (f *fakeHandler) ConntrackDeleteFilters(tableType netlink.ConntrackTableType, family netlink.InetFamily, netlinkFilters ...netlink.CustomConntrackFilter) (uint, error) {
	f.calls++
	if f.calls == 1 && f.throwEINTROnFirstCall {
		return 0, unix.EINTR
	}

	f.tableType = tableType
	f.ipFamily = family
	f.filter = netlinkFilters[0].(*conntrackFilter)
	return 0, nil
}

var _ netlinkHandler = (*fakeHandler)(nil)

func TestConntracker_ClearEntries(t *testing.T) {
	testCases := []struct {
		name                  string
		ipFamily              uint8
		filter                netlink.CustomConntrackFilter
		throwEINTROnFirstCall bool
	}{
		{
			name:     "no filter",
			ipFamily: unix.AF_INET,
		},
		{
			name:     "IPv4 filter",
			ipFamily: unix.AF_INET,
			filter: &conntrackFilter{
				protocol: unix.IPPROTO_TCP,
				serviceNodePortEndpoints: map[int]sets.Set[string]{
					32000: sets.New("10.244.10.10:8000"),
				},
				serviceIPEndpoints: map[string]sets.Set[string]{
					"10.96.10.10:80": sets.New("10.244.20.20:5000"),
				},
			},
		},
		{
			name:     "IPv6 filter",
			ipFamily: unix.AF_INET6,
			filter: &conntrackFilter{
				protocol: unix.IPPROTO_UDP,
				serviceNodePortEndpoints: map[int]sets.Set[string]{
					32000: sets.New("[fd00:1::6]:9090"),
				},
				serviceIPEndpoints: map[string]sets.Set[string]{
					"[fd00:1::5]:3000": sets.New("[fd00:1::6]:9090"),
				},
			},
		},
		{
			name:     "IPv4 filter with interrupt",
			ipFamily: unix.AF_INET,
			filter: &conntrackFilter{
				protocol: unix.IPPROTO_UDP,
				serviceNodePortEndpoints: map[int]sets.Set[string]{
					32000: sets.New("[fd00:1::6]:9090"),
				},
				serviceIPEndpoints: map[string]sets.Set[string]{
					"[fd00:1::5]:3000": sets.New("[fd00:1::6]:9090"),
				},
			},
			throwEINTROnFirstCall: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := &fakeHandler{
				throwEINTROnFirstCall: tc.throwEINTROnFirstCall,
			}
			ct := newConntracker(handler)

			var err error
			if tc.filter != nil {
				_, err = ct.ClearEntries(tc.ipFamily, tc.filter)
				require.Equal(t, netlink.ConntrackTableType(netlink.ConntrackTable), handler.tableType)
				require.Equal(t, netlink.InetFamily(tc.ipFamily), handler.ipFamily)
				require.Equal(t, tc.filter, handler.filter)
			} else {
				_, err = ct.ClearEntries(tc.ipFamily)
			}
			require.NoError(t, err)
		})
	}
}
