/*
Copyright 2023 The Kubernetes Authors.

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

package iptree

import (
	"math/rand"
	"net/netip"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
)

func Test_InsertGetDelete(t *testing.T) {
	testCases := []struct {
		name   string
		prefix netip.Prefix
	}{
		{
			name:   "ipv4",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
		},
		{
			name:   "ipv6",
			prefix: netip.MustParsePrefix("fd00:1:2:3::/124"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tree := New[int]()
			ok := tree.InsertPrefix(tc.prefix, 1)
			if ok {
				t.Fatal("should not exist")
			}
			if _, ok := tree.GetPrefix(tc.prefix); !ok {
				t.Errorf("CIDR %s not found", tc.prefix)
			}
			if ok := tree.DeletePrefix(tc.prefix); !ok {
				t.Errorf("CIDR %s not deleted", tc.prefix)
			}
			if _, ok := tree.GetPrefix(tc.prefix); ok {
				t.Errorf("CIDR %s found", tc.prefix)
			}
		})
	}

}

func TestBasicIPv4(t *testing.T) {
	tree := New[int]()
	// insert
	ipnet := netip.MustParsePrefix("192.168.0.0/24")
	ok := tree.InsertPrefix(ipnet, 1)
	if ok {
		t.Fatal("should not exist")
	}
	// check exist
	if _, ok := tree.GetPrefix(ipnet); !ok {
		t.Errorf("CIDR %s not found", ipnet)
	}

	// check does not exist
	ipnet2 := netip.MustParsePrefix("12.1.0.0/16")
	if _, ok := tree.GetPrefix(ipnet2); ok {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check insert existing prefix updates the value
	ok = tree.InsertPrefix(ipnet2, 2)
	if ok {
		t.Errorf("should not exist: %s", ipnet2)
	}

	ok = tree.InsertPrefix(ipnet2, 3)
	if !ok {
		t.Errorf("should be updated: %s", ipnet2)
	}

	if v, ok := tree.GetPrefix(ipnet2); !ok || v != 3 {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check longer prefix matching
	ipnet3 := netip.MustParsePrefix("12.1.0.2/32")
	lpm, _, ok := tree.LongestPrefixMatch(ipnet3)
	if !ok || lpm != ipnet2 {
		t.Errorf("expected %s got %s", ipnet2, lpm)
	}
}

func TestBasicIPv6(t *testing.T) {
	tree := New[int]()
	// insert
	ipnet := netip.MustParsePrefix("2001:db8::/64")
	ok := tree.InsertPrefix(ipnet, 1)
	if ok {
		t.Fatal("should not exist")
	}
	// check exist
	if _, ok := tree.GetPrefix(ipnet); !ok {
		t.Errorf("CIDR %s not found", ipnet)
	}

	// check does not exist
	ipnet2 := netip.MustParsePrefix("2001:db8:1:3:4::/64")
	if _, ok := tree.GetPrefix(ipnet2); ok {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check insert existing prefix updates the value
	ok = tree.InsertPrefix(ipnet2, 2)
	if ok {
		t.Errorf("should not exist: %s", ipnet2)
	}

	ok = tree.InsertPrefix(ipnet2, 3)
	if !ok {
		t.Errorf("should be updated: %s", ipnet2)
	}

	if v, ok := tree.GetPrefix(ipnet2); !ok || v != 3 {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check longer prefix matching
	ipnet3 := netip.MustParsePrefix("2001:db8:1:3:4::/96")
	lpm, _, ok := tree.LongestPrefixMatch(ipnet3)
	if !ok || lpm != ipnet2 {
		t.Errorf("expected %s got %s", ipnet2, lpm)
	}
}

func TestInsertGetDelete100K(t *testing.T) {
	testCases := []struct {
		name string
		is6  bool
	}{
		{
			name: "ipv4",
		},
		{
			name: "ipv6",
			is6:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cidrs := generateRandomCIDRs(tc.is6, 100*1000)
			tree := New[string]()

			for k := range cidrs {
				ok := tree.InsertPrefix(k, k.String())
				if ok {
					t.Errorf("error inserting: %v", k)
				}
			}

			if tree.Len(tc.is6) != len(cidrs) {
				t.Errorf("expected %d nodes on the tree, got %d", len(cidrs), tree.Len(tc.is6))
			}

			list := cidrs.UnsortedList()
			for _, k := range list {
				if v, ok := tree.GetPrefix(k); !ok {
					t.Errorf("CIDR %s not found", k)
					return
				} else if v != k.String() {
					t.Errorf("CIDR value %s not found", k)
					return
				}
				ok := tree.DeletePrefix(k)
				if !ok {
					t.Errorf("CIDR delete %s error", k)
				}
			}

			if tree.Len(tc.is6) != 0 {
				t.Errorf("No node expected on the tree, got: %d %v", tree.Len(tc.is6), cidrs)
			}
		})
	}
}

func Test_findAncestor(t *testing.T) {
	tests := []struct {
		name string
		a    netip.Prefix
		b    netip.Prefix
		want netip.Prefix
	}{
		{
			name: "ipv4 direct parent",
			a:    netip.MustParsePrefix("192.168.0.0/24"),
			b:    netip.MustParsePrefix("192.168.1.0/24"),
			want: netip.MustParsePrefix("192.168.0.0/23"),
		},
		{
			name: "ipv4 root parent ",
			a:    netip.MustParsePrefix("192.168.0.0/24"),
			b:    netip.MustParsePrefix("1.168.1.0/24"),
			want: netip.MustParsePrefix("0.0.0.0/0"),
		},
		{
			name: "ipv4 parent /1",
			a:    netip.MustParsePrefix("192.168.0.0/24"),
			b:    netip.MustParsePrefix("184.168.1.0/24"),
			want: netip.MustParsePrefix("128.0.0.0/1"),
		},
		{
			name: "ipv6 direct parent",
			a:    netip.MustParsePrefix("fd00:1:1:1::/64"),
			b:    netip.MustParsePrefix("fd00:1:1:2::/64"),
			want: netip.MustParsePrefix("fd00:1:1::/62"),
		},
		{
			name: "ipv6 root parent ",
			a:    netip.MustParsePrefix("fd00:1:1:1::/64"),
			b:    netip.MustParsePrefix("1:1:1:1::/64"),
			want: netip.MustParsePrefix("::/0"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findAncestor(tt.a, tt.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("findAncestor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_getBitFromAddr(t *testing.T) {
	tests := []struct {
		name string
		ip   netip.Addr
		pos  int
		want int
	}{
		// 192.168.0.0
		// 11000000.10101000.00000000.00000001
		{
			name: "ipv4 first is a one",
			ip:   netip.MustParseAddr("192.168.0.0"),
			pos:  1,
			want: 1,
		},
		{
			name: "ipv4 middle is a zero",
			ip:   netip.MustParseAddr("192.168.0.0"),
			pos:  16,
			want: 0,
		},
		{
			name: "ipv4 middle is a one",
			ip:   netip.MustParseAddr("192.168.0.0"),
			pos:  13,
			want: 1,
		},
		{
			name: "ipv4 last is a zero",
			ip:   netip.MustParseAddr("192.168.0.0"),
			pos:  32,
			want: 0,
		},
		// 2001:db8::ff00:42:8329
		// 0010000000000001:0000110110111000:0000000000000000:0000000000000000:0000000000000000:1111111100000000:0000000001000010:1000001100101001
		{
			name: "ipv6 first is a zero",
			ip:   netip.MustParseAddr("2001:db8::ff00:42:8329"),
			pos:  1,
			want: 0,
		},
		{
			name: "ipv6 middle is a zero",
			ip:   netip.MustParseAddr("2001:db8::ff00:42:8329"),
			pos:  56,
			want: 0,
		},
		{
			name: "ipv6 middle is a one",
			ip:   netip.MustParseAddr("2001:db8::ff00:42:8329"),
			pos:  81,
			want: 1,
		},
		{
			name: "ipv6 last is a one",
			ip:   netip.MustParseAddr("2001:db8::ff00:42:8329"),
			pos:  128,
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getBitFromAddr(tt.ip, tt.pos); got != tt.want {
				t.Errorf("getBitFromAddr() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestShortestPrefix(t *testing.T) {
	r := New[int]()

	keys := []string{
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
		"192.168.0.0/24",
		"192.168.0.0/16",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), 0)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len(false) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(false), len(keys))
	}

	type exp struct {
		inp string
		out string
	}
	cases := []exp{
		{"192.168.0.3/32", "192.168.0.0/16"},
		{"10.1.2.4/21", "10.0.0.0/8"},
		{"192.168.0.0/16", "192.168.0.0/16"},
		{"192.168.0.0/32", "192.168.0.0/16"},
		{"10.1.2.3/32", "10.0.0.0/8"},
	}
	for _, test := range cases {
		m, _, ok := r.ShortestPrefixMatch(netip.MustParsePrefix(test.inp))
		if !ok {
			t.Fatalf("no match: %v", test)
		}
		if m != netip.MustParsePrefix(test.out) {
			t.Fatalf("mis-match: %v %v", m, test)
		}
	}

	// not match
	_, _, ok := r.ShortestPrefixMatch(netip.MustParsePrefix("0.0.0.0/0"))
	if ok {
		t.Fatalf("match unexpected for 0.0.0.0/0")
	}
}

func TestLongestPrefixMatch(t *testing.T) {
	r := New[int]()

	keys := []string{
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
		"192.168.0.0/24",
		"192.168.0.0/16",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), 0)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len(false) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(false), len(keys))
	}

	type exp struct {
		inp string
		out string
	}
	cases := []exp{
		{"192.168.0.3/32", "192.168.0.0/24"},
		{"10.1.2.4/21", "10.0.0.0/8"},
		{"10.21.2.0/24", "10.21.0.0/16"},
		{"10.1.2.3/32", "10.1.2.3/32"},
	}
	for _, test := range cases {
		m, _, ok := r.LongestPrefixMatch(netip.MustParsePrefix(test.inp))
		if !ok {
			t.Fatalf("no match: %v", test)
		}
		if m != netip.MustParsePrefix(test.out) {
			t.Fatalf("mis-match: %v %v", m, test)
		}
	}
	// not match
	_, _, ok := r.LongestPrefixMatch(netip.MustParsePrefix("0.0.0.0/0"))
	if ok {
		t.Fatalf("match unexpected for 0.0.0.0/0")
	}
}

func TestTopLevelPrefixesV4(t *testing.T) {
	r := New[string]()

	keys := []string{
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
		"192.168.0.0/20",
		"192.168.1.0/24",
		"172.16.0.0/12",
		"172.21.23.0/24",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len(false) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(false), len(keys))
	}

	expected := []string{
		"10.0.0.0/8",
		"192.168.0.0/20",
		"172.16.0.0/12",
	}
	parents := r.TopLevelPrefixes(false)
	if len(parents) != len(expected) {
		t.Fatalf("bad len: %v %v", len(parents), len(expected))
	}

	for _, k := range expected {
		v, ok := parents[k]
		if !ok {
			t.Errorf("key %s not found", k)
		}
		if v != k {
			t.Errorf("value expected %s got %s", k, v)
		}
	}
}

func TestTopLevelPrefixesV6(t *testing.T) {
	r := New[string]()

	keys := []string{
		"2001:db8:1:2:3::/64",
		"2001:db8::/64",
		"2001:db8:1:1:1::/64",
		"2001:db8:1:1:1::/112",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}

	if r.Len(true) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(true), len(keys))
	}

	expected := []string{
		"2001:db8::/64",
		"2001:db8:1:2:3::/64",
		"2001:db8:1:1:1::/64",
	}
	parents := r.TopLevelPrefixes(true)
	if len(parents) != len(expected) {
		t.Fatalf("bad len: %v %v", len(parents), len(expected))
	}

	for _, k := range expected {
		v, ok := parents[k]
		if !ok {
			t.Errorf("key %s not found", k)
		}
		if v != k {
			t.Errorf("value expected %s got %s", k, v)
		}
	}
}

func TestWalkV4(t *testing.T) {
	r := New[int]()

	keys := []string{
		"10.0.0.0/8",
		"10.1.0.0/16",
		"10.1.1.0/24",
		"10.1.1.32/26",
		"10.1.1.33/32",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), 0)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len(false) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(false), len(keys))
	}

	// match exact prefix
	path := []string{}
	r.WalkPath(netip.MustParsePrefix("10.1.1.32/26"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[:4]) {
		t.Errorf("Walkpath expected %v got %v", keys[:4], path)
	}
	// not match on prefix
	path = []string{}
	r.WalkPath(netip.MustParsePrefix("10.1.1.33/26"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[:3]) {
		t.Errorf("Walkpath expected %v got %v", keys[:3], path)
	}
	// match exact prefix
	path = []string{}
	r.WalkPrefix(netip.MustParsePrefix("10.0.0.0/8"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys) {
		t.Errorf("WalkPrefix expected %v got %v", keys, path)
	}
	// not match on prefix
	path = []string{}
	r.WalkPrefix(netip.MustParsePrefix("10.0.0.0/9"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[1:]) {
		t.Errorf("WalkPrefix expected %v got %v", keys[1:], path)
	}
}

func TestWalkV6(t *testing.T) {
	r := New[int]()

	keys := []string{
		"2001:db8::/48",
		"2001:db8::/64",
		"2001:db8::/96",
		"2001:db8::/112",
		"2001:db8::/128",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), 0)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len(true) != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(false), len(keys))
	}

	// match exact prefix
	path := []string{}
	r.WalkPath(netip.MustParsePrefix("2001:db8::/112"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[:4]) {
		t.Errorf("Walkpath expected %v got %v", keys[:4], path)
	}
	// not match on prefix
	path = []string{}
	r.WalkPath(netip.MustParsePrefix("2001:db8::1/112"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[:3]) {
		t.Errorf("Walkpath expected %v got %v", keys[:3], path)
	}
	// match exact prefix
	path = []string{}
	r.WalkPrefix(netip.MustParsePrefix("2001:db8::/48"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys) {
		t.Errorf("WalkPrefix expected %v got %v", keys, path)
	}
	// not match on prefix
	path = []string{}
	r.WalkPrefix(netip.MustParsePrefix("2001:db8::/49"), func(k netip.Prefix, v int) bool {
		path = append(path, k.String())
		return false
	})
	if !cmp.Equal(path, keys[1:]) {
		t.Errorf("WalkPrefix expected %v got %v", keys[1:], path)
	}
}

func TestGetHostIPPrefixMatches(t *testing.T) {
	r := New[int]()

	keys := []string{
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
		"192.168.0.0/24",
		"192.168.0.0/16",
		"2001:db8::/48",
		"2001:db8::/64",
		"2001:db8::/96",
	}
	for _, k := range keys {
		ok := r.InsertPrefix(netip.MustParsePrefix(k), 0)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}

	type exp struct {
		inp string
		out []string
	}
	cases := []exp{
		{"192.168.0.3", []string{"192.168.0.0/24", "192.168.0.0/16"}},
		{"10.1.2.4", []string{"10.1.2.0/24", "10.0.0.0/8"}},
		{"10.1.2.0", []string{"10.0.0.0/8"}},
		{"10.1.2.255", []string{"10.0.0.0/8"}},
		{"192.168.0.0", []string{}},
		{"192.168.1.0", []string{"192.168.0.0/16"}},
		{"10.1.2.255", []string{"10.0.0.0/8"}},
		{"2001:db8::1", []string{"2001:db8::/96", "2001:db8::/64", "2001:db8::/48"}},
		{"2001:db8::", []string{}},
		{"2001:db8::ffff:ffff:ffff:ffff", []string{"2001:db8::/64", "2001:db8::/48"}},
	}
	for _, test := range cases {
		m := r.GetHostIPPrefixMatches(netip.MustParseAddr(test.inp))
		in := []netip.Prefix{}
		for k := range m {
			in = append(in, k)
		}
		out := []netip.Prefix{}
		for _, s := range test.out {
			out = append(out, netip.MustParsePrefix(s))
		}

		// sort by prefix bits to avoid flakes
		sort.Slice(in, func(i, j int) bool { return in[i].Bits() < in[j].Bits() })
		sort.Slice(out, func(i, j int) bool { return out[i].Bits() < out[j].Bits() })
		if !reflect.DeepEqual(in, out) {
			t.Fatalf("mis-match: %v %v", in, out)
		}
	}

	// not match
	_, _, ok := r.ShortestPrefixMatch(netip.MustParsePrefix("0.0.0.0/0"))
	if ok {
		t.Fatalf("match unexpected for 0.0.0.0/0")
	}
}

func Test_prefixContainIP(t *testing.T) {
	tests := []struct {
		name   string
		prefix netip.Prefix
		ip     netip.Addr
		want   bool
	}{
		{
			name:   "IPv4 contains",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.1"),
			want:   true,
		},
		{
			name:   "IPv4 network address",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.0"),
		},
		{
			name:   "IPv4 broadcast address",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:   "IPv4 does not contain",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.1.2"),
		},
		{
			name:   "IPv6 contains",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::1"),
			want:   true,
		},
		{
			name:   "IPv6 network address",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::"),
		},
		{
			name:   "IPv6 broadcast address",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::ffff:ffff"),
			want:   true,
		},
		{
			name:   "IPv6 does not contain",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2:1:2:3::1"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := prefixContainIP(tt.prefix, tt.ip); got != tt.want {
				t.Errorf("prefixContainIP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func BenchmarkInsertUpdate(b *testing.B) {
	r := New[bool]()
	ipList := generateRandomCIDRs(true, 20000).UnsortedList()
	for _, ip := range ipList {
		r.InsertPrefix(ip, true)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r.InsertPrefix(ipList[n%len(ipList)], true)
	}
}

func generateRandomCIDRs(is6 bool, number int) sets.Set[netip.Prefix] {
	n := 4
	if is6 {
		n = 16
	}
	cidrs := sets.Set[netip.Prefix]{}
	rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < number; i++ {
		bytes := make([]byte, n)
		for i := 0; i < n; i++ {
			bytes[i] = uint8(rand.Intn(255))
		}

		ip, ok := netip.AddrFromSlice(bytes)
		if !ok {
			continue
		}

		bits := rand.Intn(n * 8)
		prefix := netip.PrefixFrom(ip, bits).Masked()
		if prefix.IsValid() {
			cidrs.Insert(prefix)
		}
	}
	return cidrs
}
