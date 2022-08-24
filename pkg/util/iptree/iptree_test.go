/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
)

func Test_InsertGetDelete(t *testing.T) {
	testCases := []struct {
		name   string
		prefix string
		is6    bool
	}{
		{
			name:   "ipv4",
			prefix: "192.168.0.0/24",
		},
		{
			name:   "ipv6",
			prefix: "fd00:1:2:3::/124",
			is6:    true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tree := New(tc.is6)
			_, ok := tree.Insert(tc.prefix, 1)
			if ok {
				t.Fatal("should not exist")
			}
			if _, ok := tree.Get(tc.prefix); !ok {
				t.Errorf("CIDR %s not found", tc.prefix)
			}
			if _, ok := tree.Delete(tc.prefix); !ok {
				t.Errorf("CIDR %s not deleted", tc.prefix)
			}
			if _, ok := tree.Get(tc.prefix); ok {
				t.Errorf("CIDR %s found", tc.prefix)
			}
		})
	}

}

func TestBasicIPv4(t *testing.T) {
	tree := New(false)
	// insert
	ipnet := "192.168.0.0/24"
	_, ok := tree.Insert(ipnet, 1)
	if ok {
		t.Fatal("should not exist")
	}
	// check exist
	if _, ok := tree.Get(ipnet); !ok {
		t.Errorf("CIDR %s not found", ipnet)
	}

	// check does not exist
	ipnet2 := "12.1.0.0/16"
	if _, ok := tree.Get(ipnet2); ok {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check insert existing prefix fails
	_, ok = tree.Insert(ipnet2, 2)
	if ok {
		t.Errorf("should not exist: %s", ipnet2)
	}

	_, ok = tree.Insert(ipnet2, 2)
	if !ok {
		t.Errorf("should be updated: %s", ipnet2)
	}

	if _, ok := tree.Get(ipnet2); !ok {
		t.Errorf("CIDR %s not expected", ipnet2)
	}

	// check longer prefix matching
	ipnet3 := "12.1.0.2/32"
	lpm, _, ok := tree.LongestPrefixMatch(ipnet3)
	if !ok || lpm != ipnet2 {
		t.Errorf("expected %s got %s", ipnet2, lpm)
	}
}

func generateRandomCIDR(is6 bool, number int) sets.String {
	n := 4
	if is6 {
		n = 16
	}
	ips := sets.NewString()
	rand.Seed(time.Now().UnixNano())
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
			ips.Insert(prefix.String())
		}
	}
	return ips
}

// Quis custodiet ipsos custodes?
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
			ips := generateRandomCIDR(tc.is6, 100*1000)
			tree := New(tc.is6)

			for k := range ips {
				_, ok := tree.Insert(k, k)
				if ok {
					t.Errorf("error inserting: %v", k)
				}
			}

			if tree.Len() != len(ips) {
				t.Errorf("expected %d nodes on the tree, got %d", len(ips), tree.Len())
			}

			list := ips.UnsortedList()
			for _, k := range list {
				if v, ok := tree.Get(k); !ok {
					t.Errorf("CIDR %s not found", k)
					return
				} else if v.(string) != k {
					t.Errorf("CIDR value %s not found", k)
					return
				}
				_, ok := tree.Delete(k)
				if !ok {
					t.Errorf("CIDR delete %s error", k)
				}
			}

			if tree.Len() != 0 {
				t.Errorf("No node expected on the tree, got: %d %v", tree.Len(), ips)
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
	r := New(false)

	keys := []string{
		"0.0.0.0/0",
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
	}
	for _, k := range keys {
		_, ok := r.Insert(k, nil)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len() != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(), len(keys))
	}

	type exp struct {
		inp string
		out string
	}
	cases := []exp{
		{"127.0.0.0/8", "0.0.0.0/0"},
		{"10.1.2.4/21", "0.0.0.0/0"},
	}
	for _, test := range cases {
		m, _, ok := r.ShortestPrefixMatch(test.inp)
		if !ok {
			t.Fatalf("no match: %v", test)
		}
		if m != test.out {
			t.Fatalf("mis-match: %v %v", m, test)
		}
	}
}

func TestLongestPrefixMatch(t *testing.T) {
	r := New(false)

	keys := []string{
		"0.0.0.0/0",
		"10.0.0.0/8",
		"10.21.0.0/16",
		"10.221.0.0/16",
		"10.1.2.3/32",
		"10.1.2.0/24",
	}
	for _, k := range keys {
		_, ok := r.Insert(k, nil)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len() != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(), len(keys))
	}

	type exp struct {
		inp string
		out string
	}
	cases := []exp{
		{"127.0.0.0/8", "0.0.0.0/0"},
		{"10.1.2.4/21", "10.0.0.0/8"},
		{"10.21.2.0/24", "10.21.0.0/16"},
	}
	for _, test := range cases {
		m, _, ok := r.LongestPrefixMatch(test.inp)
		if !ok {
			t.Fatalf("no match: %v", test)
		}
		if m != test.out {
			t.Fatalf("mis-match: %v %v", m, test)
		}
	}
}

func TestParents(t *testing.T) {
	r := New(false)

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
		_, ok := r.Insert(k, k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len() != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(), len(keys))
	}

	expected := []string{
		"10.0.0.0/8",
		"192.168.0.0/20",
		"172.16.0.0/12",
	}
	parents := r.Parents()
	if len(parents) != len(expected) {
		t.Fatalf("bad len: %v %v", len(parents), len(expected))
	}

	for _, k := range expected {
		v, ok := parents[k]
		if !ok {
			t.Errorf("key %s not found", k)
		}
		if v.(string) != k {
			t.Errorf("value expected %s got %s", k, v.(string))
		}
	}
}

func TestToMap(t *testing.T) {
	r := New(false)

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
		_, ok := r.Insert(k, k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if r.Len() != len(keys) {
		t.Fatalf("bad len: %v %v", r.Len(), len(keys))
	}

	m := r.ToMap()
	if r.Len() != len(m) {
		t.Fatalf("bad len: %v %v", r.Len(), len(m))
	}

	for _, k := range keys {
		v, ok := m[k]
		if !ok {
			t.Errorf("key %s not found", k)
		}
		if v.(string) != k {
			t.Errorf("value expected %s got %s", k, v.(string))
		}
	}

}

func TestEqual(t *testing.T) {
	r1 := New(false)
	r2 := New(false)
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
		_, ok := r1.Insert(k, k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
		_, ok = r2.Insert(k, k)
		if ok {
			t.Errorf("unexpected update on insert %s", k)
		}
	}
	if !r1.Equal(r2) {
		t.Fatalf("expected tree to be equal")
	}

	_, ok := r2.Insert("1.1.1.1/32", "value")
	if ok {
		t.Errorf("unexpected update on insert")
	}

	if r1.Equal(r2) {
		t.Fatalf("expected tree to not be equal")
	}

	_, ok = r1.Insert("1.1.1.1/32", "value")
	if ok {
		t.Errorf("unexpected update on insert")
	}

	if !r1.Equal(r2) {
		t.Fatalf("expected tree to be equal")
	}

	_, ok = r1.Insert("1.1.1.1/32", "different-value")
	if !ok {
		t.Errorf("expected update on insert")
	}

	if r1.Equal(r2) {
		t.Fatalf("expected tree to not be equal")
	}
}

func BenchmarkInsertUpdate(b *testing.B) {
	r := New(true)
	ipList := generateRandomCIDR(true, 20000).UnsortedList()
	for _, ip := range ipList {
		r.Insert(ip, true)
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		r.Insert(ipList[n%len(ipList)], true)
	}
}
