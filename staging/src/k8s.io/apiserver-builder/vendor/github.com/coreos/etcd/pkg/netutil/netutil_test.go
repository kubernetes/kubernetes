// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package netutil

import (
	"errors"
	"net"
	"net/url"
	"reflect"
	"strconv"
	"testing"
)

func TestResolveTCPAddrs(t *testing.T) {
	defer func() { resolveTCPAddr = net.ResolveTCPAddr }()
	tests := []struct {
		urls     [][]url.URL
		expected [][]url.URL
		hostMap  map[string]string
		hasError bool
	}{
		{
			urls: [][]url.URL{
				{
					{Scheme: "http", Host: "127.0.0.1:4001"},
					{Scheme: "http", Host: "127.0.0.1:2379"},
				},
				{
					{Scheme: "http", Host: "127.0.0.1:7001"},
					{Scheme: "http", Host: "127.0.0.1:2380"},
				},
			},
			expected: [][]url.URL{
				{
					{Scheme: "http", Host: "127.0.0.1:4001"},
					{Scheme: "http", Host: "127.0.0.1:2379"},
				},
				{
					{Scheme: "http", Host: "127.0.0.1:7001"},
					{Scheme: "http", Host: "127.0.0.1:2380"},
				},
			},
		},
		{
			urls: [][]url.URL{
				{
					{Scheme: "http", Host: "infra0.example.com:4001"},
					{Scheme: "http", Host: "infra0.example.com:2379"},
				},
				{
					{Scheme: "http", Host: "infra0.example.com:7001"},
					{Scheme: "http", Host: "infra0.example.com:2380"},
				},
			},
			expected: [][]url.URL{
				{
					{Scheme: "http", Host: "10.0.1.10:4001"},
					{Scheme: "http", Host: "10.0.1.10:2379"},
				},
				{
					{Scheme: "http", Host: "10.0.1.10:7001"},
					{Scheme: "http", Host: "10.0.1.10:2380"},
				},
			},
			hostMap: map[string]string{
				"infra0.example.com": "10.0.1.10",
			},
			hasError: false,
		},
		{
			urls: [][]url.URL{
				{
					{Scheme: "http", Host: "infra0.example.com:4001"},
					{Scheme: "http", Host: "infra0.example.com:2379"},
				},
				{
					{Scheme: "http", Host: "infra0.example.com:7001"},
					{Scheme: "http", Host: "infra0.example.com:2380"},
				},
			},
			hostMap: map[string]string{
				"infra0.example.com": "",
			},
			hasError: true,
		},
		{
			urls: [][]url.URL{
				{
					{Scheme: "http", Host: "ssh://infra0.example.com:4001"},
					{Scheme: "http", Host: "ssh://infra0.example.com:2379"},
				},
				{
					{Scheme: "http", Host: "ssh://infra0.example.com:7001"},
					{Scheme: "http", Host: "ssh://infra0.example.com:2380"},
				},
			},
			hasError: true,
		},
	}
	for _, tt := range tests {
		resolveTCPAddr = func(network, addr string) (*net.TCPAddr, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}
			if tt.hostMap[host] == "" {
				return nil, errors.New("cannot resolve host.")
			}
			i, err := strconv.Atoi(port)
			if err != nil {
				return nil, err
			}
			return &net.TCPAddr{IP: net.ParseIP(tt.hostMap[host]), Port: i, Zone: ""}, nil
		}
		urls, err := resolveTCPAddrs(tt.urls)
		if tt.hasError {
			if err == nil {
				t.Errorf("expected error")
			}
			continue
		}
		if !reflect.DeepEqual(urls, tt.expected) {
			t.Errorf("expected: %v, got %v", tt.expected, urls)
		}
	}
}

func TestURLsEqual(t *testing.T) {
	defer func() { resolveTCPAddr = net.ResolveTCPAddr }()
	hostm := map[string]string{
		"example.com": "10.0.10.1",
		"first.com":   "10.0.11.1",
		"second.com":  "10.0.11.2",
	}
	resolveTCPAddr = func(network, addr string) (*net.TCPAddr, error) {
		host, port, err := net.SplitHostPort(addr)
		if _, ok := hostm[host]; !ok {
			return nil, errors.New("cannot resolve host.")
		}
		i, err := strconv.Atoi(port)
		if err != nil {
			return nil, err
		}
		return &net.TCPAddr{IP: net.ParseIP(hostm[host]), Port: i, Zone: ""}, nil
	}

	tests := []struct {
		a      []url.URL
		b      []url.URL
		expect bool
	}{
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.10.1:2379"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "example.com:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "10.0.10.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "example.com:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.10.1:2379"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2380"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2380"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "127.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "example.com:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}, {Scheme: "http", Host: "127.0.0.1:2380"}},
			expect: false,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "first.com:2379"}, {Scheme: "http", Host: "second.com:2380"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.11.1:2379"}, {Scheme: "http", Host: "10.0.11.2:2380"}},
			expect: true,
		},
		{
			a:      []url.URL{{Scheme: "http", Host: "second.com:2380"}, {Scheme: "http", Host: "first.com:2379"}},
			b:      []url.URL{{Scheme: "http", Host: "10.0.11.1:2379"}, {Scheme: "http", Host: "10.0.11.2:2380"}},
			expect: true,
		},
	}

	for _, test := range tests {
		result := urlsEqual(test.a, test.b)
		if result != test.expect {
			t.Errorf("a:%v b:%v, expected %v but %v", test.a, test.b, test.expect, result)
		}
	}
}
func TestURLStringsEqual(t *testing.T) {
	result := URLStringsEqual([]string{"http://127.0.0.1:8080"}, []string{"http://127.0.0.1:8080"})
	if !result {
		t.Errorf("unexpected result %v", result)
	}
}
