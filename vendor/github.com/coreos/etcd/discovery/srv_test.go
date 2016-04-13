// Copyright 2015 CoreOS, Inc.
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

package discovery

import (
	"errors"
	"net"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

func TestSRVGetCluster(t *testing.T) {
	defer func() {
		lookupSRV = net.LookupSRV
		resolveTCPAddr = net.ResolveTCPAddr
	}()

	name := "dnsClusterTest"
	tests := []struct {
		withSSL    []*net.SRV
		withoutSSL []*net.SRV
		urls       []string
		dns        map[string]string

		expected string
	}{
		{
			[]*net.SRV{},
			[]*net.SRV{},
			nil,
			nil,

			"",
		},
		{
			[]*net.SRV{
				{Target: "10.0.0.1", Port: 2480},
				{Target: "10.0.0.2", Port: 2480},
				{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{},
			nil,
			nil,

			"0=https://10.0.0.1:2480,1=https://10.0.0.2:2480,2=https://10.0.0.3:2480",
		},
		{
			[]*net.SRV{
				{Target: "10.0.0.1", Port: 2480},
				{Target: "10.0.0.2", Port: 2480},
				{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{
				{Target: "10.0.0.1", Port: 2380},
			},
			nil,
			nil,
			"0=https://10.0.0.1:2480,1=https://10.0.0.2:2480,2=https://10.0.0.3:2480,3=http://10.0.0.1:2380",
		},
		{
			[]*net.SRV{
				{Target: "10.0.0.1", Port: 2480},
				{Target: "10.0.0.2", Port: 2480},
				{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{
				{Target: "10.0.0.1", Port: 2380},
			},
			[]string{"https://10.0.0.1:2480"},
			nil,
			"dnsClusterTest=https://10.0.0.1:2480,0=https://10.0.0.2:2480,1=https://10.0.0.3:2480,2=http://10.0.0.1:2380",
		},
		// matching local member with resolved addr and return unresolved hostnames
		{
			[]*net.SRV{
				{Target: "1.example.com.", Port: 2480},
				{Target: "2.example.com.", Port: 2480},
				{Target: "3.example.com.", Port: 2480},
			},
			nil,
			[]string{"https://10.0.0.1:2480"},
			map[string]string{"1.example.com:2480": "10.0.0.1:2480", "2.example.com:2480": "10.0.0.2:2480", "3.example.com:2480": "10.0.0.3:2480"},

			"dnsClusterTest=https://1.example.com:2480,0=https://2.example.com:2480,1=https://3.example.com:2480",
		},
	}

	for i, tt := range tests {
		lookupSRV = func(service string, proto string, domain string) (string, []*net.SRV, error) {
			if service == "etcd-server-ssl" {
				return "", tt.withSSL, nil
			}
			if service == "etcd-server" {
				return "", tt.withoutSSL, nil
			}
			return "", nil, errors.New("Unknown service in mock")
		}
		resolveTCPAddr = func(network, addr string) (*net.TCPAddr, error) {
			if tt.dns == nil || tt.dns[addr] == "" {
				return net.ResolveTCPAddr(network, addr)
			}
			return net.ResolveTCPAddr(network, tt.dns[addr])
		}
		urls := testutil.MustNewURLs(t, tt.urls)
		str, token, err := SRVGetCluster(name, "example.com", "token", urls)
		if err != nil {
			t.Fatalf("%d: err: %#v", i, err)
		}
		if token != "token" {
			t.Errorf("%d: token: %s", i, token)
		}
		if str != tt.expected {
			t.Errorf("#%d: cluster = %s, want %s", i, str, tt.expected)
		}
	}
}
