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

package client

import (
	"errors"
	"net"
	"reflect"
	"testing"
)

func TestSRVDiscover(t *testing.T) {
	defer func() { lookupSRV = net.LookupSRV }()

	tests := []struct {
		withSSL    []*net.SRV
		withoutSSL []*net.SRV
		expected   []string
	}{
		{
			[]*net.SRV{},
			[]*net.SRV{},
			[]string{},
		},
		{
			[]*net.SRV{
				&net.SRV{Target: "10.0.0.1", Port: 2480},
				&net.SRV{Target: "10.0.0.2", Port: 2480},
				&net.SRV{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{},
			[]string{"https://10.0.0.1:2480", "https://10.0.0.2:2480", "https://10.0.0.3:2480"},
		},
		{
			[]*net.SRV{
				&net.SRV{Target: "10.0.0.1", Port: 2480},
				&net.SRV{Target: "10.0.0.2", Port: 2480},
				&net.SRV{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{
				&net.SRV{Target: "10.0.0.1", Port: 7001},
			},
			[]string{"https://10.0.0.1:2480", "https://10.0.0.2:2480", "https://10.0.0.3:2480", "http://10.0.0.1:7001"},
		},
		{
			[]*net.SRV{
				&net.SRV{Target: "10.0.0.1", Port: 2480},
				&net.SRV{Target: "10.0.0.2", Port: 2480},
				&net.SRV{Target: "10.0.0.3", Port: 2480},
			},
			[]*net.SRV{
				&net.SRV{Target: "10.0.0.1", Port: 7001},
			},
			[]string{"https://10.0.0.1:2480", "https://10.0.0.2:2480", "https://10.0.0.3:2480", "http://10.0.0.1:7001"},
		},
		{
			[]*net.SRV{
				&net.SRV{Target: "a.example.com", Port: 2480},
				&net.SRV{Target: "b.example.com", Port: 2480},
				&net.SRV{Target: "c.example.com", Port: 2480},
			},
			[]*net.SRV{},
			[]string{"https://a.example.com:2480", "https://b.example.com:2480", "https://c.example.com:2480"},
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
			return "", nil, errors.New("Unkown service in mock")
		}

		d := NewSRVDiscover()

		endpoints, err := d.Discover("example.com")
		if err != nil {
			t.Fatalf("%d: err: %#v", i, err)
		}

		if !reflect.DeepEqual(endpoints, tt.expected) {
			t.Errorf("#%d: endpoints = %v, want %v", i, endpoints, tt.expected)
		}

	}
}
