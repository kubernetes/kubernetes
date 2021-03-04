/*
Copyright 2019 The Kubernetes Authors.

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
	"testing"

	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilipvstest "k8s.io/kubernetes/pkg/util/ipvs/testing"
)

func Test_GracefulDeleteRS(t *testing.T) {
	tests := []struct {
		name         string
		vs           *utilipvs.VirtualServer
		rs           *utilipvs.RealServer
		existingIPVS *utilipvstest.FakeIPVS
		expectedIPVS *utilipvstest.FakeIPVS
		err          error
	}{
		{
			name: "graceful delete, no connections results in deleting the real server immediatetly",
			vs: &utilipvs.VirtualServer{
				Address:  net.ParseIP("1.1.1.1"),
				Protocol: "tcp",
				Port:     uint16(80),
			},
			rs: &utilipvs.RealServer{
				Address:      net.ParseIP("10.0.0.1"),
				Port:         uint16(80),
				Weight:       100,
				ActiveConn:   0,
				InactiveConn: 0,
			},
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   0,
							InactiveConn: 0,
						},
					},
				},
			},
			expectedIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {},
				},
			},
			err: nil,
		},
		{
			name: "graceful delete, real server has active connections, weight should be 0 but don't delete",
			vs: &utilipvs.VirtualServer{
				Address:  net.ParseIP("1.1.1.1"),
				Protocol: "tcp",
				Port:     uint16(80),
			},
			rs: &utilipvs.RealServer{
				Address:      net.ParseIP("10.0.0.1"),
				Port:         uint16(80),
				Weight:       100,
				ActiveConn:   10,
				InactiveConn: 0,
			},
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   10,
							InactiveConn: 0,
						},
					},
				},
			},
			expectedIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       0,
							ActiveConn:   10,
							InactiveConn: 0,
						},
					},
				},
			},
			err: nil,
		},
		{
			name: "graceful delete, real server has in-active connections, weight should be 0 but don't delete",
			vs: &utilipvs.VirtualServer{
				Address:  net.ParseIP("1.1.1.1"),
				Protocol: "tcp",
				Port:     uint16(80),
			},
			rs: &utilipvs.RealServer{
				Address:      net.ParseIP("10.0.0.1"),
				Port:         uint16(80),
				Weight:       100,
				ActiveConn:   0,
				InactiveConn: 10,
			},
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   0,
							InactiveConn: 10,
						},
					},
				},
			},
			expectedIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       0,
							ActiveConn:   0,
							InactiveConn: 10,
						},
					},
				},
			},
			err: nil,
		},
		{
			name: "graceful delete, real server has connections, but udp connections are deleted immediately",
			vs: &utilipvs.VirtualServer{
				Address:  net.ParseIP("1.1.1.1"),
				Protocol: "udp",
				Port:     uint16(80),
			},
			rs: &utilipvs.RealServer{
				Address:      net.ParseIP("10.0.0.1"),
				Port:         uint16(80),
				Weight:       100,
				ActiveConn:   10,
				InactiveConn: 10,
			},
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "udp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "udp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "udp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   10,
							InactiveConn: 10,
						},
					},
				},
			},
			expectedIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "udp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "udp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "udp",
					}: {}, // udp real server deleted immediately
				},
			},
			err: nil,
		},
		{
			name: "graceful delete, real server mismatch should be no-op",
			vs: &utilipvs.VirtualServer{
				Address:  net.ParseIP("1.1.1.1"),
				Protocol: "tcp",
				Port:     uint16(80),
			},
			rs: &utilipvs.RealServer{
				Address:      net.ParseIP("10.0.0.1"),
				Port:         uint16(81), // port mismatched
				Weight:       100,
				ActiveConn:   0,
				InactiveConn: 10,
			},
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   0,
							InactiveConn: 10,
						},
					},
				},
			},
			expectedIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   0,
							InactiveConn: 10,
						},
					},
				},
			},
			err: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipvs := test.existingIPVS
			gracefulTerminationManager := NewGracefulTerminationManager(ipvs)

			err := gracefulTerminationManager.GracefulDeleteRS(test.vs, test.rs)
			if err != test.err {
				t.Logf("actual err: %v", err)
				t.Logf("expected err: %v", test.err)
				t.Errorf("unexpected error")
			}

			if !reflect.DeepEqual(ipvs, test.expectedIPVS) {
				t.Logf("actual: %+v", ipvs)
				t.Logf("expected : %+v", test.expectedIPVS)
				t.Errorf("unexpected IPVS servers")
			}
		})
	}
}
