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
	"errors"
	"net"
	"reflect"
	"testing"
	"time"

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

func Test_deleteRsFunc(t *testing.T) {
	tests := []struct {
		name         string
		existingIPVS *utilipvstest.FakeIPVS
		item         *listItem
		deleted      bool
		err          error
	}{
		{
			name: "delete UDP real server, should be deleted",
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.1.1.1",
						Port:     53,
						Protocol: "udp",
					}: {
						Address:  net.ParseIP("1.1.1.1"),
						Protocol: "udp",
						Port:     uint16(53),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.1.1.1",
						Port:     53,
						Protocol: "udp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(53),
							Weight:       100,
							ActiveConn:   10,
							InactiveConn: 10,
						},
					},
				},
			},
			item: &listItem{
				VirtualServer: &utilipvs.VirtualServer{
					Address:  net.ParseIP("1.1.1.1"),
					Protocol: "udp",
					Port:     uint16(53),
				},
				RealServer: &utilipvs.RealServer{
					Address:      net.ParseIP("10.0.0.1"),
					Port:         uint16(53),
					Weight:       100,
					ActiveConn:   10,
					InactiveConn: 10,
				},
				createdAt: time.Now(),
			},
			deleted: true,
			err:     nil,
		},
		{
			name: "delete TCP real server with no connections, should be deleted",
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.2.3.4"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.2.3.4",
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
			item: &listItem{
				VirtualServer: &utilipvs.VirtualServer{
					Address:  net.ParseIP("1.2.3.4"),
					Protocol: "tcp",
					Port:     uint16(80),
				},
				RealServer: &utilipvs.RealServer{
					Address:      net.ParseIP("10.0.0.1"),
					Port:         uint16(80),
					Weight:       100,
					ActiveConn:   0,
					InactiveConn: 0,
				},
				createdAt: time.Now(),
			},
			deleted: true,
			err:     nil,
		},
		{
			name: "delete TCP recent real server with connections, should not be deleted",
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.2.3.4"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   1,
							InactiveConn: 1,
						},
					},
				},
			},
			item: &listItem{
				VirtualServer: &utilipvs.VirtualServer{
					Address:  net.ParseIP("1.2.3.4"),
					Protocol: "tcp",
					Port:     uint16(80),
				},
				RealServer: &utilipvs.RealServer{
					Address:      net.ParseIP("10.0.0.1"),
					Port:         uint16(80),
					Weight:       100,
					ActiveConn:   1,
					InactiveConn: 1,
				},
				// graceful termination started 1 minute ago
				createdAt: time.Now().Add(-1 * time.Minute),
			},
			deleted: false,
			err:     nil,
		},
		{
			name: "delete TCP real server with connections but after grace period, should be deleted",
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.2.3.4"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   1,
							InactiveConn: 1,
						},
					},
				},
			},
			item: &listItem{
				VirtualServer: &utilipvs.VirtualServer{
					Address:  net.ParseIP("1.2.3.4"),
					Protocol: "tcp",
					Port:     uint16(80),
				},
				RealServer: &utilipvs.RealServer{
					Address:      net.ParseIP("10.0.0.1"),
					Port:         uint16(80),
					Weight:       100,
					ActiveConn:   1,
					InactiveConn: 1,
				},
				// graceful termination started 15 minute ago
				createdAt: time.Now().Add(-15 * time.Minute),
			},
			deleted: true,
			err:     nil,
		},
		{
			name: "delete real server that doesn't exist, should err",
			existingIPVS: &utilipvstest.FakeIPVS{
				Services: map[utilipvstest.ServiceKey]*utilipvs.VirtualServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						Address:  net.ParseIP("1.2.3.4"),
						Protocol: "tcp",
						Port:     uint16(80),
					},
				},
				Destinations: map[utilipvstest.ServiceKey][]*utilipvs.RealServer{
					{
						IP:       "1.2.3.4",
						Port:     80,
						Protocol: "tcp",
					}: {
						{
							Address:      net.ParseIP("10.0.0.1"),
							Port:         uint16(80),
							Weight:       100,
							ActiveConn:   1,
							InactiveConn: 1,
						},
					},
				},
			},
			item: &listItem{
				VirtualServer: &utilipvs.VirtualServer{
					Address:  net.ParseIP("1.2.3.4"),
					Protocol: "tcp",
					Port:     uint16(80),
				},
				RealServer: &utilipvs.RealServer{
					Address:      net.ParseIP("10.0.0.2"), // 10.0.0.2 doens't exist
					Port:         uint16(80),
					Weight:       100,
					ActiveConn:   1,
					InactiveConn: 1,
				},
				createdAt: time.Now(),
			},
			deleted: true,
			err:     errors.New("Failed to delete rs \"1.2.3.4:80/tcp/10.0.0.2:80\", can't find the real server"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ipvs := test.existingIPVS
			gracefulTerminationManager := NewGracefulTerminationManager(ipvs)

			deleted, err := gracefulTerminationManager.deleteRsFunc(test.item)
			if !reflect.DeepEqual(err, test.err) {
				t.Logf("actual error: %v", err)
				t.Logf("expected error: %v", test.err)
				t.Fatal("unexpected error deleting real server")
			}

			if deleted != test.deleted {
				t.Logf("actual deleted state: %v", deleted)
				t.Logf("expected deleted state: %v", test.deleted)
				t.Errorf("unexpected deleted state for real server")
			}

		})
	}
}
