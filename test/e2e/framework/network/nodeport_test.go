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

package network

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestCalculateRange(t *testing.T) {
	testCases := []struct {
		name string
		size int32
		want int32
	}{
		{name: "default node port range", size: 2768, want: 86},
		{name: "tiny range clamps to the minimum", size: 64, want: 16},
		{name: "huge range clamps to the maximum", size: 65535, want: 128},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := calculateRange(tc.size); got != tc.want {
				t.Errorf("calculateRange(%d) = %d, want %d", tc.size, got, tc.want)
			}
		})
	}
}

func newTestPortRange() *staticPortRange {
	return &staticPortRange{
		baseport:      30000,
		length:        4,
		reservedPorts: sets.New[int32](),
	}
}

func TestStaticPortRangeGetUnusedPort(t *testing.T) {
	s := newTestPortRange()
	port, err := s.getUnusedPort()
	if err != nil {
		t.Fatalf("getUnusedPort() returned an unexpected error: %v", err)
	}
	if port < s.baseport || port >= s.baseport+s.length {
		t.Errorf("getUnusedPort() = %d, want a port in [%d,%d)", port, s.baseport, s.baseport+s.length)
	}
}

func TestStaticPortRangeExhausted(t *testing.T) {
	s := newTestPortRange()
	for i := int32(0); i < s.length; i++ {
		port, err := s.getUnusedPort()
		if err != nil {
			t.Fatalf("getUnusedPort() returned an unexpected error on iteration %d: %v", i, err)
		}
		if !s.reservePort(port) {
			t.Fatalf("reservePort(%d) = false, want true", port)
		}
	}
	if _, err := s.getUnusedPort(); err == nil {
		t.Error("getUnusedPort() returned no error with every port reserved, want an error")
	}
}

func TestStaticPortRangeReservePort(t *testing.T) {
	s := newTestPortRange()

	if s.reservePort(s.baseport - 1) {
		t.Errorf("reservePort(%d) = true for a port below the range, want false", s.baseport-1)
	}
	if !s.reservePort(s.baseport) {
		t.Errorf("reservePort(%d) = false for the first port in the range, want true", s.baseport)
	}
	if s.reservePort(s.baseport) {
		t.Errorf("reservePort(%d) = true for an already reserved port, want false", s.baseport)
	}

	last := s.baseport + s.length - 1
	if !s.reservePort(last) {
		t.Errorf("reservePort(%d) = false for the last port in the range, want true", last)
	}

	s.releasePort(s.baseport)
	if !s.reservePort(s.baseport) {
		t.Errorf("reservePort(%d) = false after releasing it, want true", s.baseport)
	}
}

func TestStaticPortRangeGetUnusedPorts(t *testing.T) {
	s := newTestPortRange()
	ports, err := s.getUnusedPorts(3)
	if err != nil {
		t.Fatalf("getUnusedPorts(3) returned an unexpected error: %v", err)
	}

	if len(ports) != 3 {
		t.Fatalf("getUnusedPorts(3) returned %d ports, want 3", len(ports))
	}
	if unique := sets.New[int32](ports...); unique.Len() != len(ports) {
		t.Errorf("getUnusedPorts(3) returned duplicate ports: %v", ports)
	}
	for _, port := range ports {
		if port < s.baseport || port >= s.baseport+s.length {
			t.Errorf("getUnusedPorts(3) returned %d, want a port in [%d,%d)", port, s.baseport, s.baseport+s.length)
		}
	}

	// The caller creates the service before reserving, so getUnusedPorts must leave
	// the ports free.
	if s.reservedPorts.Len() != 0 {
		t.Errorf("getUnusedPorts(3) reserved %d ports, want 0", s.reservedPorts.Len())
	}
}

func TestStaticPortRangeGetUnusedPortsExhausted(t *testing.T) {
	s := newTestPortRange()
	if _, err := s.getUnusedPorts(int(s.length) + 1); err == nil {
		t.Error("getUnusedPorts() returned no error asking for more ports than the range holds, want an error")
	}

	// A reserved port is not free anymore, so the whole range can no longer be handed out.
	ports, err := s.getUnusedPorts(int(s.length))
	if err != nil {
		t.Fatalf("getUnusedPorts(%d) returned an unexpected error: %v", s.length, err)
	}
	if !s.reservePorts(ports[:1]) {
		t.Fatalf("reservePorts(%v) = false, want true", ports[:1])
	}
	if _, err := s.getUnusedPorts(int(s.length)); err == nil {
		t.Error("getUnusedPorts() returned no error with a port reserved, want an error")
	}
}

func TestStaticPortRangeReservePorts(t *testing.T) {
	s := newTestPortRange()
	ports, err := s.getUnusedPorts(2)
	if err != nil {
		t.Fatalf("getUnusedPorts(2) returned an unexpected error: %v", err)
	}

	if !s.reservePorts(ports) {
		t.Fatalf("reservePorts(%v) = false, want true", ports)
	}
	for _, port := range ports {
		if s.reservePort(port) {
			t.Errorf("reservePort(%d) = true, want false because it is already reserved", port)
		}
	}

	s.releasePorts(ports)
	for _, port := range ports {
		if !s.reservePort(port) {
			t.Errorf("reservePort(%d) = false after releasePorts, want true", port)
		}
	}
}

// reservePorts must not leave the ports it already took reserved when one of them fails.
func TestStaticPortRangeReservePortsPartialFailure(t *testing.T) {
	s := newTestPortRange()
	taken := s.baseport + 1
	if !s.reservePort(taken) {
		t.Fatalf("reservePort(%d) = false, want true", taken)
	}

	if s.reservePorts([]int32{s.baseport, taken}) {
		t.Error("reservePorts() = true with an already reserved port, want false")
	}
	if s.reservedPorts.Has(s.baseport) {
		t.Errorf("reservePorts() left %d reserved after failing, want it released", s.baseport)
	}
}
