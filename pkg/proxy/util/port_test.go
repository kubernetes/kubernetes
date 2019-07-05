/*
Copyright 2017 The Kubernetes Authors.

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

package util

import "testing"

type fakeClosable struct {
	closed bool
}

func (c *fakeClosable) Close() error {
	c.closed = true
	return nil
}

func TestLocalPortString(t *testing.T) {
	testCases := []struct {
		description string
		ip          string
		port        int
		protocol    string
		expectedStr string
	}{
		{"IPv4 UDP", "1.2.3.4", 9999, "udp", "\"IPv4 UDP\" (1.2.3.4:9999/udp)"},
		{"IPv4 TCP", "5.6.7.8", 1053, "tcp", "\"IPv4 TCP\" (5.6.7.8:1053/tcp)"},
		{"IPv6 TCP", "2001:db8::1", 80, "tcp", "\"IPv6 TCP\" ([2001:db8::1]:80/tcp)"},
		{"IPv4 SCTP", "9.10.11.12", 7777, "sctp", "\"IPv4 SCTP\" (9.10.11.12:7777/sctp)"},
		{"IPv6 SCTP", "2001:db8::2", 80, "sctp", "\"IPv6 SCTP\" ([2001:db8::2]:80/sctp)"},
	}

	for _, tc := range testCases {
		lp := &LocalPort{
			Description: tc.description,
			IP:          tc.ip,
			Port:        tc.port,
			Protocol:    tc.protocol,
		}
		str := lp.String()
		if str != tc.expectedStr {
			t.Errorf("Unexpected output for %s, expected: %s, got: %s", tc.description, tc.expectedStr, str)
		}
	}
}

func TestRevertPorts(t *testing.T) {
	testCases := []struct {
		replacementPorts []LocalPort
		existingPorts    []LocalPort
		expectToBeClose  []bool
	}{
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts:   []LocalPort{},
			expectToBeClose: []bool{true, true, true},
		},
		{
			replacementPorts: []LocalPort{},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, false, false},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5003},
			},
			expectToBeClose: []bool{false, true, false},
		},
		{
			replacementPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
			},
			existingPorts: []LocalPort{
				{Port: 5001},
				{Port: 5002},
				{Port: 5003},
				{Port: 5004},
			},
			expectToBeClose: []bool{false, false, false},
		},
	}

	for i, tc := range testCases {
		replacementPortsMap := make(map[LocalPort]Closeable)
		for _, lp := range tc.replacementPorts {
			replacementPortsMap[lp] = &fakeClosable{}
		}
		existingPortsMap := make(map[LocalPort]Closeable)
		for _, lp := range tc.existingPorts {
			existingPortsMap[lp] = &fakeClosable{}
		}
		RevertPorts(replacementPortsMap, existingPortsMap)
		for j, expectation := range tc.expectToBeClose {
			if replacementPortsMap[tc.replacementPorts[j]].(*fakeClosable).closed != expectation {
				t.Errorf("Expect replacement localport %v to be %v in test case %v", tc.replacementPorts[j], expectation, i)
			}
		}
		for _, lp := range tc.existingPorts {
			if existingPortsMap[lp].(*fakeClosable).closed == true {
				t.Errorf("Expect existing localport %v to be false in test case %v", lp, i)
			}
		}
	}
}
