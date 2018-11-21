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

package portforward

import (
	"net"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/httpstream"
)

type fakeDialer struct {
	dialed             bool
	conn               httpstream.Connection
	err                error
	negotiatedProtocol string
}

func (d *fakeDialer) Dial(protocols ...string) (httpstream.Connection, string, error) {
	d.dialed = true
	return d.conn, d.negotiatedProtocol, d.err
}

func TestParsePortsAndNew(t *testing.T) {
	tests := []struct {
		input                   []string
		addresses               []string
		expectedPorts           []ForwardedPort
		expectedAddresses       []listenAddress
		expectPortParseError    bool
		expectAddressParseError bool
		expectNewError          bool
	}{
		{input: []string{}, expectNewError: true},
		{input: []string{"a"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{":a"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"-1"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"65536"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"0"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"0:0"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"a:5000"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"5000:a"}, expectPortParseError: true, expectAddressParseError: false, expectNewError: true},
		{input: []string{"5000:5000"}, addresses: []string{"127.0.0.257"}, expectPortParseError: false, expectAddressParseError: true, expectNewError: true},
		{input: []string{"5000:5000"}, addresses: []string{"::g"}, expectPortParseError: false, expectAddressParseError: true, expectNewError: true},
		{input: []string{"5000:5000"}, addresses: []string{"domain.invalid"}, expectPortParseError: false, expectAddressParseError: true, expectNewError: true},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"localhost"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "all"},
				{protocol: "tcp6", address: "::1", failureMode: "all"},
			},
		},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"localhost", "127.0.0.1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
				{protocol: "tcp6", address: "::1", failureMode: "all"},
			},
		},
		{
			input:     []string{"5000", "5000:5000", "8888:5000", "5000:8888", ":5000", "0:5000"},
			addresses: []string{"127.0.0.1", "::1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
				{5000, 5000},
				{8888, 5000},
				{5000, 8888},
				{0, 5000},
				{0, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
				{protocol: "tcp6", address: "::1", failureMode: "any"},
			},
		},
	}

	for i, test := range tests {
		parsedPorts, err := parsePorts(test.input)
		haveError := err != nil
		if e, a := test.expectPortParseError, haveError; e != a {
			t.Fatalf("%d: parsePorts: error expected=%t, got %t: %s", i, e, a, err)
		}

		// default to localhost
		if len(test.addresses) == 0 && len(test.expectedAddresses) == 0 {
			test.addresses = []string{"localhost"}
			test.expectedAddresses = []listenAddress{{protocol: "tcp4", address: "127.0.0.1"}, {protocol: "tcp6", address: "::1"}}
		}
		// assert address parser
		parsedAddresses, err := parseAddresses(test.addresses)
		haveError = err != nil
		if e, a := test.expectAddressParseError, haveError; e != a {
			t.Fatalf("%d: parseAddresses: error expected=%t, got %t: %s", i, e, a, err)
		}

		dialer := &fakeDialer{}
		expectedStopChan := make(chan struct{})
		readyChan := make(chan struct{})

		var pf *PortForwarder
		if len(test.addresses) > 0 {
			pf, err = NewOnAddresses(dialer, test.addresses, test.input, expectedStopChan, readyChan, os.Stdout, os.Stderr)
		} else {
			pf, err = New(dialer, test.input, expectedStopChan, readyChan, os.Stdout, os.Stderr)
		}
		haveError = err != nil
		if e, a := test.expectNewError, haveError; e != a {
			t.Fatalf("%d: New: error expected=%t, got %t: %s", i, e, a, err)
		}

		if test.expectPortParseError || test.expectAddressParseError || test.expectNewError {
			continue
		}

		sort.Slice(test.expectedAddresses, func(i, j int) bool { return test.expectedAddresses[i].address < test.expectedAddresses[j].address })
		sort.Slice(parsedAddresses, func(i, j int) bool { return parsedAddresses[i].address < parsedAddresses[j].address })

		if !reflect.DeepEqual(test.expectedAddresses, parsedAddresses) {
			t.Fatalf("%d: expectedAddresses: %v, got: %v", i, test.expectedAddresses, parsedAddresses)
		}

		for pi, expectedPort := range test.expectedPorts {
			if e, a := expectedPort.Local, parsedPorts[pi].Local; e != a {
				t.Fatalf("%d: local expected: %d, got: %d", i, e, a)
			}
			if e, a := expectedPort.Remote, parsedPorts[pi].Remote; e != a {
				t.Fatalf("%d: remote expected: %d, got: %d", i, e, a)
			}
		}

		if dialer.dialed {
			t.Fatalf("%d: expected not dialed", i)
		}
		if _, portErr := pf.GetPorts(); portErr == nil {
			t.Fatalf("%d: GetPorts: error expected but got nil", i)
		}

		// mock-signal the Ready channel
		close(readyChan)

		if ports, portErr := pf.GetPorts(); portErr != nil {
			t.Fatalf("%d: GetPorts: unable to retrieve ports: %s", i, portErr)
		} else if !reflect.DeepEqual(test.expectedPorts, ports) {
			t.Fatalf("%d: ports: expected %#v, got %#v", i, test.expectedPorts, ports)
		}
		if e, a := expectedStopChan, pf.stopChan; e != a {
			t.Fatalf("%d: stopChan: expected %#v, got %#v", i, e, a)
		}
		if pf.Ready == nil {
			t.Fatalf("%d: Ready should be non-nil", i)
		}
	}
}

type GetListenerTestCase struct {
	Hostname                string
	Protocol                string
	ShouldRaiseError        bool
	ExpectedListenerAddress string
}

func TestGetListener(t *testing.T) {
	var pf PortForwarder
	testCases := []GetListenerTestCase{
		{
			Hostname:                "localhost",
			Protocol:                "tcp4",
			ShouldRaiseError:        false,
			ExpectedListenerAddress: "127.0.0.1",
		},
		{
			Hostname:                "127.0.0.1",
			Protocol:                "tcp4",
			ShouldRaiseError:        false,
			ExpectedListenerAddress: "127.0.0.1",
		},
		{
			Hostname:                "::1",
			Protocol:                "tcp6",
			ShouldRaiseError:        false,
			ExpectedListenerAddress: "::1",
		},
		{
			Hostname:         "::1",
			Protocol:         "tcp4",
			ShouldRaiseError: true,
		},
		{
			Hostname:         "127.0.0.1",
			Protocol:         "tcp6",
			ShouldRaiseError: true,
		},
	}

	for i, testCase := range testCases {
		expectedListenerPort := "12345"
		listener, err := pf.getListener(testCase.Protocol, testCase.Hostname, &ForwardedPort{12345, 12345})
		if err != nil && strings.Contains(err.Error(), "cannot assign requested address") {
			t.Logf("Can't test #%d: %v", i, err)
			continue
		}
		errorRaised := err != nil

		if testCase.ShouldRaiseError != errorRaised {
			t.Errorf("Test case #%d failed: Data %v an error has been raised(%t) where it should not (or reciprocally): %v", i, testCase, testCase.ShouldRaiseError, err)
			continue
		}
		if errorRaised {
			continue
		}

		if listener == nil {
			t.Errorf("Test case #%d did not raise an error but failed in initializing listener", i)
			continue
		}

		host, port, _ := net.SplitHostPort(listener.Addr().String())
		t.Logf("Asked a %s forward for: %s:%v, got listener %s:%s, expected: %s", testCase.Protocol, testCase.Hostname, 12345, host, port, expectedListenerPort)
		if host != testCase.ExpectedListenerAddress {
			t.Errorf("Test case #%d failed: Listener does not listen on expected address: asked '%v' got '%v'", i, testCase.ExpectedListenerAddress, host)
		}
		if port != expectedListenerPort {
			t.Errorf("Test case #%d failed: Listener does not listen on expected port: asked %v got %v", i, expectedListenerPort, port)

		}
		listener.Close()

	}
}
