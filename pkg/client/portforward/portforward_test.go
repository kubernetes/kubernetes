/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/util/httpstream"
)

func TestParsePortsAndNew(t *testing.T) {
	tests := []struct {
		input            []string
		expected         []ForwardedPort
		expectParseError bool
		expectNewError   bool
	}{
		{input: []string{}, expectNewError: true},
		{input: []string{"a"}, expectParseError: true, expectNewError: true},
		{input: []string{":a"}, expectParseError: true, expectNewError: true},
		{input: []string{"-1"}, expectParseError: true, expectNewError: true},
		{input: []string{"65536"}, expectParseError: true, expectNewError: true},
		{input: []string{"0"}, expectParseError: true, expectNewError: true},
		{input: []string{"0:0"}, expectParseError: true, expectNewError: true},
		{input: []string{"a:5000"}, expectParseError: true, expectNewError: true},
		{input: []string{"5000:a"}, expectParseError: true, expectNewError: true},
		{
			input: []string{"5000", "5000:5000", "8888:5000", "5000:8888", ":5000", "0:5000"},
			expected: []ForwardedPort{
				{5000, 5000},
				{5000, 5000},
				{8888, 5000},
				{5000, 8888},
				{0, 5000},
				{0, 5000},
			},
		},
	}

	for i, test := range tests {
		parsed, err := parsePorts(test.input)
		haveError := err != nil
		if e, a := test.expectParseError, haveError; e != a {
			t.Fatalf("%d: parsePorts: error expected=%t, got %t: %s", i, e, a, err)
		}

		expectedRequest := &client.Request{}
		expectedConfig := &client.Config{}
		expectedStopChan := make(chan struct{})
		pf, err := New(expectedRequest, expectedConfig, test.input, expectedStopChan)
		haveError = err != nil
		if e, a := test.expectNewError, haveError; e != a {
			t.Fatalf("%d: New: error expected=%t, got %t: %s", i, e, a, err)
		}

		if test.expectParseError || test.expectNewError {
			continue
		}

		for pi, expectedPort := range test.expected {
			if e, a := expectedPort.Local, parsed[pi].Local; e != a {
				t.Fatalf("%d: local expected: %d, got: %d", i, e, a)
			}
			if e, a := expectedPort.Remote, parsed[pi].Remote; e != a {
				t.Fatalf("%d: remote expected: %d, got: %d", i, e, a)
			}
		}

		if e, a := expectedRequest, pf.req; e != a {
			t.Fatalf("%d: req: expected %#v, got %#v", i, e, a)
		}
		if e, a := expectedConfig, pf.config; e != a {
			t.Fatalf("%d: config: expected %#v, got %#v", i, e, a)
		}
		if e, a := test.expected, pf.ports; !reflect.DeepEqual(e, a) {
			t.Fatalf("%d: ports: expected %#v, got %#v", i, e, a)
		}
		if e, a := expectedStopChan, pf.stopChan; e != a {
			t.Fatalf("%d: stopChan: expected %#v, got %#v", i, e, a)
		}
		if pf.Ready == nil {
			t.Fatalf("%d: Ready should be non-nil", i)
		}
	}
}

type fakeUpgrader struct {
	conn *fakeUpgradeConnection
	err  error
}

func (u *fakeUpgrader) upgrade(req *client.Request, config *client.Config) (httpstream.Connection, error) {
	return u.conn, u.err
}

type fakeUpgradeConnection struct {
	closeCalled bool
	lock        sync.Mutex
	streams     map[string]*fakeUpgradeStream
	portData    map[string]string
}

func newFakeUpgradeConnection() *fakeUpgradeConnection {
	return &fakeUpgradeConnection{
		streams:  make(map[string]*fakeUpgradeStream),
		portData: make(map[string]string),
	}
}

func (c *fakeUpgradeConnection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	stream := &fakeUpgradeStream{}
	c.streams[headers.Get(api.PortHeader)] = stream
	// only simulate data on the data stream for now, not the error stream
	if headers.Get(api.StreamType) == api.StreamTypeData {
		stream.data = c.portData[headers.Get(api.PortHeader)]
	}

	return stream, nil
}

func (c *fakeUpgradeConnection) Close() error {
	c.lock.Lock()
	defer c.lock.Unlock()

	c.closeCalled = true
	return nil
}

func (c *fakeUpgradeConnection) CloseChan() <-chan bool {
	return make(chan bool)
}

func (c *fakeUpgradeConnection) SetIdleTimeout(timeout time.Duration) {
}

type fakeUpgradeStream struct {
	readCalled  bool
	writeCalled bool
	dataWritten []byte
	closeCalled bool
	resetCalled bool
	data        string
	lock        sync.Mutex
}

func (s *fakeUpgradeStream) Read(p []byte) (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.readCalled = true
	b := []byte(s.data)
	n := copy(p, b)
	return n, io.EOF
}

func (s *fakeUpgradeStream) Write(p []byte) (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.writeCalled = true
	s.dataWritten = make([]byte, len(p))
	copy(s.dataWritten, p)
	return len(p), io.EOF
}

func (s *fakeUpgradeStream) Close() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.closeCalled = true
	return nil
}

func (s *fakeUpgradeStream) Reset() error {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.resetCalled = true
	return nil
}

func (s *fakeUpgradeStream) Headers() http.Header {
	s.lock.Lock()
	defer s.lock.Unlock()
	return http.Header{}
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
			Hostname:                "[::1]",
			Protocol:                "tcp6",
			ShouldRaiseError:        false,
			ExpectedListenerAddress: "::1",
		},
		{
			Hostname:         "[::1]",
			Protocol:         "tcp4",
			ShouldRaiseError: true,
		},
		{
			Hostname:         "127.0.0.1",
			Protocol:         "tcp6",
			ShouldRaiseError: true,
		},
		{
			// IPv6 address must be put into brackets. This test reveals this.
			Hostname:         "::1",
			Protocol:         "tcp6",
			ShouldRaiseError: true,
		},
	}

	for i, testCase := range testCases {
		expectedListenerPort := "12345"
		listener, err := pf.getListener(testCase.Protocol, testCase.Hostname, &ForwardedPort{12345, 12345})
		errorRaised := err != nil

		if testCase.ShouldRaiseError != errorRaised {
			t.Errorf("Test case #%d failed: Data %v an error has been raised(%t) where it should not (or reciprocally): %v", i, testCase, testCase.ShouldRaiseError, err)
			continue
		}
		if errorRaised {
			continue
		}

		if listener == nil {
			t.Errorf("Test case #%d did not raised an error (%t) but failed in initializing listener", i, err)
			continue
		}

		host, port, _ := net.SplitHostPort(listener.Addr().String())
		t.Logf("Asked a %s forward for: %s:%v, got listener %s:%s, expected: %s", testCase.Protocol, testCase.Hostname, 12345, host, port, expectedListenerPort)
		if host != testCase.ExpectedListenerAddress {
			t.Errorf("Test case #%d failed: Listener does not listen on exepected address: asked %v got %v", i, testCase.ExpectedListenerAddress, host)
		}
		if port != expectedListenerPort {
			t.Errorf("Test case #%d failed: Listener does not listen on exepected port: asked %v got %v", i, expectedListenerPort, port)

		}
		listener.Close()

	}
}

func TestForwardPorts(t *testing.T) {
	testCases := []struct {
		Upgrader *fakeUpgrader
		Ports    []string
		Send     map[uint16]string
		Receive  map[uint16]string
		Err      bool
	}{
		{
			Upgrader: &fakeUpgrader{err: errors.New("bail")},
			Err:      true,
		},
		{
			Upgrader: &fakeUpgrader{conn: newFakeUpgradeConnection()},
			Ports:    []string{"5000"},
		},
		{
			Upgrader: &fakeUpgrader{conn: newFakeUpgradeConnection()},
			Ports:    []string{"5000", "6000"},
			Send: map[uint16]string{
				5000: "abcd",
				6000: "ghij",
			},
			Receive: map[uint16]string{
				5000: "1234",
				6000: "5678",
			},
		},
	}

	for i, testCase := range testCases {
		stopChan := make(chan struct{}, 1)

		pf, err := New(&client.Request{}, &client.Config{}, testCase.Ports, stopChan)
		hasErr := err != nil
		if hasErr != testCase.Err {
			t.Fatalf("%d: New: expected %t, got %t: %v", i, testCase.Err, hasErr, err)
		}
		if pf == nil {
			continue
		}
		pf.upgrader = testCase.Upgrader
		if testCase.Upgrader.err != nil {
			err := pf.ForwardPorts()
			hasErr := err != nil
			if hasErr != testCase.Err {
				t.Fatalf("%d: ForwardPorts: expected %t, got %t: %v", i, testCase.Err, hasErr, err)
			}
			continue
		}

		doneChan := make(chan error)
		go func() {
			doneChan <- pf.ForwardPorts()
		}()
		<-pf.Ready

		conn := testCase.Upgrader.conn

		for port, data := range testCase.Send {
			conn.lock.Lock()
			conn.portData[fmt.Sprintf("%d", port)] = testCase.Receive[port]
			conn.lock.Unlock()

			clientConn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", port))
			if err != nil {
				t.Fatalf("%d: error dialing %d: %s", i, port, err)
			}
			defer clientConn.Close()

			n, err := clientConn.Write([]byte(data))
			if err != nil && err != io.EOF {
				t.Fatalf("%d: Error sending data '%s': %s", i, data, err)
			}
			if n == 0 {
				t.Fatalf("%d: unexpected write of 0 bytes", i)
			}
			b := make([]byte, 4)
			n, err = clientConn.Read(b)
			if err != nil && err != io.EOF {
				t.Fatalf("%d: Error reading data: %s", i, err)
			}
			if !bytes.Equal([]byte(testCase.Receive[port]), b) {
				t.Fatalf("%d: expected to read '%s', got '%s'", i, testCase.Receive[port], b)
			}
		}

		// tell r.ForwardPorts to stop
		close(stopChan)

		// wait for r.ForwardPorts to actually return
		err = <-doneChan
		if err != nil {
			t.Fatalf("%d: unexpected error: %s", i, err)
		}

		if e, a := len(testCase.Send), len(conn.streams); e != a {
			t.Fatalf("%d: expected %d streams to be created, got %d", i, e, a)
		}

		if !conn.closeCalled {
			t.Fatalf("%d: expected conn closure", i)
		}
	}

}
