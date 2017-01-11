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
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	"k8s.io/kubernetes/pkg/util/httpstream"
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

		dialer := &fakeDialer{}
		expectedStopChan := make(chan struct{})
		readyChan := make(chan struct{})
		pf, err := New(dialer, test.input, expectedStopChan, readyChan, os.Stdout, os.Stderr)
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

		if dialer.dialed {
			t.Fatalf("%d: expected not dialed", i)
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
			t.Errorf("Test case #%d failed: Listener does not listen on exepected address: asked %v got %v", i, testCase.ExpectedListenerAddress, host)
		}
		if port != expectedListenerPort {
			t.Errorf("Test case #%d failed: Listener does not listen on exepected port: asked %v got %v", i, expectedListenerPort, port)

		}
		listener.Close()

	}
}

// fakePortForwarder simulates port forwarding for testing. It implements
// portforward.PortForwarder.
type fakePortForwarder struct {
	lock sync.Mutex
	// stores data expected from the stream per port
	expected map[uint16]string
	// stores data received from the stream per port
	received map[uint16]string
	// data to be sent to the stream per port
	send map[uint16]string
}

var _ portforward.PortForwarder = &fakePortForwarder{}

func (pf *fakePortForwarder) PortForward(name string, uid types.UID, port uint16, stream io.ReadWriteCloser) error {
	defer stream.Close()

	// read from the client
	received := make([]byte, len(pf.expected[port]))
	n, err := stream.Read(received)
	if err != nil {
		return fmt.Errorf("error reading from client for port %d: %v", port, err)
	}
	if n != len(pf.expected[port]) {
		return fmt.Errorf("unexpected length read from client for port %d: got %d, expected %d. data=%q", port, n, len(pf.expected[port]), string(received))
	}

	// store the received content
	pf.lock.Lock()
	pf.received[port] = string(received)
	pf.lock.Unlock()

	// send the hardcoded data to the client
	io.Copy(stream, strings.NewReader(pf.send[port]))

	return nil
}

// fakePortForwardServer creates an HTTP server that can handle port forwarding
// requests.
func fakePortForwardServer(t *testing.T, testName string, serverSends, expectedFromClient map[uint16]string) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		pf := &fakePortForwarder{
			expected: expectedFromClient,
			received: make(map[uint16]string),
			send:     serverSends,
		}
		portforward.ServePortForward(w, req, pf, "pod", "uid", 0, 10*time.Second)

		for port, expected := range expectedFromClient {
			actual, ok := pf.received[port]
			if !ok {
				t.Errorf("%s: server didn't receive any data for port %d", testName, port)
				continue
			}

			if expected != actual {
				t.Errorf("%s: server expected to receive %q, got %q for port %d", testName, expected, actual, port)
			}
		}

		for port, actual := range pf.received {
			if _, ok := expectedFromClient[port]; !ok {
				t.Errorf("%s: server unexpectedly received %q for port %d", testName, actual, port)
			}
		}
	})
}

func TestForwardPorts(t *testing.T) {
	tests := map[string]struct {
		ports       []string
		clientSends map[uint16]string
		serverSends map[uint16]string
	}{
		"forward 1 port with no data either direction": {
			ports: []string{"5000"},
		},
		"forward 2 ports with bidirectional data": {
			ports: []string{"5001", "6000"},
			clientSends: map[uint16]string{
				5001: "abcd",
				6000: "ghij",
			},
			serverSends: map[uint16]string{
				5001: "1234",
				6000: "5678",
			},
		},
	}

	for testName, test := range tests {
		server := httptest.NewServer(fakePortForwardServer(t, testName, test.serverSends, test.clientSends))

		url, _ := url.Parse(server.URL)
		exec, err := remotecommand.NewExecutor(&restclient.Config{}, "POST", url)
		if err != nil {
			t.Fatal(err)
		}

		stopChan := make(chan struct{}, 1)
		readyChan := make(chan struct{})

		pf, err := New(exec, test.ports, stopChan, readyChan, os.Stdout, os.Stderr)
		if err != nil {
			t.Fatalf("%s: unexpected error calling New: %v", testName, err)
		}

		doneChan := make(chan error)
		go func() {
			doneChan <- pf.ForwardPorts()
		}()
		<-pf.Ready

		for port, data := range test.clientSends {
			clientConn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", port))
			if err != nil {
				t.Errorf("%s: error dialing %d: %s", testName, port, err)
				server.Close()
				continue
			}
			defer clientConn.Close()

			n, err := clientConn.Write([]byte(data))
			if err != nil && err != io.EOF {
				t.Errorf("%s: Error sending data '%s': %s", testName, data, err)
				server.Close()
				continue
			}
			if n == 0 {
				t.Errorf("%s: unexpected write of 0 bytes", testName)
				server.Close()
				continue
			}
			b := make([]byte, 4)
			n, err = clientConn.Read(b)
			if err != nil && err != io.EOF {
				t.Errorf("%s: Error reading data: %s", testName, err)
				server.Close()
				continue
			}
			if !bytes.Equal([]byte(test.serverSends[port]), b) {
				t.Errorf("%s: expected to read '%s', got '%s'", testName, test.serverSends[port], b)
				server.Close()
				continue
			}
		}
		// tell r.ForwardPorts to stop
		close(stopChan)

		// wait for r.ForwardPorts to actually return
		err = <-doneChan
		if err != nil {
			t.Errorf("%s: unexpected error: %s", testName, err)
		}
		server.Close()
	}

}

func TestForwardPortsReturnsErrorWhenAllBindsFailed(t *testing.T) {
	server := httptest.NewServer(fakePortForwardServer(t, "allBindsFailed", nil, nil))
	defer server.Close()

	url, _ := url.Parse(server.URL)
	exec, err := remotecommand.NewExecutor(&restclient.Config{}, "POST", url)
	if err != nil {
		t.Fatal(err)
	}

	stopChan1 := make(chan struct{}, 1)
	defer close(stopChan1)
	readyChan1 := make(chan struct{})

	pf1, err := New(exec, []string{"5555"}, stopChan1, readyChan1, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("error creating pf1: %v", err)
	}
	go pf1.ForwardPorts()
	<-pf1.Ready

	stopChan2 := make(chan struct{}, 1)
	readyChan2 := make(chan struct{})
	pf2, err := New(exec, []string{"5555"}, stopChan2, readyChan2, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("error creating pf2: %v", err)
	}
	if err := pf2.ForwardPorts(); err == nil {
		t.Fatal("expected non-nil error for pf2.ForwardPorts")
	}
}
