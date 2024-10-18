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
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"reflect"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
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

type fakeConnection struct {
	closed      bool
	closeChan   chan bool
	dataStream  *fakeStream
	errorStream *fakeStream
	streamCount int
}

func newFakeConnection() *fakeConnection {
	return &fakeConnection{
		closeChan:   make(chan bool),
		dataStream:  &fakeStream{},
		errorStream: &fakeStream{},
	}
}

func (c *fakeConnection) CreateStream(headers http.Header) (httpstream.Stream, error) {
	switch headers.Get(v1.StreamType) {
	case v1.StreamTypeData:
		c.streamCount++
		return c.dataStream, nil
	case v1.StreamTypeError:
		c.streamCount++
		return c.errorStream, nil
	default:
		return nil, fmt.Errorf("fakeStream creation not supported for stream type %s", headers.Get(v1.StreamType))
	}
}

func (c *fakeConnection) Close() error {
	if !c.closed {
		c.closed = true
		close(c.closeChan)
	}
	return nil
}

func (c *fakeConnection) CloseChan() <-chan bool {
	return c.closeChan
}

func (c *fakeConnection) RemoveStreams(streams ...httpstream.Stream) {
	for range streams {
		c.streamCount--
	}
}

func (c *fakeConnection) SetIdleTimeout(timeout time.Duration) {
	// no-op
}

type fakeListener struct {
	net.Listener
	closeChan chan bool
}

func newFakeListener() fakeListener {
	return fakeListener{
		closeChan: make(chan bool),
	}
}

func (l *fakeListener) Accept() (net.Conn, error) {
	select {
	case <-l.closeChan:
		return nil, fmt.Errorf("listener closed")
	}
}

func (l *fakeListener) Close() error {
	close(l.closeChan)
	return nil
}

func (l *fakeListener) Addr() net.Addr {
	return fakeAddr{}
}

type fakeAddr struct{}

func (fakeAddr) Network() string { return "fake" }
func (fakeAddr) String() string  { return "fake" }

type fakeStream struct {
	headers   http.Header
	readFunc  func(p []byte) (int, error)
	writeFunc func(p []byte) (int, error)
}

func (s *fakeStream) Read(p []byte) (n int, err error)  { return s.readFunc(p) }
func (s *fakeStream) Write(p []byte) (n int, err error) { return s.writeFunc(p) }
func (*fakeStream) Close() error                        { return nil }
func (*fakeStream) Reset() error                        { return nil }
func (s *fakeStream) Headers() http.Header              { return s.headers }
func (*fakeStream) Identifier() uint32                  { return 0 }

type fakeConn struct {
	sendBuffer    *bytes.Buffer
	receiveBuffer *bytes.Buffer
}

func (f fakeConn) Read(p []byte) (int, error)       { return f.sendBuffer.Read(p) }
func (f fakeConn) Write(p []byte) (int, error)      { return f.receiveBuffer.Write(p) }
func (fakeConn) Close() error                       { return nil }
func (fakeConn) LocalAddr() net.Addr                { return nil }
func (fakeConn) RemoteAddr() net.Addr               { return nil }
func (fakeConn) SetDeadline(t time.Time) error      { return nil }
func (fakeConn) SetReadDeadline(t time.Time) error  { return nil }
func (fakeConn) SetWriteDeadline(t time.Time) error { return nil }

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
			input:     []string{"5000:5000"},
			addresses: []string{"localhost", "::1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "all"},
				{protocol: "tcp6", address: "::1", failureMode: "any"},
			},
		},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"localhost", "127.0.0.1", "::1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
				{protocol: "tcp6", address: "::1", failureMode: "any"},
			},
		},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"localhost", "127.0.0.1", "10.10.10.1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
				{protocol: "tcp6", address: "::1", failureMode: "all"},
				{protocol: "tcp4", address: "10.10.10.1", failureMode: "any"},
			},
		},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"127.0.0.1", "::1", "localhost"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
				{protocol: "tcp6", address: "::1", failureMode: "any"},
			},
		},
		{
			input:     []string{"5000:5000"},
			addresses: []string{"10.0.0.1", "127.0.0.1"},
			expectedPorts: []ForwardedPort{
				{5000, 5000},
			},
			expectedAddresses: []listenAddress{
				{protocol: "tcp4", address: "10.0.0.1", failureMode: "any"},
				{protocol: "tcp4", address: "127.0.0.1", failureMode: "any"},
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
		forwardedPort := &ForwardedPort{Local: 0, Remote: 12345}
		listener, err := pf.getListener(testCase.Protocol, testCase.Hostname, forwardedPort)
		if err != nil && strings.Contains(err.Error(), "cannot assign requested address") {
			t.Logf("Can't test #%d: %v", i, err)
			continue
		}
		expectedListenerPort := fmt.Sprintf("%d", forwardedPort.Local)
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
		t.Logf("Asked a %s forward for: %s:0, got listener %s:%s, expected: %s", testCase.Protocol, testCase.Hostname, host, port, expectedListenerPort)
		if host != testCase.ExpectedListenerAddress {
			t.Errorf("Test case #%d failed: Listener does not listen on expected address: asked '%v' got '%v'", i, testCase.ExpectedListenerAddress, host)
		}
		if port != expectedListenerPort {
			t.Errorf("Test case #%d failed: Listener does not listen on expected port: asked %v got %v", i, expectedListenerPort, port)

		}
		listener.Close()
	}
}

func TestGetPortsReturnsDynamicallyAssignedLocalPort(t *testing.T) {
	dialer := &fakeDialer{
		conn:               newFakeConnection(),
		negotiatedProtocol: PortForwardProtocolV1Name,
	}

	stopChan := make(chan struct{})
	readyChan := make(chan struct{})
	errChan := make(chan error)

	defer func() {
		close(stopChan)

		forwardErr := <-errChan
		if forwardErr != nil {
			t.Fatalf("ForwardPorts returned error: %s", forwardErr)
		}
	}()

	pf, err := New(dialer, []string{":5000"}, stopChan, readyChan, os.Stdout, os.Stderr)

	if err != nil {
		t.Fatalf("error while calling New: %s", err)
	}

	go func() {
		errChan <- pf.ForwardPorts()
		close(errChan)
	}()

	<-pf.Ready

	ports, err := pf.GetPorts()
	if err != nil {
		t.Fatalf("Failed to get ports. error: %v", err)
	}

	if len(ports) != 1 {
		t.Fatalf("expected 1 port, got %d", len(ports))
	}

	port := ports[0]
	if port.Local == 0 {
		t.Fatalf("local port is 0, expected != 0")
	}
}

func TestHandleConnection(t *testing.T) {
	out := bytes.NewBufferString("")

	pf, err := New(&fakeDialer{}, []string{":2222"}, nil, nil, out, nil)
	if err != nil {
		t.Fatalf("error while calling New: %s", err)
	}

	// Setup fake local connection
	localConnection := &fakeConn{
		sendBuffer:    bytes.NewBufferString("test data from local"),
		receiveBuffer: bytes.NewBufferString(""),
	}

	// Setup fake remote connection to send data on the data stream after it receives data from the local connection
	remoteDataToSend := bytes.NewBufferString("test data from remote")
	remoteDataReceived := bytes.NewBufferString("")
	remoteErrorToSend := bytes.NewBufferString("")
	blockRemoteSend := make(chan struct{})
	remoteConnection := newFakeConnection()
	remoteConnection.dataStream.readFunc = func(p []byte) (int, error) {
		<-blockRemoteSend // Wait for the expected data to be received before responding
		return remoteDataToSend.Read(p)
	}
	remoteConnection.dataStream.writeFunc = func(p []byte) (int, error) {
		n, err := remoteDataReceived.Write(p)
		if remoteDataReceived.String() == "test data from local" {
			close(blockRemoteSend)
		}
		return n, err
	}
	remoteConnection.errorStream.readFunc = remoteErrorToSend.Read
	pf.streamConn = remoteConnection

	// Test handleConnection
	pf.handleConnection(localConnection, ForwardedPort{Local: 1111, Remote: 2222})
	assert.Equal(t, 0, remoteConnection.streamCount, "stream count should be zero")
	assert.Equal(t, "test data from local", remoteDataReceived.String())
	assert.Equal(t, "test data from remote", localConnection.receiveBuffer.String())
	assert.Equal(t, "Handling connection for 1111\n", out.String())
}

func TestHandleConnectionSendsRemoteError(t *testing.T) {
	out := bytes.NewBufferString("")
	errOut := bytes.NewBufferString("")

	pf, err := New(&fakeDialer{}, []string{":2222"}, nil, nil, out, errOut)
	if err != nil {
		t.Fatalf("error while calling New: %s", err)
	}

	// Setup fake local connection
	localConnection := &fakeConn{
		sendBuffer:    bytes.NewBufferString(""),
		receiveBuffer: bytes.NewBufferString(""),
	}

	// Setup fake remote connection to return an error message on the error stream
	remoteDataToSend := bytes.NewBufferString("")
	remoteDataReceived := bytes.NewBufferString("")
	remoteErrorToSend := bytes.NewBufferString("error")
	remoteConnection := newFakeConnection()
	remoteConnection.dataStream.readFunc = remoteDataToSend.Read
	remoteConnection.dataStream.writeFunc = remoteDataReceived.Write
	remoteConnection.errorStream.readFunc = remoteErrorToSend.Read
	pf.streamConn = remoteConnection

	// Test handleConnection, using go-routine because it needs to be able to write to unbuffered pf.errorChan
	pf.handleConnection(localConnection, ForwardedPort{Local: 1111, Remote: 2222})

	assert.Equal(t, 0, remoteConnection.streamCount, "stream count should be zero")
	assert.Equal(t, "", remoteDataReceived.String())
	assert.Equal(t, "", localConnection.receiveBuffer.String())
	assert.Equal(t, "Handling connection for 1111\n", out.String())
}

func TestWaitForConnectionExitsOnStreamConnClosed(t *testing.T) {
	out := bytes.NewBufferString("")
	errOut := bytes.NewBufferString("")

	pf, err := New(&fakeDialer{}, []string{":2222"}, nil, nil, out, errOut)
	if err != nil {
		t.Fatalf("error while calling New: %s", err)
	}

	listener := newFakeListener()

	pf.streamConn = newFakeConnection()
	pf.streamConn.Close()

	port := ForwardedPort{}
	pf.waitForConnection(&listener, port)
}

func TestForwardPortsReturnsErrorWhenConnectionIsLost(t *testing.T) {
	dialer := &fakeDialer{
		conn:               newFakeConnection(),
		negotiatedProtocol: PortForwardProtocolV1Name,
	}

	stopChan := make(chan struct{})
	readyChan := make(chan struct{})
	errChan := make(chan error)

	pf, err := New(dialer, []string{":5000"}, stopChan, readyChan, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("failed to create new PortForwarder: %s", err)
	}

	go func() {
		errChan <- pf.ForwardPorts()
	}()

	<-pf.Ready

	// Simulate lost pod connection by closing streamConn, which should result in pf.ForwardPorts() returning an error.
	pf.streamConn.Close()

	err = <-errChan
	if err == nil {
		t.Fatalf("unexpected non-error from pf.ForwardPorts()")
	} else if err != ErrLostConnectionToPod {
		t.Fatalf("unexpected error from pf.ForwardPorts(): %s", err)
	}
}

func TestForwardPortsReturnsNilWhenStopChanIsClosed(t *testing.T) {
	dialer := &fakeDialer{
		conn:               newFakeConnection(),
		negotiatedProtocol: PortForwardProtocolV1Name,
	}

	stopChan := make(chan struct{})
	readyChan := make(chan struct{})
	errChan := make(chan error)

	pf, err := New(dialer, []string{":5000"}, stopChan, readyChan, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("failed to create new PortForwarder: %s", err)
	}

	go func() {
		errChan <- pf.ForwardPorts()
	}()

	<-pf.Ready

	// Closing (or sending to) stopChan indicates a stop request by the caller, which should result in pf.ForwardPorts()
	// returning nil.
	close(stopChan)

	err = <-errChan
	if err != nil {
		t.Fatalf("unexpected error from pf.ForwardPorts(): %s", err)
	}
}

func TestHandleRemoteSendError(t *testing.T) {
	type portForwardErrResponse struct {
		// server side's suggestions
		// guide us whether to close the connection
		CloseConnection bool `json:"closeConnection,omitempty"`
		// error detail
		Message string `json:"message,omitempty"`
	}

	testCases := []struct {
		resp portForwardErrResponse
	}{
		{
			resp: portForwardErrResponse{
				CloseConnection: false,
			},
		},
		{
			resp: portForwardErrResponse{
				CloseConnection: true,
			},
		},
		{
			resp: portForwardErrResponse{
				CloseConnection: false,
				Message:         syscall.EPIPE.Error(),
			},
		},
		{
			resp: portForwardErrResponse{
				CloseConnection: false,
				Message:         syscall.ECONNRESET.Error(),
			},
		},
		{
			resp: portForwardErrResponse{
				CloseConnection: true,
				Message:         "got unexpect err",
			},
		},
	}

	for _, testCase := range testCases {
		t.Run("", func(t *testing.T) {
			out := bytes.NewBufferString("")
			errOut := bytes.NewBufferString("")

			errStream := &bytes.Buffer{}
			if err := json.NewEncoder(errStream).Encode(testCase.resp); err != nil {
				t.Fatalf("encode the err response: %v", err)
			}

			pf, err := New(&fakeDialer{}, []string{":2222"}, nil, nil, out, errOut)
			if err != nil {
				t.Fatalf("error while calling New: %s", err)
			}

			// Setup fake local connection
			localConnection := &fakeConn{
				sendBuffer:    bytes.NewBufferString(""),
				receiveBuffer: bytes.NewBufferString(""),
			}

			// Setup fake remote connection to return an error message on the error stream
			remoteDataToSend := bytes.NewBufferString("")
			remoteDataReceived := bytes.NewBufferString("")
			remoteConnection := newFakeConnection()
			remoteConnection.dataStream.readFunc = remoteDataToSend.Read
			remoteConnection.dataStream.writeFunc = remoteDataReceived.Write
			remoteConnection.errorStream.readFunc = errStream.Read
			pf.streamConn = remoteConnection

			// Test handleConnection, using go-routine because it needs to be able to write to unbuffered pf.errorChan
			pf.handleConnection(localConnection, ForwardedPort{Local: 1111, Remote: 2222})

			underlayConn := pf.streamConn.(*fakeConnection)
			if testCase.resp.CloseConnection != underlayConn.closed {
				t.Fatalf("want %v but got %v", testCase.resp.CloseConnection, underlayConn.closed)
			}
		})
	}
}
