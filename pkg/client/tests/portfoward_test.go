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

package tests

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	restclient "k8s.io/client-go/rest"
	. "k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
	"k8s.io/kubelet/pkg/cri/streaming/portforward"
)

// fakePortForwarder simulates port forwarding for testing. It implements
// portforward.PortForwarder.
type fakePortForwarder struct {
	lock sync.Mutex
	// stores data expected from the stream per port
	expected map[int32]string
	// stores data received from the stream per port
	received map[int32]string
	// data to be sent to the stream per port
	send map[int32]string
}

var _ portforward.PortForwarder = &fakePortForwarder{}

func (pf *fakePortForwarder) PortForward(_ context.Context, name string, uid types.UID, port int32, stream io.ReadWriteCloser) error {
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
func fakePortForwardServer(t *testing.T, testName string, serverSends, expectedFromClient map[int32]string) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		pf := &fakePortForwarder{
			expected: expectedFromClient,
			received: make(map[int32]string),
			send:     serverSends,
		}
		portforward.ServePortForward(w, req, pf, "pod", "uid", nil, 0, 10*time.Second, portforward.SupportedProtocols)

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
		clientSends map[int32]string
		serverSends map[int32]string
	}{
		"forward 1 port with no data either direction": {
			ports: []string{":5000"},
		},
		"forward 2 ports with bidirectional data": {
			ports: []string{":5001", ":6000"},
			clientSends: map[int32]string{
				5001: "abcd",
				6000: "ghij",
			},
			serverSends: map[int32]string{
				5001: "1234",
				6000: "5678",
			},
		},
	}

	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			server := httptest.NewServer(fakePortForwardServer(t, testName, test.serverSends, test.clientSends))
			defer server.Close()

			transport, upgrader, err := spdy.RoundTripperFor(&restclient.Config{})
			if err != nil {
				t.Fatal(err)
			}
			url, _ := url.Parse(server.URL)
			dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", url)

			stopChan := make(chan struct{}, 1)
			readyChan := make(chan struct{})

			pf, err := New(dialer, test.ports, stopChan, readyChan, os.Stdout, os.Stderr)
			if err != nil {
				t.Fatalf("%s: unexpected error calling New: %v", testName, err)
			}

			doneChan := make(chan error)
			go func() {
				doneChan <- pf.ForwardPorts()
			}()
			<-pf.Ready

			forwardedPorts, err := pf.GetPorts()
			if err != nil {
				t.Fatal(err)
			}

			remoteToLocalMap := map[int32]int32{}
			for _, forwardedPort := range forwardedPorts {
				remoteToLocalMap[int32(forwardedPort.Remote)] = int32(forwardedPort.Local)
			}

			clientSend := func(port int32, data string) error {
				clientConn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", remoteToLocalMap[port]))
				if err != nil {
					return fmt.Errorf("%s: error dialing %d: %s", testName, port, err)

				}
				defer clientConn.Close()

				n, err := clientConn.Write([]byte(data))
				if err != nil && err != io.EOF {
					return fmt.Errorf("%s: Error sending data '%s': %s", testName, data, err)
				}
				if n == 0 {
					return fmt.Errorf("%s: unexpected write of 0 bytes", testName)
				}
				b := make([]byte, 4)
				_, err = clientConn.Read(b)
				if err != nil && err != io.EOF {
					return fmt.Errorf("%s: Error reading data: %s", testName, err)
				}
				if !bytes.Equal([]byte(test.serverSends[port]), b) {
					return fmt.Errorf("%s: expected to read '%s', got '%s'", testName, test.serverSends[port], b)
				}
				return nil
			}
			for port, data := range test.clientSends {
				if err := clientSend(port, data); err != nil {
					t.Error(err)
				}
			}
			// tell r.ForwardPorts to stop
			close(stopChan)

			// wait for r.ForwardPorts to actually return
			err = <-doneChan
			if err != nil {
				t.Errorf("%s: unexpected error: %s", testName, err)
			}
		})
	}

}

func TestForwardPortsReturnsErrorWhenAllBindsFailed(t *testing.T) {
	server := httptest.NewServer(fakePortForwardServer(t, "allBindsFailed", nil, nil))
	defer server.Close()

	transport, upgrader, err := spdy.RoundTripperFor(&restclient.Config{})
	if err != nil {
		t.Fatal(err)
	}
	url, _ := url.Parse(server.URL)
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", url)

	stopChan1 := make(chan struct{}, 1)
	defer close(stopChan1)
	readyChan1 := make(chan struct{})

	pf1, err := New(dialer, []string{":5555"}, stopChan1, readyChan1, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("error creating pf1: %v", err)
	}
	go pf1.ForwardPorts()
	<-pf1.Ready

	forwardedPorts, err := pf1.GetPorts()
	if err != nil {
		t.Fatal(err)
	}
	if len(forwardedPorts) != 1 {
		t.Fatalf("expected 1 forwarded port, got %#v", forwardedPorts)
	}
	duplicateSpec := fmt.Sprintf("%d:%d", forwardedPorts[0].Local, forwardedPorts[0].Remote)

	stopChan2 := make(chan struct{}, 1)
	readyChan2 := make(chan struct{})
	pf2, err := New(dialer, []string{duplicateSpec}, stopChan2, readyChan2, os.Stdout, os.Stderr)
	if err != nil {
		t.Fatalf("error creating pf2: %v", err)
	}
	if err := pf2.ForwardPorts(); err == nil {
		t.Fatal("expected non-nil error for pf2.ForwardPorts")
	}
}
