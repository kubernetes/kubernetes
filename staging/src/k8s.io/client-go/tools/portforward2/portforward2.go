/*
Copyright 2022 The Kubernetes Authors.

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

package portforward2

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// TODO: Replace portforward.go with this
// TODO: Create unit tests (adapt then from the existing unit tests)
// TODO: Update e2e tests

// PortForwardProtocolV1Name is the subprotocol used for port forwarding.
const PortForwardProtocolV1Name = "portforward.k8s.io"

// PortForwarder2 knows how to listen for local connections and forward them to a remote pod via an upgraded HTTP request.
type PortForwarder2 struct {
	dialer            httpstream.Dialer
	addresses         []string
	ports             []string
	reconnect         bool
	outWriter         io.Writer
	errWriter         io.Writer
	requestID         uint32
	activeConnections sync.WaitGroup
}

// New creates a new PortForwarder2 for localhost.
func New(dialer httpstream.Dialer, ports []string, outWriter io.Writer, errWriter io.Writer) *PortForwarder2 {
	return NewOnAddresses(dialer, []string{"localhost"}, ports, outWriter, errWriter)
}

// NewOnAddresses creates a new PortForwarder2 with custom listen addresses.
func NewOnAddresses(dialer httpstream.Dialer, addresses []string, ports []string, outWriter io.Writer, errWriter io.Writer) *PortForwarder2 {
	return &PortForwarder2{
		dialer:    dialer,
		addresses: addresses,
		ports:     ports,
		outWriter: outWriter,
		errWriter: errWriter,
	}
}

// ForwardPorts starts port forwarding and continues until the provided context is cancelled.
func (pf *PortForwarder2) ForwardPorts(ctx context.Context) error {
	// Keep track of listeners, so they can be closed later
	var listeners []net.Listener

	// Create a new context, with cancel, so that we can cancel the listeners to stop gracefully
	listenerContext, cancelListeners := context.WithCancel(ctx)

	// Stop all listeners and wait for all active connections to close
	defer func() {
		cancelListeners()
		for _, listener := range listeners {
			if err := listener.Close(); err != nil {
				pf.errorf("failed to close listener: %s", err) // Print the error, but don't fail during cleanup
			}
		}
		pf.activeConnections.Wait()
	}()

	// Create a listener for each listenAddress/port combination
	for _, a := range pf.addresses {
		for _, p := range pf.ports {
			listener, err := pf.listen(listenerContext, a, p)
			if err != nil {
				return err
			}
			listeners = append(listeners, listener)
		}
	}

	// Wait for context to be done
	<-listenerContext.Done()

	return nil
}

func (pf *PortForwarder2) listen(ctx context.Context, address string, port string) (net.Listener, error) {
	protocol, err := determineAddressProtocol(address)
	if err != nil {
		return nil, err
	}

	localPort, remotePort, err := parseLocalAndRemotePorts(port)
	if err != nil {
		return nil, err
	}

	addressWithPort := net.JoinHostPort(address, localPort)
	listener, err := net.Listen(protocol, addressWithPort)
	if err != nil {
		return nil, fmt.Errorf("listen failed on %s: %s", addressWithPort, err)
	}

	listenerAddress := listener.Addr().String()
	pf.printf("Forwarding from %s -> %s\n", listenerAddress, remotePort)

	// Accept and handle connections
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				select {
				case <-ctx.Done():
					return
				default:
					pf.errorf("accept failed: %s\n", err)
					continue
				}
			}
			go pf.handleConnection(ctx, conn, listenerAddress, remotePort)
		}
	}()

	return listener, nil
}

func (pf *PortForwarder2) handleConnection(ctx context.Context, localConn net.Conn, listenerAddress string, remotePort string) {
	defer localConn.Close()

	// Increment the active connections WaitGroup, so if we need to stop we can wait for all in-flight connections to complete
	pf.activeConnections.Add(1)
	defer pf.activeConnections.Done()

	// Each connection handled gets a unique, sequential request ID
	requestID := atomic.AddUint32(&pf.requestID, 1)

	// Print an informational message at the start and at the finish of each handled connection
	pf.printf("[%d] Started handling connection forwarding from %s -> %s\n", requestID, listenerAddress, remotePort)
	defer func() {
		pf.printf("[%d] Finished handling connection forwarding from %s -> %s\n", requestID, listenerAddress, remotePort)
	}()

	// Helper function for displaying error messages
	handleError := func(message string, err error) {
		if err == nil {
			pf.errorf("[%d] ERROR: %s\n", requestID, message)
		} else {
			pf.errorf("[%d] ERROR: %s: %s\n", requestID, message, err)
		}
	}

	// Establish the remote connection
	remoteConnection, _, err := pf.dialer.Dial(PortForwardProtocolV1Name)
	if err != nil {
		handleError("dial failed", err)
		return
	}
	defer remoteConnection.Close()

	// Create an error stream for the remote connection.  Immediately close it, because we will only read from it
	remoteErrorStream, err := pf.createStream(remoteConnection, v1.StreamTypeError, remotePort, requestID)
	if err != nil {
		handleError("failed to create error stream", err)
		return
	}
	remoteErrorStream.Close()

	// Create a data stream for the remote connection
	remoteDataStream, err := pf.createStream(remoteConnection, v1.StreamTypeData, remotePort, requestID)
	if err != nil {
		handleError("failed to create data stream", err)
		return
	}

	// Read everything from the error stream
	errorChan := make(chan struct{})
	go func() {
		defer close(errorChan)
		message, err := ioutil.ReadAll(remoteErrorStream)
		if err != nil {
			handleError("failed to read from error stream", err)
		} else if len(message) > 0 {
			handleError(string(message), nil)
		}
	}()

	// Read from remote to local
	doneChan := make(chan struct{})
	go func() {
		defer close(doneChan)
		bytesRead, err := io.Copy(localConn, remoteDataStream)
		if err != nil {
			handleError("failed to read from remote data stream", err)
		}
		klog.V(3).Infof("bytes read = %d", bytesRead)
	}()

	// Write from local to remote
	go func() {
		defer remoteDataStream.Close()
		bytesWritten, err := io.Copy(remoteDataStream, localConn)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			handleError("failed to write to remote data stream", err)
		}
		klog.V(3).Infof("bytes written = %d", bytesWritten)
	}()

	select {
	case <-ctx.Done():
		remoteDataStream.Reset() // Reset data stream to unblock io.Copy
	case <-doneChan:
	}

	// Always wait for errorChan
	<-errorChan
}

func (pf *PortForwarder2) createStream(connection httpstream.Connection, streamType string, port string, requestID uint32) (httpstream.Stream, error) {
	headers := http.Header{}
	headers.Set(v1.StreamType, streamType)
	headers.Set(v1.PortHeader, port)
	headers.Set(v1.PortForwardRequestIDHeader, fmt.Sprintf("%d", requestID))
	return connection.CreateStream(headers)
}

func (pf *PortForwarder2) printf(format string, a ...interface{}) {
	if pf.outWriter != nil {
		fmt.Fprintf(pf.outWriter, format, a...)
	}
}

func (pf *PortForwarder2) errorf(format string, a ...interface{}) {
	if pf.errWriter != nil {
		fmt.Fprintf(pf.outWriter, format, a...)
	}
}

func determineAddressProtocol(address string) (string, error) {
	if netutils.ParseIPSloppy(address).To4() != nil {
		return "tcp4", nil
	}
	if netutils.ParseIPSloppy(address) != nil {
		return "tcp6", nil
	}
	ips, err := net.LookupIP(address)
	if err != nil {
		return "", err
	}
	for _, ip := range ips {
		if netutils.IsIPv4(ip) {
			return "tcp4", nil
		}
		if netutils.IsIPv6(ip) {
			return "tcp6", nil
		}
	}
	return "", fmt.Errorf("unable to determine address protocol for %q", address)
}

func parseLocalAndRemotePorts(portString string) (string, string, error) {
	// Split the local and remote ports and handle if only a single port is specified
	parts := strings.Split(portString, ":")
	var localString, remoteString string
	if len(parts) == 1 {
		localString = parts[0]
		remoteString = parts[0]
	} else if len(parts) == 2 {
		localString = parts[0]
		if localString == "" {
			// support :5000
			localString = "0"
		}
		remoteString = parts[1]
	} else {
		return "", "", fmt.Errorf("invalid port format")
	}

	// Make sure the local port can be parsed as a 16-bit unsigned int
	localPort, err := strconv.ParseUint(localString, 10, 16)
	if err != nil {
		return "", "", fmt.Errorf("error parsing local port %q: %s", localString, err)
	}

	// Make sure the remote port can be parsed as a 16-bit unsigned int and is not 0
	remotePort, err := strconv.ParseUint(remoteString, 10, 16)
	if err != nil {
		return "", "", fmt.Errorf("error parsing remote port %q: %s", remoteString, err)
	}
	if remotePort == 0 {
		return "", "", fmt.Errorf("remote port must be > 0")
	}

	return strconv.FormatUint(localPort, 10), strconv.FormatUint(remotePort, 10), nil
}
