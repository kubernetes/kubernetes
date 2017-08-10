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

package ws

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"
	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/util/wsstream"
)

const protocolName = "v4." + wsstream.ChannelWebSocketProtocol

// ForwardedPort contains a Local:Remote port pairing.
type ForwardedPort struct {
	Local  uint16
	Remote uint16
}

// PortForwarder knows how to listen for local connections and forward them to
// a remote pod via an upgraded HTTP request.
type PortForwarder struct {
	ports    []ForwardedPort
	stopChan <-chan struct{}

	dialer DialFunc
	out    io.Writer
	errOut io.Writer
	Ready  chan struct{}

	maxConnections int

	lock              sync.Mutex
	activeConnections int
	closed            bool
	connections       []*websocket.Conn
	listeners         []io.Closer
}

// DialFunc will establish a websocket connection using the requested protocols for the provided port or return
// an error.
type DialFunc func(port uint16, subprotocols ...string) (*websocket.Conn, error)

// New creates a new PortForwarder.
func New(dialer DialFunc, ports []string, maxConnections int, stopChan <-chan struct{}, readyChan chan struct{}, out, errOut io.Writer) (*PortForwarder, error) {
	if len(ports) == 0 {
		return nil, errors.New("You must specify at least 1 port")
	}
	parsedPorts, err := parsePorts(ports)
	if err != nil {
		return nil, err
	}
	return &PortForwarder{
		dialer:         dialer,
		ports:          parsedPorts,
		stopChan:       stopChan,
		Ready:          readyChan,
		out:            out,
		errOut:         errOut,
		maxConnections: maxConnections,
	}, nil
}

// ForwardPorts formats and executes a port forwarding request. The connection will remain
// open until stopChan is closed.
func (pf *PortForwarder) ForwardPorts() error {
	defer pf.Close()

	return pf.forward()
}

// forward dials the remote host specific in req, upgrades the request, starts
// listeners for each port specified in ports, and forwards local connections
// to the remote host via streams.
func (pf *PortForwarder) forward() error {
	var err error

	listenSuccess := false
	for _, port := range pf.ports {
		err = pf.listenOnPort(&port)
		switch {
		case err == nil:
			listenSuccess = true
		default:
			if pf.errOut != nil {
				fmt.Fprintf(pf.errOut, "Unable to listen on port %d: %v\n", port.Local, err)
			}
		}
	}

	if !listenSuccess {
		return fmt.Errorf("Unable to listen on any of the requested ports: %v", pf.ports)
	}

	if pf.Ready != nil {
		close(pf.Ready)
	}

	// wait for interrupt or conn closure
	select {
	case <-pf.stopChan:
		runtime.HandleError(errors.New("lost connection to pod"))
	}

	return nil
}

// listenOnPort delegates tcp4 and tcp6 listener creation and waits for connections on both of these addresses.
// If both listener creation fail, an error is raised.
func (pf *PortForwarder) listenOnPort(port *ForwardedPort) error {
	errTcp4 := pf.listenOnPortAndAddress(port, "tcp4", "127.0.0.1")
	errTcp6 := pf.listenOnPortAndAddress(port, "tcp6", "[::1]")
	if errTcp4 != nil && errTcp6 != nil {
		return fmt.Errorf("All listeners failed to create with the following errors: %s, %s", errTcp4, errTcp6)
	}
	return nil
}

// listenOnPortAndAddress delegates listener creation and waits for new connections
// in the background f
func (pf *PortForwarder) listenOnPortAndAddress(port *ForwardedPort, protocol string, address string) error {
	listener, err := pf.getListener(protocol, address, port)
	if err != nil {
		return err
	}
	pf.listeners = append(pf.listeners, listener)
	go pf.waitForConnection(listener, *port)
	return nil
}

// getListener creates a listener on the interface targeted by the given hostname on the given port with
// the given protocol. protocol is in net.Listen style which basically admits values like tcp, tcp4, tcp6
func (pf *PortForwarder) getListener(protocol string, hostname string, port *ForwardedPort) (net.Listener, error) {
	listener, err := net.Listen(protocol, net.JoinHostPort(hostname, strconv.Itoa(int(port.Local))))
	if err != nil {
		return nil, fmt.Errorf("Unable to create listener: Error %s", err)
	}
	listenerAddress := listener.Addr().String()
	host, localPort, _ := net.SplitHostPort(listenerAddress)
	localPortUInt, err := strconv.ParseUint(localPort, 10, 16)

	if err != nil {
		return nil, fmt.Errorf("Error parsing local port: %s from %s (%s)", err, listenerAddress, host)
	}
	port.Local = uint16(localPortUInt)
	if pf.out != nil {
		fmt.Fprintf(pf.out, "Forwarding from %s:%d -> %d\n", hostname, localPortUInt, port.Remote)
	}

	return listener, nil
}

// waitForConnection waits for new connections to listener and handles them in
// the background.
func (pf *PortForwarder) waitForConnection(listener net.Listener, port ForwardedPort) {
	for {
		conn, err := listener.Accept()
		if err != nil {
			// TODO consider using something like https://github.com/hydrogen18/stoppableListener?
			if !strings.Contains(strings.ToLower(err.Error()), "use of closed network connection") {
				runtime.HandleError(fmt.Errorf("Error accepting connection on port %d: %v", port.Local, err))
			}
			return
		}

		go func() {
			defer conn.Close()
			if !pf.startConnection() {
				runtime.HandleError(fmt.Errorf("too many connections, can't open port %d -> %d", port.Local, port.Remote))
				return
			}
			glog.V(6).Infof("Starting connection %v", port)
			defer pf.endConnection()
			if err := pf.handleConnection(conn, port); err != nil {
				runtime.HandleError(err)
			} else {
				glog.V(6).Infof("Finished connection %v", port)
			}
		}()
	}
}

// handleConnection copies data between the local connection and the stream to
// the remote server.
func (pf *PortForwarder) handleConnection(conn net.Conn, port ForwardedPort) error {
	if pf.out != nil {
		fmt.Fprintf(pf.out, "Handling connection for %d\n", port.Local)
	}

	glog.V(4).Infof("Dialing remote port %d with protocol %s", port.Remote, protocolName)
	ws, err := pf.dialer(port.Remote, protocolName)
	if err != nil {
		return fmt.Errorf("error upgrading connection: %s", err)
	}
	// track this connection for Close()
	pf.addConnection(ws)

	defer func() {
		glog.V(6).Infof("About to close connection for port %d -> %d", port.Local, port.Remote)
		ws.Close()
		pf.removeConnection(ws)
		glog.V(6).Infof("Closed connection for %d -> %d", port.Local, port.Remote)
	}()

	wsConn := wsstream.NewConn(map[string]wsstream.ChannelProtocolConfig{
		protocolName: {Binary: true, Channels: []wsstream.ChannelType{wsstream.ReadWriteChannel, wsstream.ReadChannel}},
	})
	actualProtocol, streams, err := wsConn.OpenChannels(ws)
	if err != nil {
		return fmt.Errorf("error setting up channels for %d -> %d: %v", port.Local, port.Remote, err)
	}
	if actualProtocol != protocolName {
		return fmt.Errorf("server did not use our supported protocol %s: %s", protocolName, actualProtocol)
	}
	glog.V(4).Infof("Got %d streams for communication", len(streams))

	errorChan := make(chan error)
	go func() {
		defer close(errorChan)
		if err := checkPort(streams[1], port.Remote); err != nil {
			errorChan <- fmt.Errorf("error establishing error channel: %v", err)
			return
		}
		glog.V(6).Infof("Got expected prefix from stream 1")

		message, err := ioutil.ReadAll(streams[1])
		switch {
		case err != nil:
			errorChan <- fmt.Errorf("error reading from error stream for port %d -> %d: %v", port.Local, port.Remote, err)
		case len(message) > 0:
			errorChan <- fmt.Errorf("an error occurred forwarding %d -> %d: %v", port.Local, port.Remote, string(message))
		}
		glog.V(6).Infof("Reading error channel port %d -> %d finished", port.Local, port.Remote)
	}()

	localError := make(chan struct{})
	remoteDone := make(chan struct{})

	go func() {
		defer close(remoteDone)

		if err := checkPort(streams[0], port.Remote); err != nil {
			runtime.HandleError(fmt.Errorf("error establishing stream: %v", err))
			return
		}
		glog.V(6).Infof("Got expected prefix from stream 0")
		// Copy from the remote side to the local port.
		if _, err := io.Copy(conn, streams[0]); err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			runtime.HandleError(fmt.Errorf("error copying from remote stream to local connection: %v", err))
		}

		// inform the select below that the remote copy is done
		glog.V(6).Infof("Reading remote port %d -> %d finished", port.Local, port.Remote)
	}()

	go func() {
		defer streams[0].Close()

		// Copy from the local port to the remote side.
		if _, err := io.Copy(streams[0], conn); err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			runtime.HandleError(fmt.Errorf("error copying from local connection to remote stream: %v", err))
			// break out of the select below without waiting for the other copy to finish
			close(localError)
		}
		glog.V(6).Infof("Writing local port %d -> %d finished", port.Local, port.Remote)
	}()

	// wait for either a local->remote error or for copying from remote->local to finish
	select {
	case <-remoteDone:
	case <-localError:
	}

	wsConn.Close()

	// always expect something on errorChan (it may be nil)
	return <-errorChan
}

// checkPort verifies that the first two bytes from the stream have the expected header.
func checkPort(r io.Reader, expected uint16) error {
	var data [2]byte
	if _, err := io.ReadAtLeast(r, data[:], 2); err != nil {
		return err
	}
	if actual := binary.LittleEndian.Uint16(data[:]); actual != expected {
		return fmt.Errorf("expected to receive port %d as header, but got %d", expected, actual)
	}
	return nil
}

func (pf *PortForwarder) Close() {
	pf.lock.Lock()
	defer pf.lock.Unlock()

	pf.closed = true

	// stop all listeners
	for _, l := range pf.listeners {
		if err := l.Close(); err != nil {
			runtime.HandleError(fmt.Errorf("error closing listener: %v", err))
		}
	}
	pf.listeners = nil
	for _, c := range pf.connections {
		if err := c.Close(); err != nil {
			runtime.HandleError(fmt.Errorf("error closing connection: %v", err))
		}
	}
	pf.connections = nil
}

func (pf *PortForwarder) startConnection() bool {
	pf.lock.Lock()
	defer pf.lock.Unlock()
	if pf.closed {
		return false
	}
	if pf.activeConnections >= pf.maxConnections {
		return false
	}
	pf.activeConnections++
	return true
}

func (pf *PortForwarder) endConnection() {
	pf.lock.Lock()
	defer pf.lock.Unlock()
	if pf.activeConnections > 0 {
		pf.activeConnections--
	}
}

func (pf *PortForwarder) addConnection(conn *websocket.Conn) {
	pf.lock.Lock()
	defer pf.lock.Unlock()
	if pf.closed {
		conn.Close()
		return
	}
	pf.connections = append(pf.connections, conn)
}

func (pf *PortForwarder) removeConnection(conn *websocket.Conn) {
	pf.lock.Lock()
	defer pf.lock.Unlock()
	for i, c := range pf.connections {
		if c == conn {
			pf.connections = append(pf.connections[:i], pf.connections[i+1:]...)
			break
		}
	}
}

/*
	valid port specifications:

	5000
	- forwards from localhost:5000 to pod:5000

	8888:5000
	- forwards from localhost:8888 to pod:5000

	0:5000
	:5000
	- selects a random available local port,
	  forwards from localhost:<random port> to pod:5000
*/
func parsePorts(ports []string) ([]ForwardedPort, error) {
	var forwards []ForwardedPort
	for _, portString := range ports {
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
			return nil, fmt.Errorf("Invalid port format '%s'", portString)
		}

		localPort, err := strconv.ParseUint(localString, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("Error parsing local port '%s': %s", localString, err)
		}

		remotePort, err := strconv.ParseUint(remoteString, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("Error parsing remote port '%s': %s", remoteString, err)
		}
		if remotePort == 0 {
			return nil, fmt.Errorf("Remote port must be > 0")
		}

		forwards = append(forwards, ForwardedPort{uint16(localPort), uint16(remotePort)})
	}

	return forwards, nil
}
