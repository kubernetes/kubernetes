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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
	"github.com/golang/glog"
)

type upgrader interface {
	upgrade(*client.Request, *client.Config) (httpstream.Connection, error)
}

type defaultUpgrader struct{}

func (u *defaultUpgrader) upgrade(req *client.Request, config *client.Config) (httpstream.Connection, error) {
	return req.Upgrade(config, spdy.NewRoundTripper)
}

// PortForwarder knows how to listen for local connections and forward them to
// a remote pod via an upgraded HTTP request.
type PortForwarder struct {
	req      *client.Request
	config   *client.Config
	ports    []ForwardedPort
	stopChan <-chan struct{}

	streamConn httpstream.Connection
	listeners  []io.Closer
	upgrader   upgrader
	Ready      chan struct{}
}

// ForwardedPort contains a Local:Remote port pairing.
type ForwardedPort struct {
	Local  uint16
	Remote uint16
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

// New creates a new PortForwarder.
func New(req *client.Request, config *client.Config, ports []string, stopChan <-chan struct{}) (*PortForwarder, error) {
	if len(ports) == 0 {
		return nil, errors.New("You must specify at least 1 port")
	}
	parsedPorts, err := parsePorts(ports)
	if err != nil {
		return nil, err
	}

	return &PortForwarder{
		req:      req,
		config:   config,
		ports:    parsedPorts,
		stopChan: stopChan,
		Ready:    make(chan struct{}),
	}, nil
}

// ForwardPorts formats and executes a port forwarding request. The connection will remain
// open until stopChan is closed.
func (pf *PortForwarder) ForwardPorts() error {
	defer pf.Close()

	if pf.upgrader == nil {
		pf.upgrader = &defaultUpgrader{}
	}
	var err error
	pf.streamConn, err = pf.upgrader.upgrade(pf.req, pf.config)
	if err != nil {
		return fmt.Errorf("Error upgrading connection: %s", err)
	}
	defer pf.streamConn.Close()

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
		if err != nil {
			glog.Warningf("Unable to listen on port %d: %v", port, err)
		}
		listenSuccess = true
	}

	if !listenSuccess {
		return fmt.Errorf("Unable to listen on any of the requested ports: %v", pf.ports)
	}

	close(pf.Ready)

	// wait for interrupt or conn closure
	select {
	case <-pf.stopChan:
	case <-pf.streamConn.CloseChan():
		glog.Errorf("Lost connection to pod")
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
	listener, err := net.Listen(protocol, fmt.Sprintf("%s:%d", hostname, port.Local))
	if err != nil {
		glog.Errorf("Unable to create listener: Error %s", err)
		return nil, err
	}
	listenerAddress := listener.Addr().String()
	host, localPort, _ := net.SplitHostPort(listenerAddress)
	localPortUInt, err := strconv.ParseUint(localPort, 10, 16)

	if err != nil {
		return nil, fmt.Errorf("Error parsing local port: %s from %s (%s)", err, listenerAddress, host)
	}
	port.Local = uint16(localPortUInt)
	glog.Infof("Forwarding from %s:%d -> %d", hostname, localPortUInt, port.Remote)

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
				glog.Errorf("Error accepting connection on port %d: %v", port.Local, err)
			}
			return
		}
		go pf.handleConnection(conn, port)
	}
}

// handleConnection copies data between the local connection and the stream to
// the remote server.
func (pf *PortForwarder) handleConnection(conn net.Conn, port ForwardedPort) {
	defer conn.Close()

	glog.Infof("Handling connection for %d", port.Local)

	errorChan := make(chan error)
	doneChan := make(chan struct{}, 2)

	// create error stream
	headers := http.Header{}
	headers.Set(api.StreamType, api.StreamTypeError)
	headers.Set(api.PortHeader, fmt.Sprintf("%d", port.Remote))
	errorStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		glog.Errorf("Error creating error stream for port %d -> %d: %v", port.Local, port.Remote, err)
		return
	}
	defer errorStream.Reset()
	go func() {
		message, err := ioutil.ReadAll(errorStream)
		if err != nil && err != io.EOF {
			errorChan <- fmt.Errorf("Error reading from error stream for port %d -> %d: %v", port.Local, port.Remote, err)
		}
		if len(message) > 0 {
			errorChan <- fmt.Errorf("An error occurred forwarding %d -> %d: %v", port.Local, port.Remote, string(message))
		}
	}()

	// create data stream
	headers.Set(api.StreamType, api.StreamTypeData)
	dataStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		glog.Errorf("Error creating forwarding stream for port %d -> %d: %v", port.Local, port.Remote, err)
		return
	}
	// Send a Reset when this function exits to completely tear down the stream here
	// and in the remote server.
	defer dataStream.Reset()

	go func() {
		// Copy from the remote side to the local port. We won't get an EOF from
		// the server as it has no way of knowing when to close the stream.  We'll
		// take care of closing both ends of the stream with the call to
		// stream.Reset() when this function exits.
		if _, err := io.Copy(conn, dataStream); err != nil && err != io.EOF && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error copying from remote stream to local connection: %v", err)
		}
		doneChan <- struct{}{}
	}()

	go func() {
		// Copy from the local port to the remote side. Here we will be able to know
		// when the Copy gets an EOF from conn, as that will happen as soon as conn is
		// closed (i.e. client disconnected).
		if _, err := io.Copy(dataStream, conn); err != nil && err != io.EOF && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error copying from local connection to remote stream: %v", err)
		}
		doneChan <- struct{}{}
	}()

	select {
	case err := <-errorChan:
		glog.Error(err)
	case <-doneChan:
	}
}

func (pf *PortForwarder) Close() {
	// stop all listeners
	for _, l := range pf.listeners {
		if err := l.Close(); err != nil {
			glog.Errorf("Error closing listener: %v", err)
		}
	}
}
