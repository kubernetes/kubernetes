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
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// TODO move to API machinery and re-unify with kubelet/server/portfoward
// The subprotocol "portforward.k8s.io" is used for port forwarding.
const PortForwardProtocolV1Name = "portforward.k8s.io"

// PortForwarder knows how to listen for local connections and forward them to
// a remote pod via an upgraded HTTP request.
type PortForwarder struct {
	portForwardProtocol string
	ports               []ForwardedPort
	stopChan            <-chan struct{}

	dialer        httpstream.Dialer
	streamConn    httpstream.Connection
	listeners     []io.Closer
	Ready         chan struct{}
	requestIDLock sync.Mutex
	requestID     int
	out           io.Writer
	errOut        io.Writer
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

func parsePortForwardProtocol(s string) (string, error) {
	switch s {
	case "UDP", "udp", "UDP4", "udp4":
		return api.PortForwardProtocolTypeUdp4, nil
	case "UDP6", "udp6":
		return api.PortForwardProtocolTypeUdp6, nil
	case "TCP", "tcp", "TCP4", "tcp4":
		return api.PortForwardProtocolTypeTcp4, nil
	case "TCP6", "tcp6":
		return api.PortForwardProtocolTypeTcp6, nil
	default:
		return "", errors.New("Protocol must be UDP4/6 or TCP4/6")
	}
}

// New creates a new PortForwarder.
func New(dialer httpstream.Dialer, protocol string, ports []string, stopChan <-chan struct{}, readyChan chan struct{}, out, errOut io.Writer) (*PortForwarder, error) {
	if len(ports) == 0 {
		return nil, errors.New("You must specify at least 1 port")
	}
	parsedPorts, err := parsePorts(ports)
	if err != nil {
		return nil, err
	}
	parsedProtocol, err := parsePortForwardProtocol(protocol)
	if err != nil {
		return nil, err
	}

	return &PortForwarder{
		portForwardProtocol: parsedProtocol,
		dialer:              dialer,
		ports:               parsedPorts,
		stopChan:            stopChan,
		Ready:               readyChan,
		out:                 out,
		errOut:              errOut,
	}, nil
}

// ForwardPorts formats and executes a port forwarding request. The connection will remain
// open until stopChan is closed.
func (pf *PortForwarder) ForwardPorts() error {
	defer pf.Close()

	var err error
	pf.streamConn, _, err = pf.dialer.Dial(PortForwardProtocolV1Name)
	if err != nil {
		return fmt.Errorf("error upgrading connection: %s", err)
	}
	defer pf.streamConn.Close()

	return pf.forward()
}

// forward dials the remote host specific in req, upgrades the request, starts
// listeners for each port specified in ports, and forwards local connections
// to the remote host via streams.
func (pf *PortForwarder) forward() error {
	if pf.isUdp() {
		for _, port := range pf.ports {
			addr := net.UDPAddr{
				Port: int(port.Local),
				IP:   net.ParseIP("127.0.0.1"),
			}
			conn, err := net.ListenUDP("udp", &addr)
			if err != nil {
				return fmt.Errorf("ListenUDP error %s", err.Error())
			}
			go pf.waitForUDPSocket(conn, port)
		}

	} else {
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
	}

	// wait for interrupt or conn closure
	select {
	case <-pf.stopChan:
	case <-pf.streamConn.CloseChan():
		runtime.HandleError(errors.New("lost connection to pod"))
	}
	return nil
}

func (pf *PortForwarder) Close() {
	// stop all listeners
	for _, l := range pf.listeners {
		if err := l.Close(); err != nil {
			runtime.HandleError(fmt.Errorf("error closing listener: %v", err))
		}
	}
}
