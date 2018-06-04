/*
Copyright 2018 The Kubernetes Authors.

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
	"bufio"
	"errors"
	"fmt"
	"github.com/golang/glog"
	"io"
	"io/ioutil"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/runtime"
	"math"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// TODO move to API machinery and re-unify with kubelet/server/portfoward
// The subprotocol "remoteportforward.k8s.io" is used for remote port forwarding.
const RemotePortForwardProtocolV1Name = "remoteportforward.k8s.io"

// RemotePortForwarder knows how to listen for remote connections and forward them to
// a local port via an upgraded HTTP request.
type RemotePortForwarder struct {
	ports    []ForwardedPort
	stopChan <-chan struct{}

	dialer        httpstream.Dialer
	streamConn    httpstream.Connection
	Ready         chan struct{}
	requestIDLock sync.Mutex
	requestID     int
	out           io.Writer
	errOut        io.Writer
	newStreamChan chan httpstream.Stream
}

// New creates a new RemotePortForwarder.
func NewRemotePortForwarder(dialer httpstream.Dialer, ports []string, stopChan <-chan struct{}, readyChan chan struct{}, out, errOut io.Writer) (*RemotePortForwarder, error) {
	if len(ports) == 0 {
		return nil, errors.New("You must specify at least 1 port")
	}
	parsedPorts, err := parsePorts(ports)
	if err != nil {
		return nil, err
	}
	return &RemotePortForwarder{
		dialer:        dialer,
		ports:         parsedPorts,
		stopChan:      stopChan,
		Ready:         readyChan,
		out:           out,
		errOut:        errOut,
		newStreamChan: make(chan httpstream.Stream),
	}, nil
}

// streams channel will receive a stream whenever the stream connection receives a new stream request from the other end
func streamReceived(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streams <- stream
		return nil
	}
}

func (pf *RemotePortForwarder) Close() {
}

// ForwardPorts executes a port forwarding request. The connection will remain
// open until stopChan is closed.
func (pf *RemotePortForwarder) ForwardPorts() error {
	defer pf.Close()

	var err error
	// FIXME: We should use a new protocol name for remote portforward, like RemotePortForwardProtocolV1Name.
	pf.streamConn, _, err = pf.dialer.Dial(streamReceived(pf.newStreamChan), PortForwardProtocolV1Name)
	if err != nil {
		return fmt.Errorf("error upgrading connection: %s", err)
	}
	defer pf.streamConn.Close()
	return pf.remoteForward()
}

func (pf *RemotePortForwarder) remoteForward() error {
	var err error

	anyListenerSuccess := false
	for _, port := range pf.ports {
		glog.Infof("yuxzhu: will listen on remote port %d", port.Local)
		err = pf.listenOnRemotePort(&port)
		switch {
		case err == nil:
			anyListenerSuccess = true
			fmt.Fprintf(pf.out, "Forwarding from %d to 127.0.0.1:%d...\n", port.Local, port.Remote)
		default:
			if pf.errOut != nil {
				fmt.Fprintf(pf.errOut, "Unable to listen on port %d in the pod: %v\n", port.Local, err)
			}
		}
	}
	if !anyListenerSuccess {
		return fmt.Errorf("Unable to listen on any of the requested Pod ports")
	}

	go pf.waitForRemoteStreams()
	if pf.Ready != nil {
		close(pf.Ready)
	}

	// wait for interrupt or conn closure
	select {
	case <-pf.stopChan:
		glog.Infof("yuxzhu: stopChan received for remote port forwarding...")
	case <-pf.streamConn.CloseChan():
		glog.Infof("yuxzhu: streamConn.CloseChan() received remote port forwarding...")
		runtime.HandleError(errors.New("lost connection to pod"))
	}
	glog.Infof("yuxzhu: remoteForward stopped")
	return nil
}

// remoteListenOnPort asks the remote server to listen on the specified port in the pod.
// A pair of streams, aka listener stream pair, will be created for communication between kubectl and the remote listener.
// The error listener stream is used to get error message from the remote server if the remote listeners fail to start, like what local port forwarding does.
// The data listener stream is used to send ping/pong heartbeats to keep the remote port open.
// When the data listener stream closes, the remote listener should stop.
func (pf *RemotePortForwarder) listenOnRemotePort(port *ForwardedPort) error {
	requestID := pf.nextRequestID()

	// create error stream
	headers := http.Header{}
	headers.Set(v1.StreamType, v1.StreamTypeError)
	headers.Set(v1.PortForwardRemoteHeader, "1")
	headers.Set(v1.PortHeader, strconv.Itoa(int(port.Local)))
	headers.Set(v1.PortForwardRequestIDHeader, strconv.Itoa(requestID))
	errorStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		runtime.HandleError(fmt.Errorf("error creating error listener stream for remote port forwarding %d -> %d: %v", port.Local, port.Remote, err))
		return err
	}
	// we're not writing to error control stream
	errorStream.Close()

	// create data stream
	headers.Set(v1.StreamType, v1.StreamTypeData)
	dataStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		return fmt.Errorf("error creating data listener stream for remote port forwarding %d -> %d: %v", port.Local, port.Remote, err)
	}
	//pf.listeners = append(pf.listeners, dataStream)

	go pf.handleListenerStreamPair(port, dataStream, errorStream)

	return nil
}

func (pf *RemotePortForwarder) handleListenerStreamPair(port *ForwardedPort, dataStream, errorStream httpstream.Stream) {
	defer dataStream.Close()
	errorChan := make(chan error)
	go func() {
		defer close(errorChan)
		message, err := ioutil.ReadAll(errorStream)
		switch {
		case err != nil:
			errorChan <- fmt.Errorf("error reading from listener error stream for remote port forwarding %d -> %d: %v", port.Local, port.Remote, err)
		case len(message) > 0:
			errorChan <- fmt.Errorf("an error occurred forwarding remote port %d -> %d: %v", port.Local, port.Remote, string(message))
		}
	}()

	dataStreamDone := make(chan struct{})
	go func() {
		defer close(dataStreamDone)
		scanner := bufio.NewScanner(dataStream)
		// receive incoming commands sent by kubelet
		for scanner.Scan() {
			line := scanner.Text()
			glog.Infof("remote port forwarder receives data from remote: %v", line)
		}
		if err := scanner.Err(); err != nil {
			runtime.HandleError(fmt.Errorf("error reading from command data stream for remote port forwarding %d -> %d: %v", port.Local, port.Remote, err))
		}
	}()

	// heartbeats
	go func() {
		defer close(dataStreamDone)
		var err error
		for range time.NewTicker(30 * time.Second).C {
			_, err = dataStream.Write([]byte("PING"))
			if err != nil {
				runtime.HandleError(fmt.Errorf("error sending a heartbeat for remote port forwarding %d -> %d: %v", port.Local, port.Remote, err))
				break
			}
		}

	}()

	// wait for dataStreamDone
	<-dataStreamDone

	// always expect something on errorChan (it may be nil)
	err := <-errorChan
	if err != nil {
		runtime.HandleError(err)
	}
}

// waitForRemoteStreams waits for remote streams from the Pod and handles them in
// the background.
func (pf *RemotePortForwarder) waitForRemoteStreams() error {
	for {
		glog.Info("yuxzhu: waiting from remote connections")
		select {
		case stream, ok := <-pf.newStreamChan:
			if !ok {
				break
			}
			glog.Info("yuxzhu: incoming remote connection")
			go pf.handleRemoteConnection(stream)
		}
	}
	return nil
}

func (pf *RemotePortForwarder) handleRemoteConnection(stream httpstream.Stream) {
	defer stream.Close()
	// make sure it has a valid port header
	sourcePortString := stream.Headers().Get(v1.PortHeader)
	if len(sourcePortString) == 0 {
		runtime.HandleError(fmt.Errorf("%q header is required", v1.PortHeader))
		return
	}
	sourcePort, err := strconv.Atoi(sourcePortString)
	if err != nil {
		runtime.HandleError(fmt.Errorf("invalid port number: %v", err))
		return
	}
	// FIXME: assuming pod port != 0 (if so, remote listener will choose a random port, which hasn't been implemented)
	if sourcePort <= 0 || sourcePort > math.MaxUint16 {
		runtime.HandleError(fmt.Errorf("source port number is out of range: %d", sourcePort))
		return
	}
	// lookup the local port
	destPort := 0
	for _, port := range pf.ports {
		if int(port.Local) == sourcePort {
			destPort = int(port.Remote)
		}
	}
	if destPort == 0 {
		runtime.HandleError(fmt.Errorf("port number is not found in current remote portforward request"))
		return
	}
	glog.Infof("connecting to local port %d.", destPort)
	conn, err := net.Dial("tcp4", net.JoinHostPort("localhost", strconv.Itoa(destPort)))
	if err != nil {
		runtime.HandleError(fmt.Errorf("Failed to connect to local port %d", destPort))
		return
	}
	defer conn.Close()
	readChan := make(chan interface{})
	writeChan := make(chan interface{})
	go func() {
		// Copy from the local port to the remote side.
		if _, err := io.Copy(stream, conn); err != nil {
			runtime.HandleError(fmt.Errorf("error copying from local to remote for remote port forwarding %d->%d: %v", sourcePort, destPort, err))
		}
		glog.Infof("local -> pod %d closed", sourcePort)
		close(writeChan)
	}()
	go func() {
		// Copy from the remote side to the local port.
		if _, err := io.Copy(conn, stream); err != nil {
			runtime.HandleError(fmt.Errorf("error copying from remote to local for remote port forwarding %d->%d: %v", sourcePort, destPort, err))
		}
		glog.Infof("pod %d->local closed", sourcePort)
		close(readChan)
	}()

	select {
	case <-readChan:
	case <-writeChan:
	}
	glog.Infof("connection lost for remote port forwarding %d->%d.", sourcePort, destPort)
}

func (pf *RemotePortForwarder) nextRequestID() int {
	pf.requestIDLock.Lock()
	defer pf.requestIDLock.Unlock()
	id := pf.requestID
	pf.requestID++
	return id
}
