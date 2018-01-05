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
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/apimachinery/pkg/util/httpstream"
)

func (pf *PortForwarder) isUdp() bool {
	if pf.portForwardProtocol == api.PortForwardProtocolTypeUdp4 ||
		pf.portForwardProtocol == api.PortForwardProtocolTypeUdp6 {
		return true
	}
	return false
}

type UDPSocket map[string]io.ReadWriteCloser

func (pf *PortForwarder) setupBackToUDPClient(stream httpstream.Stream, connUDP *net.UDPConn, addr *net.UDPAddr) {
	bufRecv := make([]byte, 1024)
	//forever loop
	fmt.Fprintf(pf.out, "setupBackToUDPClient started...\n")
	for {
		_, err := stream.Read(bufRecv)
		if err != nil {
			fmt.Fprintf(pf.out, "setupBackToUDPClient: Error read from spdy: %s\n", err)
			break
		} else {
			fmt.Fprintf(pf.out, "setupBackToUDPClient 2: received from spdy, data: %s\n", string(bufRecv[:1024]))
		}

		_, err = connUDP.WriteToUDP(bufRecv[:1024], addr)
		if err != nil {
			fmt.Fprintln(pf.out, "setupBackToUDPClient 2: Error write to udp %v: %s \n", addr, err)
			break
		} else {
			fmt.Fprintf(pf.out, "setupBackToUDPClient 2: send to udp %v\n, data: %s\n", addr, string(bufRecv[:1024]))
		}
	}
}

func createKey(addr *net.UDPAddr) string {
	return addr.IP.String() + strconv.Itoa(addr.Port)
}

func (pf *PortForwarder) waitForUDPSocket(conn *net.UDPConn, port ForwardedPort) {
	bufSend := make([]byte, 1024)
	socketMap := make(UDPSocket)

	fmt.Fprintf(pf.out, "waitForUDPSocket started\n")

	for {
		rn, rmAddr, err := conn.ReadFromUDP(bufSend)
		if err != nil {
			fmt.Fprintf(pf.out, "ReadFromUDP wrong\n")
		}

		if socketMap[createKey(rmAddr)] == nil {
			dataStream, errorStream, err := pf.createStream(conn, port)
			if err != nil {
				runtime.HandleError(err)
				continue
			}

			go pf.handleErrorStream(errorStream, port)
			go pf.setupBackToUDPClient(dataStream, conn, rmAddr)
			socketMap[createKey(rmAddr)] = dataStream
		}

		_, err = socketMap[createKey(rmAddr)].Write(bufSend[:rn])
		if err != nil {
			fmt.Fprintf(pf.out, "goroutine 1: Error write to spdy:", err)
			break
		} else {
			fmt.Fprintf(pf.out, "goroutine 1: send to spdy, data: %s\n", string(bufSend[:rn]))
		}
	}
}

func (pf *PortForwarder) handleErrorStream(errorStream httpstream.Stream, port ForwardedPort) {
	// we're not writing to this stream
	errorStream.Close()

	errorChan := make(chan error)
	go func() {
		message, err := ioutil.ReadAll(errorStream)
		switch {
		case err != nil:
			errorChan <- fmt.Errorf("error reading from error stream for port %d -> %d: %v", port.Local, port.Remote, err)
		case len(message) > 0:
			errorChan <- fmt.Errorf("an error occurred forwarding %d -> %d: %v", port.Local, port.Remote, string(message))
		}
		close(errorChan)
	}()

	err := <-errorChan
	if err != nil {
		runtime.HandleError(err)
	}

}
func (pf *PortForwarder) createStream(conn *net.UDPConn, port ForwardedPort) (httpstream.Stream, httpstream.Stream, error) {
	requestID := pf.nextRequestID()

	// create error stream
	headers := http.Header{}
	headers.Set(api.PortForwardProtocolType, api.PortForwardProtocolTypeUdp4)
	headers.Set(v1.StreamType, v1.StreamTypeError)
	headers.Set(v1.PortHeader, fmt.Sprintf("%d", port.Remote))
	headers.Set(v1.PortForwardRequestIDHeader, strconv.Itoa(requestID))
	errorStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating error stream for port %d -> %d: %v", port.Local, port.Remote, err)
	}
	// create data stream
	headers.Set(v1.StreamType, v1.StreamTypeData)
	dataStream, err := pf.streamConn.CreateStream(headers)
	if err != nil {
		return nil, nil, fmt.Errorf("error creating forwarding stream for port %d -> %d: %v", port.Local, port.Remote, err)
	}

	return dataStream, errorStream, nil
}

func ReadFromStreamAndSendToUDP(stream io.ReadWriteCloser, connUDP *net.UnixConn) error {
	bufSend := make([]byte, 1024)
	bufRecv := make([]byte, 1024)
	glog.V(3).Infof("haha:  we are into ReadFromStreamAndSendToUDP \n")

	go func() error {
		glog.V(3).Infof("haha: goroutine 1 started...\n")
		//forever loop
		for {
			n, err := stream.Read(bufSend)
			if err != nil {
				glog.V(3).Infof("haha: goroutine 1: error read from spdy: %s\n", err)
				break
			} else {
				glog.V(3).Infof("haha: goroutine 1: received from spdy, data: %s\n", string(bufSend))
			}
			_, err = connUDP.Write(bufSend[:n])
			if err != nil {
				glog.V(3).Infof("haha: goroutine 1: error write to udp: %s\n", err)
				break
			} else {
				glog.V(3).Infof("haha: goroutine 1: send to udp, data: %s\n", string(bufSend))
			}
		}
		return nil
	}()

	go func() error {
		//forever loop
		glog.V(3).Infof("haha: goroutine 2 started...\n")
		for {
			_, addr, err := connUDP.ReadFrom(bufRecv)
			if err != nil {
				glog.V(3).Infof("haha: goroutine 2: error read from udp: %s\n", err)
				break
			} else {
				glog.V(3).Infof("haha: goroutine 2: received from udp, data: %s, from %s\n", string(bufRecv), addr)
			}
			_, err = stream.Write(bufRecv)
			if err != nil {
				glog.V(3).Infof("haha: goroutine 2: error write to spdy: %s\n", err)
				break
			} else {
				glog.V(3).Infof("haha: goroutine 2: send to spdy, data: %s\n", string(bufRecv))
			}
		}
		return nil
	}()

	return nil
}

func ReadFromUDPAndSendToStream(out io.Writer, stream io.ReadWriteCloser, connUDP *net.UDPConn) error {

	bufSend := make([]byte, 1024)
	bufRecv := make([]byte, 1024)

	var TMP *net.UDPAddr

	go func() error {
		//forever loop
		fmt.Fprintf(out, "goroutine 1 started...\n")
		for {
			rn, rmAddr, err := connUDP.ReadFromUDP(bufSend)
			if err != nil {
				fmt.Fprintf(out, "goroutine 1: Error read from udp:", err)
				break
			} else {
				TMP = rmAddr
				fmt.Fprintf(out, "goroutine 1: received from udp, data: %s\n", string(bufSend[:rn]))
			}
			/////////////////////////////////////////////////////////
			_, err = stream.Write(bufSend[:rn])
			if err != nil {
				fmt.Fprintf(out, "goroutine 1: Error write to spdy:", err)
				break
			} else {
				fmt.Fprintf(out, "goroutine 1: send to spdy, data: %s\n", string(bufSend[:rn]))
			}
		}
		return nil
	}()

	go func() error {
		//forever loop
		fmt.Fprintf(out, "goroutine 2 started...\n")
		for {
			_, err := stream.Read(bufRecv)
			if err != nil {
				fmt.Fprintf(out, "goroutine 2: Error read from spdy: %s\n", err)
				break
			} else {
				fmt.Fprintf(out, "goroutine 2: received from spdy, data: %s\n", string(bufRecv[:1024]))
			}
			/////////////////////////////////////////////////////////////////////

			_, err = connUDP.WriteToUDP(bufRecv[:1024], TMP)
			if err != nil {
				fmt.Fprintln(out, "goroutine 2: Error write to udp %v: %s \n", TMP, err)
				break
			} else {
				fmt.Fprintf(out, "goroutine 2: send to udp %v\n, data: %s\n", TMP, string(bufRecv[:1024]))
			}

		}
		return nil
	}()

	return nil
}
