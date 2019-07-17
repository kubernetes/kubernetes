/*
Copyright 2016 The Kubernetes Authors.

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

package nat

/*
client/server for testing CLOSE_WAIT timeout condition in iptables NAT.

client              server
  |                   |
  |<--tcp handshake-->|
  |<-------fin--------| half-close from server
  |                   | client is in CLOSE_WAIT
*/

import (
	"errors"
	"io"
	"log"
	"net"
	"time"

	"k8s.io/kubernetes/test/images/agnhost/net/common"
)

// leakedConnection is a global variable that should leak the active
// connection assigned here.
var leakedConnection *net.TCPConn

// CloseWaitServerOptions holds server JSON options.
type CloseWaitServerOptions struct {
	// Address to bind for the test
	LocalAddr string
	// Timeout to wait after sending the FIN.
	PostFinTimeoutSeconds int
}

type closeWaitServer struct {
	options *CloseWaitServerOptions
}

// NewCloseWaitServer returns a new Runner.
func NewCloseWaitServer() common.Runner {
	return &closeWaitServer{}
}

// NewOptions allocates new options structure.
func (server *closeWaitServer) NewOptions() interface{} {
	return &CloseWaitServerOptions{}
}

// Run the server-side of the test.
func (server *closeWaitServer) Run(logger *log.Logger, rawOptions interface{}) error {
	if options, ok := rawOptions.(*CloseWaitServerOptions); ok {
		server.options = options
	} else {
		return errors.New("invalid type")
	}

	logger.Printf("Run %v", server.options)

	addr, err := net.ResolveTCPAddr("tcp", server.options.LocalAddr)
	if err != nil {
		return err
	}

	listener, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return err
	}
	defer listener.Close()

	logger.Printf("Server listening on %v", addr)

	conn, err := listener.AcceptTCP()
	if err != nil {
		return err
	}
	defer conn.Close()

	logger.Printf("Client connected")

	// Send client half-close FIN so client is now in CLOSE_WAIT. We keep
	// the client -> server pipe open to verify whether or not the NAT
	// dropped our connection.
	if err := conn.CloseWrite(); err != nil {
		return err
	}

	logger.Printf("Server sent FIN, waiting %v seconds",
		server.options.PostFinTimeoutSeconds)

	<-time.After(time.Duration(server.options.PostFinTimeoutSeconds) * time.Second)

	logger.Printf("Done")

	return nil
}

// CloseWaitClientOptions holds client JSON options.
type CloseWaitClientOptions struct {
	// RemoteAddr of the server to connect to.
	RemoteAddr string
	// TimeoutSeconds on I/O with the server.
	TimeoutSeconds int
	// Half-close timeout (to give the test time to check the status of the
	// conntrack table entry.
	PostFinTimeoutSeconds int
	// Leak connection (assign to global variable so connection persists
	// as long as the process remains.
	LeakConnection bool
}

type closeWaitClient struct {
	options *CloseWaitClientOptions
}

// NewCloseWaitClient creates a new runner
func NewCloseWaitClient() common.Runner {
	return &closeWaitClient{}
}

// NewOptions allocates new options structure.
func (client *closeWaitClient) NewOptions() interface{} {
	return &CloseWaitClientOptions{}
}

// Run the client.m
func (client *closeWaitClient) Run(logger *log.Logger, rawOptions interface{}) error {
	if options, ok := rawOptions.(*CloseWaitClientOptions); ok {
		client.options = options
	} else {
		return errors.New("invalid type")
	}

	logger.Printf("Run %v", client.options)

	addr, err := net.ResolveTCPAddr("tcp", client.options.RemoteAddr)
	if err != nil {
		return err
	}

	conn, err := net.DialTCP("tcp", nil, addr)
	if err != nil {
		return err
	}
	if !client.options.LeakConnection {
		defer conn.Close()
	}

	logger.Printf("Connected to server")

	if client.options.TimeoutSeconds > 0 {
		delay := time.Duration(client.options.TimeoutSeconds) * time.Second
		conn.SetReadDeadline(time.Now().Add(delay))
	}

	buf := make([]byte, 1, 1)
	size, err := conn.Read(buf)

	if err != nil && err != io.EOF {
		return err
	}

	if size != 0 {
		return errors.New("Got data but expected EOF")
	}

	logger.Printf("Server has half-closed the connection, waiting %v seconds",
		client.options.PostFinTimeoutSeconds)

	if client.options.LeakConnection {
		logger.Printf("Leaking client connection (assigning to global variable)")
		leakedConnection = conn
	}

	<-time.After(
		time.Duration(client.options.PostFinTimeoutSeconds) * time.Second)

	logger.Printf("Done")

	return nil
}
