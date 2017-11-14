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

// +build !windows

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"fmt"
	"net"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

const (
	sockFile = "/tmp/kms-provider.sock"
)

func TestUnixSockEndpoint(t *testing.T) {
	// Start the gRPC server that listens on unix socket.
	listener, err := unixSockListner()
	if err != nil {
		t.Fatal(err)
	}

	server := startTestKmsProvider(listener)
	defer func() {
		server.Stop()
		if err := cleanSockFile(); err != nil {
			t.Fatal(err)
		}
	}()

	endpoint := unixProtocol + "://" + sockFile
	verifyService(t, endpoint, "", "", "")
}

func unixSockListner() (net.Listener, error) {
	if err := cleanSockFile(); err != nil {
		return nil, err
	}

	listener, err := net.Listen(unixProtocol, sockFile)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}

	return listener, nil
}

func cleanSockFile() error {
	err := unix.Unlink(sockFile)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete the socket file, error: %v", err)
	}
	return nil
}
