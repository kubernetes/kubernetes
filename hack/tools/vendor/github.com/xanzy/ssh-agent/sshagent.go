//
// Copyright 2015, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// +build !windows

package sshagent

import (
	"errors"
	"fmt"
	"net"
	"os"

	"golang.org/x/crypto/ssh/agent"
)

// New returns a new agent.Agent that uses a unix socket
func New() (agent.Agent, net.Conn, error) {
	if !Available() {
		return nil, nil, errors.New("SSH agent requested but SSH_AUTH_SOCK not-specified")
	}

	sshAuthSock := os.Getenv("SSH_AUTH_SOCK")

	conn, err := net.Dial("unix", sshAuthSock)
	if err != nil {
		return nil, nil, fmt.Errorf("Error connecting to SSH_AUTH_SOCK: %v", err)
	}

	return agent.NewClient(conn), conn, nil
}

// Available returns true is a auth socket is defined
func Available() bool {
	return os.Getenv("SSH_AUTH_SOCK") != ""
}
