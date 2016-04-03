// Copyright 2015 Google Inc. All Rights Reserved.
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

package client

import (
	"fmt"
	"net"

	"github.com/golang/glog"
)

type Client struct {
	HostPort  string
	Namespace string
	conn      net.Conn
}

func (self *Client) Open() error {
	conn, err := net.Dial("udp", self.HostPort)
	if err != nil {
		glog.Errorf("failed to open udp connection to %q: %v", self.HostPort, err)
		return err
	}
	self.conn = conn
	return nil
}

func (self *Client) Close() error {
	self.conn.Close()
	self.conn = nil
	return nil
}

// Simple send to statsd daemon without sampling.
func (self *Client) Send(namespace, containerName, key string, value uint64) error {
	// only send counter value
	formatted := fmt.Sprintf("%s.%s.%s:%d|g", namespace, containerName, key, value)
	_, err := fmt.Fprintf(self.conn, formatted)
	if err != nil {
		glog.V(3).Infof("failed to send data %q: %v", formatted, err)
		return err
	}
	return nil
}

func New(hostPort string) (*Client, error) {
	Client := Client{HostPort: hostPort}
	if err := Client.Open(); err != nil {
		return nil, err
	}
	return &Client, nil
}
