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

package preflight

import (
	"fmt"
	"net"
	"net/url"
	"time"
)

const connectionTimeout = 1 * time.Second

type connection interface {
	serverReachable(address string) (bool, error)
	parseServerList(serverList []string) error
	CheckEtcdServers() (bool, error)
}

// EtcdConnection holds the Etcd server list
type EtcdConnection struct {
	ServerList []string
}

func (EtcdConnection) serverReachable(connURL *url.URL) error {
	scheme := connURL.Scheme
	if scheme == "http" || scheme == "https" || scheme == "tcp" {
		scheme = "tcp"
	}
	conn, err := net.DialTimeout(scheme, connURL.Host, connectionTimeout)
	if err == nil {
		defer conn.Close()
	}
	return err
}

func parseServerURI(serverURI string) (*url.URL, error) {
	connURL, err := url.Parse(serverURI)
	if err != nil {
		return &url.URL{}, fmt.Errorf("unable to parse etcd url: %v", err)
	}
	return connURL, nil
}

// CheckEtcdServers will attempt to reach all etcd servers once. If any
// can be reached, return true.
func (con EtcdConnection) CheckEtcdServers() (done bool, err error) {
	// Attempt to reach every Etcd server in order
	for _, serverURI := range con.ServerList {
		var host *url.URL
		host, err = parseServerURI(serverURI)
		if err != nil {
			return false, err
		}
		if err = con.serverReachable(host); err == nil {
			return true, err
		}
	}
	// con.ServerList is empty or last server was not reachable
	return false, err
}
