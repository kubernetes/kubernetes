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
	serverReachable(address string) bool
	parseServerList(serverList []string) error
	CheckEtcdServers() (bool, error)
}

type EtcdConnection struct {
	ServerList []string
}

func (EtcdConnection) serverReachable(address string) bool {
	if conn, err := net.DialTimeout("tcp", address, connectionTimeout); err == nil {
		defer conn.Close()
		return true
	}
	return false
}

func parseServerURI(serverURI string) (string, error) {
	connUrl, err := url.Parse(serverURI)
	if err != nil {
		return "", fmt.Errorf("unable to parse etcd url: %v", err)
	}
	return connUrl.Host, nil
}

// CheckEtcdServers will attempt to reach all etcd servers once. If any
// can be reached, return true.
func (con EtcdConnection) CheckEtcdServers() (done bool, err error) {
	// Attempt to reach every Etcd server in order
	for _, serverUri := range con.ServerList {
		host, err := parseServerURI(serverUri)
		if err != nil {
			return false, err
		}
		if con.serverReachable(host) {
			return true, nil
		}
	}
	return false, nil
}
