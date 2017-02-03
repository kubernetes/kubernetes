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

	utilwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

const retryLimit = 60
const retryInterval = 1 * time.Second

type connection interface {
	serverReachable(address string) bool
	parseServerList(serverList []string) error
	checkEtcdServers() (bool, error)
}

type etcdConnection struct {
	hosts []string
}

func (etcdConnection) serverReachable(address string) bool {
	if conn, err := net.Dial("tcp", address); err == nil {
		defer conn.Close()
		return true
	}
	return false
}

func (con *etcdConnection) parseServerList(serverList []string) error {
	con.hosts = make([]string, len(serverList))
	for idx, serverURI := range(serverList) {
		connUrl, err := url.Parse(serverURI)
		if err != nil {
			return fmt.Errorf("unable to parse etcd url: %v", err)
		}
		con.hosts[idx] = connUrl.Host
	}
	return nil
}

// checkEtcdServers will attempt to reach all etcd servers once. If any
// can be reached, return true.
func (con etcdConnection) checkEtcdServers() (done bool, err error) {
	// Attempt to reach every Etcd server in order
	for _, host := range con.hosts {
		if con.serverReachable(host) {return true, nil}
	}
	return false, nil
}

func waitForAvailableEtcd(etcd connection) error {
	err := utilwait.PollImmediate(retryInterval, retryLimit, etcd.checkEtcdServers)
	if err != nil {
		return fmt.Errorf("unable to reach any etcd server: %v", err)
	}
	return nil
}

// RunAPIServerChecks ensures the apiserver is ready to execute. This function
// will block until etcd is ready.
func RunAPIServerChecks(s *options.ServerRunOptions) error {
	etcd := new(etcdConnection)
	err := etcd.parseServerList(s.Etcd.StorageConfig.ServerList)
	if err != nil {return err}
	return waitForAvailableEtcd(etcd)
}
