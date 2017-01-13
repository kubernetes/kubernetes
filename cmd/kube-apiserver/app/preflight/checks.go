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
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"net"
	"net/url"
	"time"
	"fmt"
	"errors"
)

var retryLimit int = 6
var retryInterval time.Duration = 10

type clock interface {
	Sleep(time.Duration)
}

type realTimer struct{}
func (realTimer) Sleep(d time.Duration) {time.Sleep(d)}

var timer clock = new(realTimer)

type connection interface {
	checkConnection(string) bool
}

type realEtcdConnection struct {}

var etcdConnection connection = new(realEtcdConnection)

func (realEtcdConnection) checkConnection(connString string) bool {
	if conn, err := net.Dial("tcp", connString); err != nil {
		conn.Close()
		return true
	}
	return false
}

func WaitForEtcd(serverList []string) error {
	for _, connString := range serverList {
		connUrl, err := url.Parse(connString)
		if err != nil {
			return errors.New(fmt.Sprintf("error parsing Etcd URI: %v", err))
		}

		done := etcdConnection.checkConnection(connUrl.Host)
		retries := 0
		for (!done) && (retries < retryLimit) {
			retries += 1
			timer.Sleep(retryInterval)
			done = etcdConnection.checkConnection(connUrl.Host)
		}
		if retries >= retryLimit {
			return errors.New(fmt.Sprint("unable to reach Etcd server: %s", connString))
		}
	}
	return nil
}

func RunApiserverChecks(s *options.ServerRunOptions) error {
	return WaitForEtcd(s.Etcd.StorageConfig.ServerList)
}
