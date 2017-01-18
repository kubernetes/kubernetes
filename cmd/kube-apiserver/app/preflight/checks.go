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
	"net"
	"net/url"
	"time"
	"errors"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

const retryLimit = 6
const retryInterval = 10 * time.Second

type clock interface {
	Sleep(time.Duration)
}

type realTimer struct{}
func (realTimer) Sleep(d time.Duration) {time.Sleep(d)}

var timer clock = new(realTimer)

type connection interface {
	checkConnection(address string) bool
}

type etcdConnection struct {}

var etcd connection = new(etcdConnection)

func (etcdConnection) checkConnection(address string) bool {
	if conn, err := net.Dial("tcp", address); err == nil {
		defer conn.Close()
		return true
	}
	return false
}

func checkEtcdServer(address string, foundEtcd chan struct{})  {
	retries := 0
	for retries < retryLimit {
		retries += 1
		timer.Sleep(retryInterval)
		if etcd.checkConnection(address) {
			select {
			case foundEtcd <- struct{}{}:
			default:
			}
		}
	}
}

func WaitForEtcd(serverList []string) error {
	foundEtcd := make(chan struct{}, 1)
	timeout := make(chan struct{})

	// Attempt to reach every Etcd server in goroutines
	for _, serverURI := range serverList {
		connUrl, _ := url.Parse(serverURI)
		go checkEtcdServer(connUrl.Host, foundEtcd)
	}

	// overall timeout
	go func() {
		timer.Sleep(retryInterval * retryLimit)
		timeout <- struct{}{}
	}()

	select {
	case <-foundEtcd:
		{return nil}
	case <- timeout:
		{return errors.New("Unable to reach any Etcd server")}
	}
}

func RunApiserverChecks(s *options.ServerRunOptions) error {
	return WaitForEtcd(s.Etcd.StorageConfig.ServerList)
}
