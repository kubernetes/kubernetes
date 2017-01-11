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
	"fmt"

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

const retryLimit = 60
const retryInterval = 1 * time.Second

type clock interface {
	Sleep(time.Duration)
	After(time.Duration) <-chan time.Time
}

type connectionTimer struct{}
func (connectionTimer) Sleep(d time.Duration) {time.Sleep(d)}
func (connectionTimer) After(d time.Duration) <-chan time.Time {return time.After(d)}

type connection interface {
	serverReachable(address string) bool
}

type etcdConnection struct {}

func (etcdConnection) serverReachable(address string) bool {
	if conn, err := net.Dial("tcp", address); err == nil {
		defer conn.Close()
		return true
	}
	return false
}

func checkEtcdServer(address string, foundEtcd chan struct{}, stop chan struct{}, timer clock, etcd connection) {
	retries := 0
	for retries < retryLimit {
		retries += 1
		select {
		case <-stop:
			return
		default:
		}
		if etcd.serverReachable(address) {
			select {
			case foundEtcd <- struct{}{}:
			default:
			}
		}
		timer.Sleep(retryInterval)
	}
}

func WaitForEtcd(serverList []string, timer clock, etcd connection) error {
	foundEtcd := make(chan struct{}, 1)
	stop := make(chan struct{})
	defer close(stop)

	// Attempt to reach every Etcd server in goroutines
	for _, serverURI := range serverList {
		connUrl, err := url.Parse(serverURI)
		if err != nil {
			return fmt.Errorf("unable to parse etcd url: %v", err)
		}
		go checkEtcdServer(connUrl.Host, foundEtcd, stop, timer, etcd)
	}

	timeout := timer.After(retryInterval * retryLimit)

	select {
	case <-foundEtcd:
		return nil
	case <- timeout:
		return errors.New("unable to reach any etcd server")
	}
}

func RunApiserverChecks(s *options.ServerRunOptions) error {
	timer := new(connectionTimer)
	etcd := new(etcdConnection)
	return WaitForEtcd(s.Etcd.StorageConfig.ServerList, timer, etcd)
}
