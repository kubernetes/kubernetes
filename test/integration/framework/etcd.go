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

package framework

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"strconv"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/transport"
	"go.etcd.io/etcd/server/v3/embed"
	"k8s.io/klog/v2"
)

var etcdURL = ""

// getAvailablePort returns a TCP port that is available for binding.
func getAvailablePorts(count int) ([]int, error) {
	ports := []int{}
	for i := 0; i < count; i++ {
		l, err := net.Listen("tcp", ":0")
		if err != nil {
			return nil, fmt.Errorf("could not bind to a port: %v", err)
		}
		// It is possible but unlikely that someone else will bind this port before we get a chance to use it.
		defer l.Close()
		ports = append(ports, l.Addr().(*net.TCPAddr).Port)
	}
	return ports, nil
}

// startEtcd executes an embedded etcd instance.
func startEtcd() (func(), error) {
	dataDir := "integration_test_etcd_data"
	etcdDir, err := ioutil.TempDir(os.TempDir(), dataDir)
	if err != nil {
		return nil, fmt.Errorf("unable to make temp etcd data dir %s: %v", dataDir, err)
	}
	klog.Infof("storing etcd data in: %v", etcdDir)
	os.Chmod(etcdDir, 0700)

	cfg := embed.NewConfig()
	cfg.Dir = etcdDir
	cfg.Name = "etcd-integration"

	cfg.UnsafeNoFsync = true

	// use standard ports if they are free
	ports := []int{2379, 2380}
	conn, err := net.Dial("tcp", "127.0.0.1:2379")
	if err == nil {
		klog.Infof("etcd ports are not free at 127.0.0.1:2379")
		conn.Close()
		ports, err = getAvailablePorts(2)
		if err != nil {
			return nil, err
		}
	}
	clientURL := url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(ports[0]))}
	etcdURL = clientURL.String()
	peerURL := url.URL{Scheme: "http", Host: net.JoinHostPort("127.0.0.1", strconv.Itoa(ports[1]))}

	cfg.LPUrls = []url.URL{peerURL}
	cfg.APUrls = []url.URL{peerURL}
	cfg.LCUrls = []url.URL{clientURL}
	cfg.ACUrls = []url.URL{clientURL}
	cfg.ClientAutoTLS = false
	cfg.PeerAutoTLS = false
	cfg.AuthToken = ""
	cfg.ClientTLSInfo = transport.TLSInfo{}
	cfg.PeerTLSInfo = transport.TLSInfo{}
	cfg.InitialCluster = cfg.InitialClusterFromName(cfg.Name)
	cfg.LogLevel = "debug"

	e, err := embed.StartEtcd(cfg)
	if err != nil {
		return nil, err
	}
	stop := func() {
		e.Close()
		err = os.RemoveAll(etcdDir)
		if err != nil {
			klog.Warningf("error during etcd cleanup: %v", err)
		}
	}

	select {
	case <-e.Server.ReadyNotify():
	case <-time.After(60 * time.Second):
		e.Server.Stop() // trigger a shutdown
		return nil, err
	}

	return stop, nil
}

// EtcdMain starts an etcd instance before running tests.
func EtcdMain(tests func() int) {
	stop, err := startEtcd()
	if err != nil {
		klog.Fatalf("cannot run integration tests: unable to start etcd: %v", err)
	}
	result := tests()
	stop() // Don't defer this. See os.Exit documentation.
	os.Exit(result)
}

// GetEtcdURL returns the URL of the etcd instance started by EtcdMain.
func GetEtcdURL() string {
	return etcdURL
}
