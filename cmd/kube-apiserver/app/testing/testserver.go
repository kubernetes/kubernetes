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

package testing

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// StartTestServer starts a etcd server and kube-apiserver. A rest client config and a tear-down func
// are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporariy
// 		 files that becaues Golang testing's call to os.Exit will not give a stop channel go routine
// 		 enough time to remove temporariy files.
func StartTestServer(t *testing.T) (result *restclient.Config, tearDownForCaller TearDownFunc, err error) {
	var tmpDir string
	var etcdServer *etcdtesting.EtcdTestServer

	// TODO : Remove TrackStorageCleanup below when PR
	// https://github.com/kubernetes/kubernetes/pull/50690
	// merges as that shuts down storage properly
	registry.TrackStorageCleanup()

	stopCh := make(chan struct{})
	tearDown := func() {
		registry.CleanupStorage()
		close(stopCh)
		if etcdServer != nil {
			etcdServer.Terminate(t)
		}
		if len(tmpDir) != 0 {
			os.RemoveAll(tmpDir)
		}
	}
	defer func() {
		if tearDownForCaller == nil {
			tearDown()
		}
	}()

	t.Logf("Starting etcd...")
	etcdServer, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	tmpDir, err = ioutil.TempDir("", "kubernetes-kube-apiserver")
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create temp dir: %v", err)
	}

	s := options.NewServerRunOptions()
	s.InsecureServing.BindPort = 0
	s.SecureServing.Listener, s.SecureServing.BindPort, err = createListenerOnFreePort()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create listener: %v", err)
	}

	s.SecureServing.ServerCert.CertDirectory = tmpDir
	s.ServiceClusterIPRange.IP = net.IPv4(10, 0, 0, 0)
	s.ServiceClusterIPRange.Mask = net.CIDRMask(16, 32)
	s.Etcd.StorageConfig = *storageConfig
	s.Etcd.DefaultStorageMediaType = "application/json"
	s.Admission.PluginNames = strings.Split("Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds", ",")
	s.APIEnablement.RuntimeConfig.Set("api/all=true")

	t.Logf("Starting kube-apiserver on port %d...", s.SecureServing.BindPort)
	server, err := app.CreateServerChain(s, stopCh)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create server chain: %v", err)
	}
	go func(stopCh <-chan struct{}) {
		if err := server.PrepareRun().Run(stopCh); err != nil {
			t.Errorf("kube-apiserver failed run: %v", err)
		}
	}(stopCh)

	t.Logf("Waiting for /healthz to be ok...")
	client, err := kubernetes.NewForConfig(server.LoopbackClientConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create a client: %v", err)
	}
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		result := client.CoreV1().RESTClient().Get().AbsPath("/healthz").Do()
		status := 0
		result.StatusCode(&status)
		if status == 200 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to wait for /healthz to return ok: %v", err)
	}

	// from here the caller must call tearDown
	return server.LoopbackClientConfig, tearDown, nil
}

// StartTestServerOrDie calls StartTestServer with up to 5 retries on bind error and dies with
// t.Fatal if it does not succeed.
func StartTestServerOrDie(t *testing.T) (*restclient.Config, TearDownFunc) {
	config, td, err := StartTestServer(t)
	if err == nil {
		return config, td
	}

	t.Fatalf("Failed to launch server: %v", err)
	return nil, nil
}

func createListenerOnFreePort() (net.Listener, int, error) {
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		return nil, 0, err
	}

	// get port
	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		ln.Close()
		return nil, 0, fmt.Errorf("invalid listen address: %q", ln.Addr().String())
	}

	return ln, tcpAddr.Port, nil
}
