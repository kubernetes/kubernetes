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
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
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
	stopCh := make(chan struct{})
	tearDown := func() {
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
	etcdServer, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t, api.Scheme)

	tmpDir, err = ioutil.TempDir("", "kubernetes-kube-apiserver")
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to create temp dir: %v", err)
	}

	s := options.NewServerRunOptions()
	s.InsecureServing.BindPort = 0
	s.SecureServing.BindPort = freePort()
	s.SecureServing.ServerCert.CertDirectory = tmpDir
	s.ServiceClusterIPRange.IP = net.IPv4(10, 0, 0, 0)
	s.ServiceClusterIPRange.Mask = net.CIDRMask(16, 32)
	s.Etcd.StorageConfig = *storageConfig
	s.Etcd.DefaultStorageMediaType = "application/json"
	s.Admission.PluginNames = strings.Split("Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds", ",")
	s.APIEnablement.RuntimeConfig.Set("api/all=true")

	t.Logf("Starting kube-apiserver...")
	runErrCh := make(chan error, 1)
	server, err := app.CreateServerChain(s, stopCh)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to create server chain: %v", err)
	}
	go func(stopCh <-chan struct{}) {
		if err := server.PrepareRun().Run(stopCh); err != nil {
			t.Logf("kube-apiserver exited uncleanly: %v", err)
			runErrCh <- err
		}
	}(stopCh)

	t.Logf("Waiting for /healthz to be ok...")
	client, err := kubernetes.NewForConfig(server.LoopbackClientConfig)
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to create a client: %v", err)
	}
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		select {
		case err := <-runErrCh:
			return false, err
		default:
		}

		result := client.CoreV1().RESTClient().Get().AbsPath("/healthz").Do()
		status := 0
		result.StatusCode(&status)
		if status == 200 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("Failed to wait for /healthz to return ok: %v", err)
	}

	// from here the caller must call tearDown
	return server.LoopbackClientConfig, tearDown, nil
}

// StartTestServerOrDie calls StartTestServer with up to 5 retries on bind error and dies with
// t.Fatal if it does not succeed.
func StartTestServerOrDie(t *testing.T) (*restclient.Config, TearDownFunc) {
	// retry test because the bind might fail due to a race with another process
	// binding to the port. We cannot listen to :0 (then the kernel would give us
	// a port which is free for sure), so we need this workaround.
	for retry := 0; retry < 5 && !t.Failed(); retry++ {
		config, td, err := StartTestServer(t)
		if err == nil {
			return config, td
		}
		if err != nil && !strings.Contains(err.Error(), "bind") {
			break
		}
		t.Logf("Bind error, retrying...")
	}

	t.Fatalf("Failed to launch server")
	return nil, nil
}

func freePort() int {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
