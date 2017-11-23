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
	"testing"
	"time"

	pflag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	ClientConfig *restclient.Config        // Rest client config
	ServerOpts   *options.ServerRunOptions // ServerOpts
	TearDownFn   TearDownFunc              // TearDown function
	TmpDir       string                    // Temp Dir used, by the apiserver
}

// StartTestServer starts a etcd server and kube-apiserver. A rest client config and a tear-down func,
// and location of the tmpdir are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporariy
// 		 files that becaues Golang testing's call to os.Exit will not give a stop channel go routine
// 		 enough time to remove temporariy files.
func StartTestServer(t *testing.T, customFlags []string, storageConfig *storagebackend.Config) (result TestServer, err error) {

	// TODO : Remove TrackStorageCleanup below when PR
	// https://github.com/kubernetes/kubernetes/pull/50690
	// merges as that shuts down storage properly
	registry.TrackStorageCleanup()

	stopCh := make(chan struct{})
	tearDown := func() {
		registry.CleanupStorage()
		close(stopCh)
		if len(result.TmpDir) != 0 {
			os.RemoveAll(result.TmpDir)
		}
	}
	defer func() {
		if result.TearDownFn == nil {
			tearDown()
		}
	}()

	result.TmpDir, err = ioutil.TempDir("", "kubernetes-kube-apiserver")
	if err != nil {
		return result, fmt.Errorf("failed to create temp dir: %v", err)
	}

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)

	s := options.NewServerRunOptions()
	s.AddFlags(fs)

	s.InsecureServing.BindPort = 0

	s.SecureServing.Listener, s.SecureServing.BindPort, err = createListenerOnFreePort()
	if err != nil {
		return result, fmt.Errorf("failed to create listener: %v", err)
	}
	s.SecureServing.ServerCert.CertDirectory = result.TmpDir
	s.ServiceClusterIPRange.IP = net.IPv4(10, 0, 0, 0)
	s.ServiceClusterIPRange.Mask = net.CIDRMask(16, 32)
	s.Etcd.StorageConfig = *storageConfig
	s.APIEnablement.RuntimeConfig.Set("api/all=true")

	fs.Parse(customFlags)

	t.Logf("Starting kube-apiserver on port %d...", s.SecureServing.BindPort)
	server, err := app.CreateServerChain(s, stopCh)
	if err != nil {
		return result, fmt.Errorf("failed to create server chain: %v", err)

	}
	go func(stopCh <-chan struct{}) {
		if err := server.PrepareRun().Run(stopCh); err != nil {
			t.Errorf("kube-apiserver failed run: %v", err)
		}
	}(stopCh)

	t.Logf("Waiting for /healthz to be ok...")

	client, err := kubernetes.NewForConfig(server.LoopbackClientConfig)
	if err != nil {
		return result, fmt.Errorf("failed to create a client: %v", err)
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
		return result, fmt.Errorf("failed to wait for /healthz to return ok: %v", err)
	}

	// from here the caller must call tearDown
	result.ClientConfig = server.LoopbackClientConfig
	result.ServerOpts = s
	result.TearDownFn = tearDown

	return result, nil
}

// StartTestServerOrDie calls StartTestServer t.Fatal if it does not succeed.
func StartTestServerOrDie(t *testing.T, flags []string, storageConfig *storagebackend.Config) *TestServer {

	result, err := StartTestServer(t, flags, storageConfig)
	if err == nil {
		return &result
	}

	t.Fatalf("failed to launch server: %v", err)
	return nil
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
