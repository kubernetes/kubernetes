/*
Copyright 2018 The Kubernetes Authors.

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
	"path"
	"runtime"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
)

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServerInstanceOptions Instance options the TestServer
type TestServerInstanceOptions struct {
	// DisableStorageCleanup Disable the automatic storage cleanup
	DisableStorageCleanup bool
}

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	ClientConfig *restclient.Config                              // Rest client config
	ServerOpts   *options.CustomResourceDefinitionsServerOptions // ServerOpts
	TearDownFn   TearDownFunc                                    // TearDown function
	TmpDir       string                                          // Temp Dir used, by the apiserver
}

// Logger allows t.Testing and b.Testing to be passed to StartTestServer and StartTestServerOrDie
type Logger interface {
	Errorf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})
	Logf(format string, args ...interface{})
}

// NewDefaultTestServerOptions Default options for TestServer instances
func NewDefaultTestServerOptions() *TestServerInstanceOptions {
	return &TestServerInstanceOptions{
		DisableStorageCleanup: false,
	}
}

// StartTestServer starts a apiextensions-apiserver. A rest client config and a tear-down func,
// and location of the tmpdir are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporary
// 		 files that because Golang testing's call to os.Exit will not give a stop channel go routine
// 		 enough time to remove temporary files.
func StartTestServer(t Logger, instanceOptions *TestServerInstanceOptions, customFlags []string, storageConfig *storagebackend.Config) (result TestServer, err error) {
	if instanceOptions == nil {
		instanceOptions = NewDefaultTestServerOptions()
	}

	// TODO : Remove TrackStorageCleanup below when PR
	// https://github.com/kubernetes/kubernetes/pull/50690
	// merges as that shuts down storage properly
	if !instanceOptions.DisableStorageCleanup {
		registry.TrackStorageCleanup()
	}

	stopCh := make(chan struct{})
	tearDown := func() {
		if !instanceOptions.DisableStorageCleanup {
			registry.CleanupStorage()
		}
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

	result.TmpDir, err = ioutil.TempDir("", "apiextensions-apiserver")
	if err != nil {
		return result, fmt.Errorf("failed to create temp dir: %v", err)
	}

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)

	s := options.NewCustomResourceDefinitionsServerOptions(os.Stdout, os.Stderr)
	s.AddFlags(fs)

	s.RecommendedOptions.SecureServing.Listener, s.RecommendedOptions.SecureServing.BindPort, err = createLocalhostListenerOnFreePort()
	if err != nil {
		return result, fmt.Errorf("failed to create listener: %v", err)
	}
	s.RecommendedOptions.SecureServing.ServerCert.CertDirectory = result.TmpDir
	s.RecommendedOptions.SecureServing.ExternalAddress = s.RecommendedOptions.SecureServing.Listener.Addr().(*net.TCPAddr).IP // use listener addr although it is a loopback device

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return result, fmt.Errorf("failed to get current file")
	}
	s.RecommendedOptions.SecureServing.ServerCert.FixtureDirectory = path.Join(path.Dir(thisFile), "testdata")

	if storageConfig != nil {
		s.RecommendedOptions.Etcd.StorageConfig = *storageConfig
	}
	s.APIEnablement.RuntimeConfig.Set("api/all=true")

	fs.Parse(customFlags)

	if err := s.Complete(); err != nil {
		return result, fmt.Errorf("failed to set default options: %v", err)
	}
	if err := s.Validate(); err != nil {
		return result, fmt.Errorf("failed to validate options: %v", err)
	}

	t.Logf("runtime-config=%v", s.APIEnablement.RuntimeConfig)
	t.Logf("Starting apiextensions-apiserver on port %d...", s.RecommendedOptions.SecureServing.BindPort)

	config, err := s.Config()
	if err != nil {
		return result, fmt.Errorf("failed to create config from options: %v", err)
	}
	server, err := config.Complete().New(genericapiserver.NewEmptyDelegate())
	if err != nil {
		return result, fmt.Errorf("failed to create server: %v", err)
	}

	errCh := make(chan error)
	go func(stopCh <-chan struct{}) {
		if err := server.GenericAPIServer.PrepareRun().Run(stopCh); err != nil {
			errCh <- err
		}
	}(stopCh)

	t.Logf("Waiting for /healthz to be ok...")

	client, err := kubernetes.NewForConfig(server.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		return result, fmt.Errorf("failed to create a client: %v", err)
	}
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		select {
		case err := <-errCh:
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
		return result, fmt.Errorf("failed to wait for /healthz to return ok: %v", err)
	}

	// from here the caller must call tearDown
	result.ClientConfig = server.GenericAPIServer.LoopbackClientConfig
	result.ServerOpts = s
	result.TearDownFn = tearDown

	return result, nil
}

// StartTestServerOrDie calls StartTestServer t.Fatal if it does not succeed.
func StartTestServerOrDie(t Logger, instanceOptions *TestServerInstanceOptions, flags []string, storageConfig *storagebackend.Config) *TestServer {
	result, err := StartTestServer(t, instanceOptions, flags, storageConfig)
	if err == nil {
		return &result
	}

	t.Fatalf("failed to launch server: %v", err)
	return nil
}

func createLocalhostListenerOnFreePort() (net.Listener, int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
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
