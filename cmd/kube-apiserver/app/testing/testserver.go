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
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	testutil "k8s.io/kubernetes/test/utils"
)

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServerInstanceOptions Instance options the TestServer
type TestServerInstanceOptions struct {
	// DisableStorageCleanup Disable the automatic storage cleanup
	DisableStorageCleanup bool

	// Enable cert-auth for the kube-apiserver
	EnableCertAuth bool
}

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	ClientConfig *restclient.Config        // Rest client config
	ServerOpts   *options.ServerRunOptions // ServerOpts
	TearDownFn   TearDownFunc              // TearDown function
	TmpDir       string                    // Temp Dir used, by the apiserver
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
		EnableCertAuth:        true,
	}
}

// StartTestServer starts a etcd server and kube-apiserver. A rest client config and a tear-down func,
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

	result.TmpDir, err = ioutil.TempDir("", "kubernetes-kube-apiserver")
	if err != nil {
		return result, fmt.Errorf("failed to create temp dir: %v", err)
	}

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)

	s := options.NewServerRunOptions()
	for _, f := range s.Flags().FlagSets {
		fs.AddFlagSet(f)
	}
	s.InsecureServing.BindPort = 0

	s.SecureServing.Listener, s.SecureServing.BindPort, err = createLocalhostListenerOnFreePort()
	if err != nil {
		return result, fmt.Errorf("failed to create listener: %v", err)
	}
	s.SecureServing.ServerCert.CertDirectory = result.TmpDir

	if instanceOptions.EnableCertAuth {
		// create certificates for aggregation and client-cert auth
		proxySigningKey, err := testutil.NewPrivateKey()
		if err != nil {
			return result, err
		}
		proxySigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "front-proxy-ca"}, proxySigningKey)
		if err != nil {
			return result, err
		}
		proxyCACertFile := path.Join(s.SecureServing.ServerCert.CertDirectory, "proxy-ca.crt")
		if err := ioutil.WriteFile(proxyCACertFile, testutil.EncodeCertPEM(proxySigningCert), 0644); err != nil {
			return result, err
		}
		s.Authentication.RequestHeader.ClientCAFile = proxyCACertFile
		clientSigningKey, err := testutil.NewPrivateKey()
		if err != nil {
			return result, err
		}
		clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
		if err != nil {
			return result, err
		}
		clientCACertFile := path.Join(s.SecureServing.ServerCert.CertDirectory, "client-ca.crt")
		if err := ioutil.WriteFile(clientCACertFile, testutil.EncodeCertPEM(clientSigningCert), 0644); err != nil {
			return result, err
		}
		s.Authentication.ClientCert.ClientCA = clientCACertFile
	}

	s.SecureServing.ExternalAddress = s.SecureServing.Listener.Addr().(*net.TCPAddr).IP // use listener addr although it is a loopback device

	pkgPath, err := pkgPath(t)
	if err != nil {
		return result, err
	}
	s.SecureServing.ServerCert.FixtureDirectory = filepath.Join(pkgPath, "testdata")

	s.ServiceClusterIPRanges = "10.0.0.0/16"
	s.Etcd.StorageConfig = *storageConfig
	s.APIEnablement.RuntimeConfig.Set("api/all=true")

	if err := fs.Parse(customFlags); err != nil {
		return result, err
	}
	completedOptions, err := app.Complete(s)
	if err != nil {
		return result, fmt.Errorf("failed to set default ServerRunOptions: %v", err)
	}

	t.Logf("runtime-config=%v", completedOptions.APIEnablement.RuntimeConfig)
	t.Logf("Starting kube-apiserver on port %d...", s.SecureServing.BindPort)
	server, err := app.CreateServerChain(completedOptions, stopCh)
	if err != nil {
		return result, fmt.Errorf("failed to create server chain: %v", err)
	}

	errCh := make(chan error)
	go func(stopCh <-chan struct{}) {
		prepared, err := server.PrepareRun()
		if err != nil {
			errCh <- err
		} else if err := prepared.Run(stopCh); err != nil {
			errCh <- err
		}
	}(stopCh)

	t.Logf("Waiting for /healthz to be ok...")

	client, err := kubernetes.NewForConfig(server.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		return result, fmt.Errorf("failed to create a client: %v", err)
	}

	// wait until healthz endpoint returns ok
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		result := client.CoreV1().RESTClient().Get().AbsPath("/healthz").Do(context.TODO())
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

	// wait until default namespace is created
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		if _, err := client.CoreV1().Namespaces().Get(context.TODO(), "default", metav1.GetOptions{}); err != nil {
			if !errors.IsNotFound(err) {
				t.Logf("Unable to get default namespace: %v", err)
			}
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return result, fmt.Errorf("failed to wait for default namespace to be created: %v", err)
	}

	// from here the caller must call tearDown
	result.ClientConfig = restclient.CopyConfig(server.GenericAPIServer.LoopbackClientConfig)
	result.ClientConfig.QPS = 1000
	result.ClientConfig.Burst = 10000
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

// pkgPath returns the absolute file path to this package's directory. With go
// test, we can just look at the runtime call stack. However, bazel compiles go
// binaries with the -trimpath option so the simple approach fails however we
// can consult environment variables to derive the path.
//
// The approach taken here works for both go test and bazel on the assumption
// that if and only if trimpath is passed, we are running under bazel.
func pkgPath(t Logger) (string, error) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("failed to get current file")
	}

	pkgPath := filepath.Dir(thisFile)

	// If we find bazel env variables, then -trimpath was passed so we need to
	// construct the path from the environment.
	if testSrcdir, testWorkspace := os.Getenv("TEST_SRCDIR"), os.Getenv("TEST_WORKSPACE"); testSrcdir != "" && testWorkspace != "" {
		t.Logf("Detected bazel env varaiables: TEST_SRCDIR=%q TEST_WORKSPACE=%q", testSrcdir, testWorkspace)
		pkgPath = filepath.Join(testSrcdir, testWorkspace, pkgPath)
	}

	// If the path is still not absolute, something other than bazel compiled
	// with -trimpath.
	if !filepath.IsAbs(pkgPath) {
		return "", fmt.Errorf("can't construct an absolute path from %q", pkgPath)
	}

	t.Logf("Resolved testserver package path to: %q", pkgPath)

	return pkgPath, nil
}
