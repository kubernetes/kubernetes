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
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"time"

	"github.com/spf13/pflag"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	serveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storageversion"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/klog/v2"
	"k8s.io/kube-aggregator/pkg/apiserver"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/test/utils"
)

// This key is for testing purposes only and is not considered secure.
const ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZmTmUhuanLjPA2CLquXivuwBDHTt5XYwgIr/kA1LtRoAoGCCqGSM49
AwEHoUQDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp/C/ASqiIGUeeKQtX0
/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END EC PRIVATE KEY-----`

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServerInstanceOptions Instance options the TestServer
type TestServerInstanceOptions struct {
	// Enable cert-auth for the kube-apiserver
	EnableCertAuth bool
	// Wrap the storage version interface of the created server's generic server.
	StorageVersionWrapFunc func(storageversion.Manager) storageversion.Manager
}

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	ClientConfig      *restclient.Config        // Rest client config
	ServerOpts        *options.ServerRunOptions // ServerOpts
	TearDownFn        TearDownFunc              // TearDown function
	TmpDir            string                    // Temp Dir used, by the apiserver
	EtcdClient        *clientv3.Client          // used by tests that need to check data migrated from APIs that are no longer served
	EtcdStoragePrefix string                    // storage prefix in etcd
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
		EnableCertAuth: true,
	}
}

// StartTestServer starts a etcd server and kube-apiserver. A rest client config and a tear-down func,
// and location of the tmpdir are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporary
//
//	files that because Golang testing's call to os.Exit will not give a stop channel go routine
//	enough time to remove temporary files.
func StartTestServer(t Logger, instanceOptions *TestServerInstanceOptions, customFlags []string, storageConfig *storagebackend.Config) (result TestServer, err error) {
	if instanceOptions == nil {
		instanceOptions = NewDefaultTestServerOptions()
	}

	result.TmpDir, err = os.MkdirTemp("", "kubernetes-kube-apiserver")
	if err != nil {
		return result, fmt.Errorf("failed to create temp dir: %v", err)
	}

	stopCh := make(chan struct{})
	var errCh chan error
	tearDown := func() {
		// Closing stopCh is stopping apiserver and cleaning up
		// after itself, including shutting down its storage layer.
		close(stopCh)

		// If the apiserver was started, let's wait for it to
		// shutdown clearly.
		if errCh != nil {
			err, ok := <-errCh
			if ok && err != nil {
				klog.Errorf("Failed to shutdown test server clearly: %v", err)
			}
		}
		os.RemoveAll(result.TmpDir)
	}
	defer func() {
		if result.TearDownFn == nil {
			tearDown()
		}
	}()

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)

	s := options.NewServerRunOptions()
	for _, f := range s.Flags().FlagSets {
		fs.AddFlagSet(f)
	}

	s.SecureServing.Listener, s.SecureServing.BindPort, err = createLocalhostListenerOnFreePort()
	if err != nil {
		return result, fmt.Errorf("failed to create listener: %v", err)
	}
	s.SecureServing.ServerCert.CertDirectory = result.TmpDir

	if instanceOptions.EnableCertAuth {
		// set up default headers for request header auth
		reqHeaders := serveroptions.NewDelegatingAuthenticationOptions()
		s.Authentication.RequestHeader = &reqHeaders.RequestHeader

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
		if err := os.WriteFile(proxyCACertFile, testutil.EncodeCertPEM(proxySigningCert), 0644); err != nil {
			return result, err
		}
		s.Authentication.RequestHeader.ClientCAFile = proxyCACertFile

		// give the kube api server an "identity" it can use to for request header auth
		// so that aggregated api servers can understand who the calling user is
		s.Authentication.RequestHeader.AllowedNames = []string{"ash", "misty", "brock"}
		// make a client certificate for the api server - common name has to match one of our defined names above
		tenThousandHoursLater := time.Now().Add(10_000 * time.Hour)
		clientCrtOfAPIServer, signer, err := pkiutil.NewCertAndKey(proxySigningCert, proxySigningKey, &pkiutil.CertConfig{
			Config: cert.Config{
				CommonName: "misty",
				Usages: []x509.ExtKeyUsage{
					x509.ExtKeyUsageClientAuth,
				},
			},
			NotAfter:           &tenThousandHoursLater,
			PublicKeyAlgorithm: x509.ECDSA,
		})
		if err != nil {
			return result, err
		}
		if err := pkiutil.WriteCertAndKey(s.SecureServing.ServerCert.CertDirectory, "misty-crt", clientCrtOfAPIServer, signer); err != nil {
			return result, err
		}
		s.ProxyClientKeyFile = path.Join(s.SecureServing.ServerCert.CertDirectory, "misty-crt.key")
		s.ProxyClientCertFile = path.Join(s.SecureServing.ServerCert.CertDirectory, "misty-crt.crt")

		clientSigningKey, err := testutil.NewPrivateKey()
		if err != nil {
			return result, err
		}
		clientSigningCert, err := cert.NewSelfSignedCACert(cert.Config{CommonName: "client-ca"}, clientSigningKey)
		if err != nil {
			return result, err
		}
		clientCACertFile := path.Join(s.SecureServing.ServerCert.CertDirectory, "client-ca.crt")
		if err := os.WriteFile(clientCACertFile, testutil.EncodeCertPEM(clientSigningCert), 0644); err != nil {
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

	saSigningKeyFile, err := os.CreateTemp("/tmp", "insecure_test_key")
	if err != nil {
		t.Fatalf("create temp file failed: %v", err)
	}
	defer os.RemoveAll(saSigningKeyFile.Name())
	if err = os.WriteFile(saSigningKeyFile.Name(), []byte(ecdsaPrivateKey), 0666); err != nil {
		t.Fatalf("write file %s failed: %v", saSigningKeyFile.Name(), err)
	}
	s.ServiceAccountSigningKeyFile = saSigningKeyFile.Name()
	s.Authentication.ServiceAccounts.Issuers = []string{"https://foo.bar.example.com"}
	s.Authentication.ServiceAccounts.KeyFiles = []string{saSigningKeyFile.Name()}

	completedOptions, err := app.Complete(s)
	if err != nil {
		return result, fmt.Errorf("failed to set default ServerRunOptions: %v", err)
	}

	if errs := completedOptions.Validate(); len(errs) != 0 {
		return result, fmt.Errorf("failed to validate ServerRunOptions: %v", utilerrors.NewAggregate(errs))
	}

	t.Logf("runtime-config=%v", completedOptions.APIEnablement.RuntimeConfig)
	t.Logf("Starting kube-apiserver on port %d...", s.SecureServing.BindPort)
	server, err := app.CreateServerChain(completedOptions)
	if err != nil {
		return result, fmt.Errorf("failed to create server chain: %v", err)
	}
	if instanceOptions.StorageVersionWrapFunc != nil {
		server.GenericAPIServer.StorageVersionManager = instanceOptions.StorageVersionWrapFunc(server.GenericAPIServer.StorageVersionManager)
	}

	errCh = make(chan error)
	go func(stopCh <-chan struct{}) {
		defer close(errCh)
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
	err = wait.Poll(100*time.Millisecond, time.Minute, func() (bool, error) {
		select {
		case err := <-errCh:
			return false, err
		default:
		}

		req := client.CoreV1().RESTClient().Get().AbsPath("/healthz")
		// The storage version bootstrap test wraps the storage version post-start
		// hook, so the hook won't become health when the server bootstraps
		if instanceOptions.StorageVersionWrapFunc != nil {
			// We hardcode the param instead of having a new instanceOptions field
			// to avoid confusing users with more options.
			storageVersionCheck := fmt.Sprintf("poststarthook/%s", apiserver.StorageVersionPostStartHookName)
			req.Param("exclude", storageVersionCheck)
		}
		result := req.Do(context.TODO())
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

	tlsInfo := transport.TLSInfo{
		CertFile:      storageConfig.Transport.CertFile,
		KeyFile:       storageConfig.Transport.KeyFile,
		TrustedCAFile: storageConfig.Transport.TrustedCAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return result, err
	}
	etcdConfig := clientv3.Config{
		Endpoints:   storageConfig.Transport.ServerList,
		DialTimeout: 20 * time.Second,
		DialOptions: []grpc.DialOption{
			grpc.WithBlock(), // block until the underlying connection is up
		},
		TLS: tlsConfig,
	}
	etcdClient, err := clientv3.New(etcdConfig)
	if err != nil {
		return result, err
	}

	// from here the caller must call tearDown
	result.ClientConfig = restclient.CopyConfig(server.GenericAPIServer.LoopbackClientConfig)
	result.ClientConfig.QPS = 1000
	result.ClientConfig.Burst = 10000
	result.ServerOpts = s
	result.TearDownFn = func() {
		tearDown()
		etcdClient.Close()
	}
	result.EtcdClient = etcdClient
	result.EtcdStoragePrefix = storageConfig.Prefix

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
