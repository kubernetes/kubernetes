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
	"context"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app"
	"k8s.io/cloud-provider/app/config"
	"k8s.io/cloud-provider/options"
	cliflag "k8s.io/component-base/cli/flag"
)

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	LoopbackClientConfig *restclient.Config // Rest client config using the magic token
	Options              *options.CloudControllerManagerOptions
	Config               *config.CompletedConfig
	TearDownFn           TearDownFunc // TearDown function
	TmpDir               string       // Temp Dir used, by the apiserver
}

// Logger allows t.Testing and b.Testing to be passed to StartTestServer and StartTestServerOrDie
type Logger interface {
	Errorf(format string, args ...interface{})
	Fatalf(format string, args ...interface{})
	Logf(format string, args ...interface{})
}

// StartTestServer starts a cloud-controller-manager. A rest client config and a tear-down func,
// and location of the tmpdir are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporary
// 		 files that because Golang testing's call to os.Exit will not give a stop channel go routine
// 		 enough time to remove temporary files.
func StartTestServer(t Logger, customFlags []string) (result TestServer, err error) {
	stopCh := make(chan struct{})
	configDoneCh := make(chan struct{})
	var capturedConfig config.CompletedConfig
	tearDown := func() {
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

	result.TmpDir, err = ioutil.TempDir("", "cloud-controller-manager")
	if err != nil {
		return result, fmt.Errorf("failed to create temp dir: %v", err)
	}

	s, err := options.NewCloudControllerManagerOptions()
	if err != nil {
		return TestServer{}, err
	}

	cloudInitializer := func(config *config.CompletedConfig) cloudprovider.Interface {
		capturedConfig = *config
		// send signal to indicate the capturedConfig has been properly set
		close(configDoneCh)
		cloudConfig := config.ComponentConfig.KubeCloudShared.CloudProvider
		cloud, err := cloudprovider.InitCloudProvider(cloudConfig.Name, cloudConfig.CloudConfigFile)
		if err != nil {
			t.Fatalf("Cloud provider could not be initialized: %v", err)
		}
		s.SecureServing.ServerCert.CertDirectory = result.TmpDir
		if cloud == nil {
			t.Fatalf("Cloud provider is nil")
		}
		return cloud
	}
	fss := cliflag.NamedFlagSets{}
	command := app.NewCloudControllerManagerCommand(s, cloudInitializer, app.DefaultInitFuncConstructors, fss, stopCh)

	commandArgs := []string{}
	listeners := []net.Listener{}
	disableInsecure := false
	disableSecure := false
	for _, arg := range customFlags {
		if strings.HasPrefix(arg, "--secure-port=") {
			if arg == "--secure-port=0" {
				commandArgs = append(commandArgs, arg)
				disableSecure = true
			}
		} else if strings.HasPrefix(arg, "--port=") {
			if arg == "--port=0" {
				commandArgs = append(commandArgs, arg)
				disableInsecure = true
			}
		} else if strings.HasPrefix(arg, "--cert-dir=") {
			// skip it
		} else {
			commandArgs = append(commandArgs, arg)
		}
	}

	if !disableSecure {
		listener, bindPort, err := createListenerOnFreePort()
		if err != nil {
			return result, fmt.Errorf("failed to create listener: %v", err)
		}
		listeners = append(listeners, listener)
		commandArgs = append(commandArgs, fmt.Sprintf("--secure-port=%d", bindPort))
		commandArgs = append(commandArgs, fmt.Sprintf("--cert-dir=%s", result.TmpDir))

		t.Logf("cloud-controller-manager will listen securely on port %d...", bindPort)
	}
	if !disableInsecure {
		listener, bindPort, err := createListenerOnFreePort()
		if err != nil {
			return result, fmt.Errorf("failed to create listener: %v", err)
		}
		listeners = append(listeners, listener)
		commandArgs = append(commandArgs, fmt.Sprintf("--port=%d", bindPort))

		t.Logf("cloud-controller-manager will listen securely on port %d...", bindPort)
	}
	for _, listener := range listeners {
		listener.Close()
	}

	errCh := make(chan error)
	go func() {
		command.SetArgs(commandArgs)
		if err := command.Execute(); err != nil {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case <-configDoneCh:

	case err := <-errCh:
		return result, err
	}

	t.Logf("Waiting for /healthz to be ok...")
	client, err := kubernetes.NewForConfig(capturedConfig.LoopbackClientConfig)
	if err != nil {
		return result, fmt.Errorf("failed to create a client: %v", err)
	}
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

	// from here the caller must call tearDown
	result.LoopbackClientConfig = capturedConfig.LoopbackClientConfig
	result.Options = s
	result.Config = &capturedConfig
	result.TearDownFn = tearDown

	return result, nil
}

// StartTestServerOrDie calls StartTestServer t.Fatal if it does not succeed.
func StartTestServerOrDie(t Logger, flags []string) *TestServer {
	result, err := StartTestServer(t, flags)
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
