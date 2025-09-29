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
	"net"
	"testing"
	"time"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/wait"
	utilcompatibility "k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/compatibility"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
)

func init() {
	// If instantiated more than once or together with other servers, the
	// servers would try to modify the global logging state. This must get
	// ignored during testing.
	logsapi.ReapplyHandling = logsapi.ReapplyHandlingIgnoreUnchanged
}

// TearDownFunc is to be called to tear down a test server.
type TearDownFunc func()

// TestServer return values supplied by kube-test-ApiServer
type TestServer struct {
	LoopbackClientConfig *restclient.Config // Rest client config using the magic token
	Options              *options.KubeControllerManagerOptions
	Config               *kubecontrollerconfig.Config
	TearDownFn           TearDownFunc // TearDown function
	TmpDir               string       // Temp Dir used, by the apiserver
}

// StartTestServer starts a kube-controller-manager. A rest client config and a tear-down func,
// and location of the tmpdir are returned.
//
// Note: we return a tear-down func instead of a stop channel because the later will leak temporary
// files that because Golang testing's call to os.Exit will not give a stop channel go routine
// enough time to remove temporary files.
func StartTestServer(t *testing.T, ctx context.Context, customFlags []string) (result TestServer, err error) {
	logger := klog.FromContext(ctx)
	ctx, cancel := context.WithCancel(ctx)
	var errCh chan error
	tearDown := func() {
		cancel()

		// If the kube-controller-manager was started, let's wait for
		// it to shutdown cleanly.
		if errCh != nil {
			err, ok := <-errCh
			if ok && err != nil {
				logger.Error(err, "Failed to shutdown test server cleanly")
			}
		}
	}
	defer func() {
		if result.TearDownFn == nil {
			tearDown()
		}
	}()

	result.TmpDir = t.TempDir()

	fs := pflag.NewFlagSet("test", pflag.PanicOnError)

	s, err := options.NewKubeControllerManagerOptions()
	if err != nil {
		return TestServer{}, err
	}
	// set up new instance of ComponentGlobalsRegistry instead of using the DefaultComponentGlobalsRegistry to avoid contention in parallel tests.
	featureGate := utilfeature.DefaultMutableFeatureGate.DeepCopy()
	effectiveVersion := utilcompatibility.DefaultKubeEffectiveVersionForTest()
	effectiveVersion.SetEmulationVersion(featureGate.EmulationVersion())
	componentGlobalsRegistry := compatibility.NewComponentGlobalsRegistry()
	if err := componentGlobalsRegistry.Register(compatibility.DefaultKubeComponent, effectiveVersion, featureGate); err != nil {
		return result, err
	}
	s.ComponentGlobalsRegistry = componentGlobalsRegistry

	all, disabled, aliases := app.KnownControllers(), app.ControllersDisabledByDefault(), app.ControllerAliases()
	namedFlagSets := s.Flags(all, disabled, aliases)
	for _, f := range namedFlagSets.FlagSets {
		fs.AddFlagSet(f)
	}
	fs.Parse(customFlags)
	s.ParsedFlags = &namedFlagSets

	if err := s.ComponentGlobalsRegistry.Set(); err != nil {
		return result, err
	}
	// If the local ComponentGlobalsRegistry is changed by the flags,
	// we need to copy the new feature values back to the DefaultFeatureGate because most feature checks still use the DefaultFeatureGate.
	if !featureGate.EmulationVersion().EqualTo(utilfeature.DefaultMutableFeatureGate.EmulationVersion()) {
		featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultMutableFeatureGate, effectiveVersion.EmulationVersion())
	}
	featureOverrides := map[featuregate.Feature]bool{}
	for f := range utilfeature.DefaultMutableFeatureGate.GetAll() {
		if featureGate.Enabled(f) != utilfeature.DefaultFeatureGate.Enabled(f) {
			featureOverrides[f] = featureGate.Enabled(f)
		}
	}
	if len(featureOverrides) > 0 {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featureOverrides)
	}

	if s.SecureServing.BindPort != 0 {
		s.SecureServing.Listener, s.SecureServing.BindPort, err = createListenerOnFreePort()
		if err != nil {
			return result, fmt.Errorf("failed to create listener: %v", err)
		}
		s.SecureServing.ServerCert.CertDirectory = result.TmpDir

		logger.Info("kube-controller-manager will listen securely", "port", s.SecureServing.BindPort)
	}

	config, err := s.Config(ctx, all, disabled, aliases)
	if err != nil {
		return result, fmt.Errorf("failed to create config from options: %v", err)
	}

	errCh = make(chan error)
	go func(ctx context.Context) {
		defer close(errCh)

		if err := app.Run(ctx, config.Complete()); err != nil {
			errCh <- err
		}
	}(ctx)

	logger.Info("Waiting for /healthz to be ok...")
	client, err := kubernetes.NewForConfig(config.LoopbackClientConfig)
	if err != nil {
		return result, fmt.Errorf("failed to create a client: %v", err)
	}
	err = wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
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
	result.LoopbackClientConfig = config.LoopbackClientConfig
	result.Options = s
	result.Config = config
	result.TearDownFn = tearDown

	return result, nil
}

// StartTestServerOrDie calls StartTestServer t.Fatal if it does not succeed.
func StartTestServerOrDie(t *testing.T, ctx context.Context, flags []string) *TestServer {
	result, err := StartTestServer(t, ctx, flags)
	if err != nil {
		t.Fatalf("failed to launch server: %v", err)
	}
	return &result
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
