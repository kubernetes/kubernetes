/*
Copyright 2023 The Kubernetes Authors.

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

package app

import (
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"context"
	"errors"
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	cloudprovider "k8s.io/cloud-provider"
	cloudcontrollerconfig "k8s.io/cloud-provider/app/config"
	"k8s.io/cloud-provider/names"
	"k8s.io/cloud-provider/options"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics"
	metricsfeatures "k8s.io/component-base/metrics/features"
	controllermanagerapp "k8s.io/controller-manager/app"
	"k8s.io/controller-manager/pkg/clientbuilder"
	cmfeatures "k8s.io/controller-manager/pkg/features"
	"net/http"
	"net/http/httptest"
	"time"
)

func TestCloudControllerNamesConsistency(t *testing.T) {
	controllerNameRegexp := regexp.MustCompile("^[a-z]([-a-z]*[a-z])?$")

	for name := range DefaultInitFuncConstructors {
		if !controllerNameRegexp.MatchString(name) {
			t.Errorf("name consistency check failed: controller %q must consist of lower case alphabetic characters or '-', and must start and end with an alphabetic character", name)
		}
		if !strings.HasSuffix(name, "-controller") {
			t.Errorf("name consistency check failed: controller %q must have \"-controller\" suffix", name)
		}
	}
}

func TestCloudControllerNamesDeclaration(t *testing.T) {
	declaredControllers := sets.New(
		names.CloudNodeController,
		names.ServiceLBController,
		names.NodeRouteController,
		names.CloudNodeLifecycleController,
	)

	for name := range DefaultInitFuncConstructors {
		if !declaredControllers.Has(name) {
			t.Errorf("name declaration check failed: controller name %q should be declared in  \"controller_names.go\" and added to this test", name)
		}
	}
}

func TestNativeHistogramsFeatureGateApplied(t *testing.T) {
	testCases := []struct {
		name       string
		newCommand func(t *testing.T) *cobra.Command
	}{
		{
			name: "NewCloudControllerManagerCommand",
			newCommand: func(t *testing.T) *cobra.Command {
				s, err := options.NewCloudControllerManagerOptions()
				if err != nil {
					t.Fatal(err)
				}
				return NewCloudControllerManagerCommand(s, nil, DefaultInitFuncConstructors, names.CCMControllerAliases(), cliflag.NamedFlagSets{}, wait.NeverStop)
			},
		},
		{
			name: "CommandBuilder",
			newCommand: func(t *testing.T) *cobra.Command {
				cb := NewBuilder()
				cb.RegisterDefaultControllers()
				return cb.BuildCommand()
			},
		},
	}

	originalReapplyHandling := logsapi.ReapplyHandling
	logsapi.ReapplyHandling = logsapi.ReapplyHandlingIgnoreUnchanged
	t.Cleanup(func() { logsapi.ReapplyHandling = originalReapplyHandling })

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Registered before SetFeatureGateDuringTest so that it runs after the gate is restored.
			t.Cleanup(func() { metricsfeatures.ApplyFeatureGates(utilfeature.DefaultFeatureGate) })
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, metricsfeatures.NativeHistograms, false)
			metricsfeatures.ApplyFeatureGates(utilfeature.DefaultFeatureGate)
			if histogramIsNative(t) {
				t.Fatal("histogram is native before the command has run")
			}

			// The kubeconfig does not exist, so the command fails in Config,
			// after it has applied the feature gates.
			cmd := tc.newCommand(t)
			cmd.SetArgs([]string{
				"--kubeconfig=" + filepath.Join(t.TempDir(), "missing-kubeconfig"),
				"--feature-gates=NativeHistograms=true",
			})
			if err := cmd.Execute(); err == nil {
				t.Fatal("expected the command to fail")
			}

			if !histogramIsNative(t) {
				t.Error("histogram is not native although the NativeHistograms feature gate is enabled")
			}
		})
	}
}

func histogramIsNative(t *testing.T) bool {
	t.Helper()
	h := metrics.NewHistogram(&metrics.HistogramOpts{Name: "test_histogram", Help: "test"})
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(h)
	h.Observe(0.5)
	mfs, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() == "test_histogram" {
			return mf.GetMetric()[0].GetHistogram().Schema != nil
		}
	}
	t.Fatal("histogram not gathered")
	return false
}

// bareClientBuilder implements only the cloud provider's builder interface,
// standing in for an out-of-tree builder without discovery clients.
type bareClientBuilder struct {
	config *restclient.Config
}

func (b bareClientBuilder) Config(name string) (*restclient.Config, error) {
	cfg := *b.config
	return restclient.AddUserAgent(&cfg, name), nil
}
func (b bareClientBuilder) ConfigOrDie(name string) *restclient.Config {
	cfg, _ := b.Config(name)
	return cfg
}
func (b bareClientBuilder) Client(name string) (clientset.Interface, error) {
	return clientset.NewForConfig(b.ConfigOrDie(name))
}
func (b bareClientBuilder) ClientOrDie(name string) clientset.Interface {
	return clientset.NewForConfigOrDie(b.ConfigOrDie(name))
}

func TestGatedClientBuilders(t *testing.T) {
	server := serviceAccountServer(t)

	for _, useServiceAccounts := range []bool{false, true} {
		t.Run(fmt.Sprintf("serviceAccountCredentials=%v", useServiceAccounts), func(t *testing.T) {
			kubeconfig := &restclient.Config{Host: server.URL}
			c := &cloudcontrollerconfig.Config{Kubeconfig: kubeconfig, Client: clientset.NewForConfigOrDie(kubeconfig)}
			c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials = useServiceAccounts

			wraps := 0
			gate := func(rt http.RoundTripper) http.RoundTripper {
				wraps++
				return rt
			}

			rootClientBuilder, controllerClientBuilder := gatedClientBuilders(c.Complete(), gate)

			// Building a client applies the gate to its transport.
			rootClientBuilder.ClientOrDie("root")
			controllerClientBuilder.ClientOrDie("controller")
			if wraps != 2 {
				t.Errorf("gate applied %d times building a root and a controller client, want 2", wraps)
			}
			if kubeconfig.WrapTransport != nil {
				t.Errorf("the controller manager's kubeconfig must not be modified")
			}

			if useServiceAccounts {
				if _, ok := controllerClientBuilder.(*clientbuilder.DynamicControllerClientBuilder); !ok {
					t.Errorf("controller builder is %T, want the service account builder", controllerClientBuilder)
				}
			} else if got, ok := controllerClientBuilder.(clientbuilder.SimpleControllerClientBuilder); !ok || got != rootClientBuilder {
				t.Errorf("controller builder is %T, want the root builder", controllerClientBuilder)
			}
		})
	}
}

func TestControllerManagerClientBuilder(t *testing.T) {
	cfg := &restclient.Config{Host: "https://localhost:6443"}

	simple := clientbuilder.SimpleControllerClientBuilder{ClientConfig: cfg}
	if got := controllerManagerClientBuilder(simple); got != cloudprovider.ControllerClientBuilder(simple) {
		t.Errorf("a controller manager builder must be returned as is, got %#v", got)
	}

	adapted := controllerManagerClientBuilder(bareClientBuilder{config: cfg})
	if _, err := adapted.DiscoveryClient("controller"); err != nil {
		t.Errorf("adapted builder cannot build a discovery client: %v", err)
	}
	if adapted.DiscoveryClientOrDie("controller") == nil {
		t.Errorf("adapted builder returned a nil discovery client")
	}
}

// notLeadingElection returns an election whose elector has not acquired its
// lease, so its write gate is closed.
func notLeadingElection(t *testing.T) *electionRun {
	t.Helper()
	le, err := leaderelection.NewLeaderElector(leaderelection.LeaderElectionConfig{
		Lock: &resourcelock.LeaseLock{
			LeaseMeta:  metav1.ObjectMeta{Namespace: "kube-system", Name: "test"},
			Client:     fakeclientset.NewSimpleClientset().CoordinationV1(),
			LockConfig: resourcelock.ResourceLockConfig{Identity: "test"},
		},
		LeaseDuration: 15 * time.Second,
		RenewDeadline: 10 * time.Second,
		RetryPeriod:   2 * time.Second,
		Callbacks:     leaderelection.LeaderCallbacks{OnStartedLeading: func(context.Context) {}, OnStoppedLeading: func() {}},
		Recovery:      &leaderelection.RecoveryConfig{},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &electionRun{elector: le}
}

// TestClientBuildersFor checks which builders controllers get: the configured
// ones unless recovery is enabled and the controllers run under an election,
// in which case writes are rejected while the election is not leading.
func TestClientBuildersFor(t *testing.T) {
	server := serviceAccountServer(t)
	ctx := context.Background()
	for _, useServiceAccounts := range []bool{false, true} {
		for _, tc := range []struct {
			name      string
			election  bool
			recovery  bool
			wantGated bool
		}{
			{name: "no election"},
			{name: "election without recovery", election: true},
			{name: "election with recovery", election: true, recovery: true, wantGated: true},
		} {
			t.Run(fmt.Sprintf("serviceAccountCredentials=%v/%s", useServiceAccounts, tc.name), func(t *testing.T) {
				kubeconfig := &restclient.Config{Host: server.URL}
				configured := clientbuilder.SimpleControllerClientBuilder{ClientConfig: &restclient.Config{Host: server.URL, UserAgent: "configured"}}
				c := &cloudcontrollerconfig.Config{Kubeconfig: kubeconfig, Client: clientset.NewForConfigOrDie(kubeconfig), ClientBuilder: configured}
				c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials = useServiceAccounts
				var election *electionRun
				switch {
				case tc.recovery:
					election = notLeadingElection(t)
				case tc.election:
					election = &electionRun{}
				}

				rootClientBuilder, controllerClientBuilder := clientBuildersFor(c.Complete(), election)
				if !tc.wantGated {
					if rootClientBuilder.ClientConfig != kubeconfig {
						t.Errorf("expected the root builder over the kubeconfig itself")
					}
					if got, ok := controllerClientBuilder.(clientbuilder.SimpleControllerClientBuilder); !ok || got != configured {
						t.Errorf("expected the configured controller builder, got %#v", controllerClientBuilder)
					}
				}
				for name, b := range map[string]clientbuilder.ControllerClientBuilder{"controller": controllerClientBuilder, "root": rootClientBuilder} {
					rest := b.ClientOrDie(name).CoreV1().RESTClient()
					if err := rest.Get().AbsPath("/api/v1/namespaces").Do(ctx).Error(); err != nil {
						t.Errorf("%s builder: read failed: %v", name, err)
					}
					err := rest.Post().AbsPath("/api/v1/namespaces").Body([]byte("{}")).Do(ctx).Error()
					switch {
					case tc.wantGated && !errors.Is(err, leaderelection.ErrWriteGateClosed):
						t.Errorf("%s builder: write must be rejected while not leading, got %v", name, err)
					case !tc.wantGated && err != nil:
						t.Errorf("%s builder: write failed: %v", name, err)
					}
				}
			})
		}
	}
}

// serviceAccountServer answers token requests with a token and every other
// request with a service account, which is all the dynamic client builder
// needs to hand out per-controller clients.
func serviceAccountServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/token") {
			_, _ = fmt.Fprintf(w, `{"kind":"TokenRequest","apiVersion":"authentication.k8s.io/v1","spec":{"expirationSeconds":3600},"status":{"token":"t","expirationTimestamp":%q}}`, time.Now().Add(time.Hour).Format(time.RFC3339))
			return
		}
		_, _ = w.Write([]byte(`{"kind":"ServiceAccount","apiVersion":"v1","metadata":{"name":"x","namespace":"kube-system"}}`))
	}))
	t.Cleanup(server.Close)
	return server
}

// TestClientBuilderFor checks that cloud controllers keep building clients from
// the configured builder unless LeaderElectionRecovery is enabled, in which
// case they use the controller context's, which carries the write gate.
func TestClientBuilderFor(t *testing.T) {
	configured := clientbuilder.SimpleControllerClientBuilder{ClientConfig: &restclient.Config{Host: "https://configured"}}
	fromContext := clientbuilder.SimpleControllerClientBuilder{ClientConfig: &restclient.Config{Host: "https://context"}}
	completed := (&cloudcontrollerconfig.Config{ClientBuilder: configured}).Complete()
	controllerContext := controllermanagerapp.ControllerContext{ClientBuilder: fromContext}

	for _, recovery := range []bool{false, true} {
		t.Run(fmt.Sprintf("recovery=%v", recovery), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, cmfeatures.LeaderElectionRecovery, recovery)
			want := cloudprovider.ControllerClientBuilder(configured)
			if recovery {
				want = fromContext
			}
			if got := clientBuilderFor(controllerContext, completed); got != want {
				t.Errorf("got builder over %v, want %v", got.(clientbuilder.SimpleControllerClientBuilder).ClientConfig.Host, want.(clientbuilder.SimpleControllerClientBuilder).ClientConfig.Host)
			}
		})
	}
}
