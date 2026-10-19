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
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/server/healthz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/leaderelection"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/client-go/transport"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/controller-manager/pkg/clientbuilder"
	cmfeatures "k8s.io/controller-manager/pkg/features"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

func TestControllerNamesConsistency(t *testing.T) {
	controllerNameRegexp := regexp.MustCompile("^[a-z]([-a-z]*[a-z])?$")

	for _, name := range KnownControllers() {
		if !controllerNameRegexp.MatchString(name) {
			t.Errorf("name consistency check failed: controller %q must consist of lower case alphabetic characters or '-', and must start and end with an alphabetic character", name)
		}
		if !strings.HasSuffix(name, "-controller") {
			t.Errorf("name consistency check failed: controller %q must have \"-controller\" suffix", name)
		}
	}
}

func TestControllerNamesDeclaration(t *testing.T) {
	declaredControllers := sets.NewString(
		names.ServiceAccountTokenController,
		names.EndpointsController,
		names.EndpointSliceController,
		names.EndpointSliceMirroringController,
		names.ReplicationControllerController,
		names.PodGarbageCollectorController,
		names.ResourceQuotaController,
		names.NamespaceController,
		names.ServiceAccountController,
		names.GarbageCollectorController,
		names.DaemonSetController,
		names.JobController,
		names.PodGroupProtectionController,
		names.DeploymentController,
		names.ReplicaSetController,
		names.HorizontalPodAutoscalerController,
		names.DisruptionController,
		names.StatefulSetController,
		names.CronJobController,
		names.CertificateSigningRequestSigningController,
		names.CertificateSigningRequestApprovingController,
		names.CertificateSigningRequestCleanerController,
		names.PodCertificateRequestCleanerController,
		names.TTLController,
		names.BootstrapSignerController,
		names.TokenCleanerController,
		names.NodeIpamController,
		names.NodeLifecycleController,
		names.TaintEvictionController,
		names.PersistentVolumeBinderController,
		names.PersistentVolumeAttachDetachController,
		names.PersistentVolumeExpanderController,
		names.ClusterRoleAggregationController,
		names.PersistentVolumeClaimProtectionController,
		names.PersistentVolumeProtectionController,
		names.VolumeAttributesClassProtectionController,
		names.TTLAfterFinishedController,
		names.RootCACertificatePublisherController,
		names.KubeAPIServerClusterTrustBundlePublisherController,
		names.EphemeralVolumeController,
		names.StorageVersionGarbageCollectorController,
		names.ResourceClaimController,
		names.DeviceTaintEvictionController,
		names.ResourcePoolStatusRequestController,
		names.LegacyServiceAccountTokenCleanerController,
		names.ValidatingAdmissionPolicyStatusController,
		names.ServiceCIDRController,
		names.StorageVersionMigratorController,
		names.SELinuxWarningController,
		names.EvictionRequestController,
		names.ImperativeEvictionResponderController,
	)

	for _, name := range KnownControllers() {
		if !declaredControllers.Has(name) {
			t.Errorf("name declaration check failed: controller name %q should be declared in  \"controller_names.go\" and added to this test", name)
		}
	}
}

func TestNewControllerDescriptorsShouldNotPanic(t *testing.T) {
	NewControllerDescriptors()
}

func TestNewControllerDescriptorsAlwaysReturnsDescriptorsForAllControllers(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		"AllAlpha": false,
		"AllBeta":  false,
	})

	controllersWithoutFeatureGates := KnownControllers()

	// AllBeta must be enabled before AllAlpha to resolve dependencies.
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		"AllBeta":  true,
		"AllAlpha": true,
	})

	controllersWithFeatureGates := KnownControllers()

	if diff := cmp.Diff(controllersWithoutFeatureGates, controllersWithFeatureGates); diff != "" {
		t.Errorf("unexpected controllers after enabling feature gates, NewControllerDescriptors should always return all controller descriptors. Controllers should define required feature gates in ControllerDescriptor.requiredFeatureGates. Diff of returned controllers:\n%s", diff)
	}
}

func TestFeatureGatedControllersShouldNotDefineAliases(t *testing.T) {
	featureGateRegex := regexp.MustCompile("^([a-zA-Z0-9]+)")

	alphaFeatures := sets.NewString()
	for _, featureText := range utilfeature.DefaultFeatureGate.KnownFeatures() {
		// we have to parse this from KnownFeatures, because usage of mutable FeatureGate is not allowed in unit tests
		feature := featureGateRegex.FindString(featureText)
		if strings.Contains(featureText, string(featuregate.Alpha)) && feature != "AllAlpha" {
			alphaFeatures.Insert(feature)
		}

	}

	for name, controller := range NewControllerDescriptors() {
		if len(controller.GetAliases()) == 0 {
			continue
		}

		requiredFeatureGates := controller.GetRequiredFeatureGates()
		if len(requiredFeatureGates) == 0 {
			continue
		}

		areAllRequiredFeaturesAlpha := true
		for _, feature := range requiredFeatureGates {
			if !alphaFeatures.Has(string(feature)) {
				areAllRequiredFeaturesAlpha = false
				break
			}
		}

		if areAllRequiredFeaturesAlpha {
			t.Errorf("alias check failed: controller name %q should not be aliased as it is still guarded by alpha feature gates (%v) and thus should have only a canonical name", name, requiredFeatureGates)
		}
	}
}

func TestRunControllers(t *testing.T) {
	testCases := []struct {
		name                     string
		newController            func(ctx context.Context) Controller
		shutdownTimeout          time.Duration
		expectedCleanTermination bool
	}{
		{
			name: "clean shutdown",
			newController: func(testCtx context.Context) Controller {
				return newControllerLoop(func(ctx context.Context) {
					<-ctx.Done()
				}, "controller-A")
			},
			shutdownTimeout:          10 * time.Second,
			expectedCleanTermination: true,
		},
		{
			name: "shutdown timeout",
			newController: func(testCtx context.Context) Controller {
				return newControllerLoop(func(ctx context.Context) {
					<-testCtx.Done()
				}, "controller-A")
			},
			shutdownTimeout:          50 * time.Millisecond,
			expectedCleanTermination: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			controllerCtx := ControllerContext{}
			controllerCtx.ComponentConfig.Generic.Controllers = []string{"controller-A"}

			// testCtx is used to make sure the controller failing to shut down can exit after the test is finished.
			testCtx, cancelTest := context.WithCancel(ctx)
			defer cancelTest()

			initFuncCalled := false
			descriptor := &ControllerDescriptor{
				name: "controller-A",
				constructor: func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
					initFuncCalled = true
					return tc.newController(testCtx), nil
				},
			}

			var healthChecks mockHealthCheckAdder
			controllers, err := BuildControllers(ctx, controllerCtx, map[string]*ControllerDescriptor{
				"controller-A": descriptor,
			}, nil, &healthChecks)
			if err != nil {
				t.Fatalf("unexpected error building controllers: %v", err)
			}
			if !initFuncCalled {
				t.Errorf("controller constructor was not called")
			}
			if len(healthChecks.Checks) == 0 {
				t.Errorf("expected health check to be added")
			}

			// ctx is used to wait in the controller for shutdown and also to start the shutdown timeout in RunControllers.
			// To start the shutdown timeout immediately, we start with a cancelled context already.
			ctx, cancelController := context.WithCancel(ctx)
			cancelController()

			cleanShutdown := RunControllers(ctx, controllerCtx, controllers, 0, tc.shutdownTimeout)
			if cleanShutdown != tc.expectedCleanTermination {
				t.Errorf("expected clean shutdown %v, got %v", tc.expectedCleanTermination, cleanShutdown)
			}
		})
	}
}

type mockHealthCheckAdder struct {
	Checks []healthz.HealthChecker
}

func (m *mockHealthCheckAdder) AddHealthChecker(checks ...healthz.HealthChecker) {
	m.Checks = append(m.Checks, checks...)
}

func runControllers(
	ctx context.Context, controllerCtx ControllerContext,
	controllerDescriptors map[string]*ControllerDescriptor, healthzChecks HealthCheckAdder,
) error {
	controllers, err := BuildControllers(ctx, controllerCtx, controllerDescriptors, nil, healthzChecks)
	if err != nil {
		return err
	}
	RunControllers(ctx, controllerCtx, controllers, 0, 0)
	return nil
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

func TestCreateClientBuilders(t *testing.T) {
	server := serviceAccountServer(t)
	for _, useServiceAccounts := range []bool{false, true} {
		for _, gated := range []bool{false, true} {
			t.Run(fmt.Sprintf("serviceAccountCredentials=%v/gate=%v", useServiceAccounts, gated), func(t *testing.T) {
				kubeconfig := &restclient.Config{Host: server.URL}
				c := &config.Config{Kubeconfig: kubeconfig, Client: clientset.NewForConfigOrDie(kubeconfig)}
				c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials = useServiceAccounts

				wraps := 0
				var gate transport.WrapperFunc
				if gated {
					gate = func(rt http.RoundTripper) http.RoundTripper {
						wraps++
						return rt
					}
				}

				clientBuilder, rootClientBuilder := createClientBuilders(c.Complete(), gate)

				// Building a client applies the gate to its transport.
				rootClientBuilder.ClientOrDie("root")
				clientBuilder.ClientOrDie("controller")
				wantWraps := 0
				if gated {
					wantWraps = 2
				}
				if wraps != wantWraps {
					t.Errorf("gate applied %d times building a root and a controller client, want %d", wraps, wantWraps)
				}
				if kubeconfig.WrapTransport != nil {
					t.Errorf("the controller manager's kubeconfig must not be modified")
				}

				if useServiceAccounts {
					if _, ok := clientBuilder.(*clientbuilder.DynamicControllerClientBuilder); !ok {
						t.Errorf("controller builder is %T, want the service account builder", clientBuilder)
					}
				} else if clientBuilder != rootClientBuilder {
					t.Errorf("controller builder is %T, want the root builder", clientBuilder)
				}
			})
		}
	}
}

func TestRecoveryConfigFor(t *testing.T) {
	deadline := func(d time.Duration) *config.CompletedConfig {
		c := &config.Config{}
		c.ComponentConfig.Generic.LeaderElectionRecoveryDeadline = metav1.Duration{Duration: d}
		return c.Complete()
	}
	if got := recoveryConfigFor(deadline(0)); got == nil || got.RecoveryDeadline != nil {
		t.Errorf("zero deadline: got %+v, want a config with no deadline", got)
	}
	if got := recoveryConfigFor(deadline(30 * time.Second)); got == nil || got.RecoveryDeadline == nil || *got.RecoveryDeadline != 30*time.Second {
		t.Errorf("30s deadline: got %+v, want a config with a 30s deadline", got)
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

// TestClientBuildersFor checks which builders controllers get: the controller
// manager's own unless recovery is enabled and the controllers run under an
// election, in which case writes are rejected while the election is not leading.
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
				c := &config.Config{Kubeconfig: kubeconfig, Client: clientset.NewForConfigOrDie(kubeconfig)}
				c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials = useServiceAccounts
				completed := c.Complete()
				clientBuilder, rootClientBuilder := createClientBuilders(completed, nil)
				var election *electionRun
				switch {
				case tc.recovery:
					election = notLeadingElection(t)
				case tc.election:
					election = &electionRun{}
				}

				gotClient, gotRoot := clientBuildersFor(completed, clientBuilder, rootClientBuilder, election)
				if !tc.wantGated && (gotClient != clientBuilder || gotRoot != rootClientBuilder) {
					t.Fatalf("expected the controller manager's own builders back, got %T and %T", gotClient, gotRoot)
				}
				for name, b := range map[string]clientbuilder.TypedClientBuilder{"controller": gotClient, "root": gotRoot} {
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

func TestLeaderElectionRecoveryEnabled(t *testing.T) {
	for _, tc := range []struct {
		name          string
		recovery, cle bool
		want          bool
	}{
		{name: "gate off", want: false},
		{name: "gate on", recovery: true, want: true},
		{name: "gate on with coordinated leader election", recovery: true, cle: true, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, cmfeatures.LeaderElectionRecovery, tc.recovery)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.CoordinatedLeaderElection, tc.cle)
			logger, _ := ktesting.NewTestContext(t)
			if got := leaderElectionRecoveryEnabled(logger); got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

type testPausable struct{ paused bool }

func (p *testPausable) Pause()  { p.paused = true }
func (p *testPausable) Resume() { p.paused = false }

// TestBuildControllersRecoveryOptIn verifies that BuildControllers registers
// the controllers that implement Pausable with the election, and only those,
// so they are paused while the controller manager is not leading.
func TestBuildControllersRecoveryOptIn(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	pausable := &testPausable{}
	specialPausable := &testPausable{}
	pausableConstructor := func(p *testPausable) func(context.Context, ControllerContext, string) (Controller, error) {
		return func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
			return newPausableControllerLoop(func(context.Context) {}, controllerName, p), nil
		}
	}
	plainConstructor := func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
		return newControllerLoop(func(context.Context) {}, controllerName), nil
	}
	descriptors := map[string]*ControllerDescriptor{
		"pausable": {name: "pausable", constructor: pausableConstructor(pausable)},
		"plain":    {name: "plain", constructor: plainConstructor},
		// Skipped by its constructor: nothing runs, so nothing to pause.
		"skipped": {name: "skipped", constructor: func(context.Context, ControllerContext, string) (Controller, error) { return nil, nil }},
		// Disabled by --controllers: never built.
		"disabled": {name: "disabled", constructor: plainConstructor},
		// Built through the special-handling path, like the tokens controller.
		names.ServiceAccountTokenController: {name: names.ServiceAccountTokenController, constructor: pausableConstructor(specialPausable), requiresSpecialHandling: true},
	}

	t.Run("recovery enabled", func(t *testing.T) {
		controllerCtx := ControllerContext{}
		controllerCtx.ComponentConfig.Generic.Controllers = []string{"*", "-disabled"}
		// The election is not leading, so registered controllers are paused at once.
		controllerCtx.PausableRegistrar = notLeadingElection(t).elector

		var healthChecks mockHealthCheckAdder
		controllers, err := BuildControllers(ctx, controllerCtx, descriptors, nil, &healthChecks)
		if err != nil {
			t.Fatalf("unexpected error building controllers: %v", err)
		}

		if !pausable.paused || !specialPausable.paused {
			t.Error("the pausable and the special-handling controllers must be registered with the election and paused while it is not leading")
		}
		// The plain controller is built and runs unpaused.
		if len(controllers) != 3 {
			t.Errorf("expected the pausable, plain and special-handling controllers to be built, got %d controllers", len(controllers))
		}
	})

	t.Run("recovery disabled", func(t *testing.T) {
		// Without a registrar nothing is recorded and building still works.
		controllerCtx := ControllerContext{}
		controllerCtx.ComponentConfig.Generic.Controllers = []string{"*"}
		var healthChecks mockHealthCheckAdder
		if _, err := BuildControllers(ctx, controllerCtx, descriptors, nil, &healthChecks); err != nil {
			t.Fatalf("unexpected error building controllers: %v", err)
		}
	})
}
