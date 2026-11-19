/*
Copyright 2026 The Kubernetes Authors.

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

package plugins

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/tools/events"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// requiredEPTestPlugin declares PreBind, PostBind and Reserve as extension
// points that must be enabled together, mimicking a plugin that acquires a
// node-scoped lock in PreBind and releases it only in PostBind (bind succeeded)
// or Unreserve (bind failed).
type requiredEPTestPlugin struct {
	preBindCalled atomic.Int64
}

var _ fwk.RequiredExtensionPointsPlugin = &requiredEPTestPlugin{}

func (p *requiredEPTestPlugin) Name() string { return "required-ep-test-plugin" }

func (p *requiredEPTestPlugin) RequiredExtensionPoints() []string {
	return []string{fwk.ExtensionPointPreBind, fwk.ExtensionPointPostBind, fwk.ExtensionPointReserve}
}

func (p *requiredEPTestPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	return &fwk.PreBindPreFlightResult{AllowParallel: true}, fwk.NewStatus(fwk.Success, "")
}

func (p *requiredEPTestPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	p.preBindCalled.Add(1)
	return nil
}

func (p *requiredEPTestPlugin) PostBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
}

func (p *requiredEPTestPlugin) Reserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	return nil
}

func (p *requiredEPTestPlugin) Unreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
}

// requiredEPRegistryAndProfile registers requiredEPTestPlugin and enables it at
// exactly the requested extension points (the shared InitRegistryAndConfig
// helper only enables a plugin at the first extension point it implements, so
// the profile is built explicitly here).
func requiredEPRegistryAndProfile(t *testing.T, plugin *requiredEPTestPlugin, preBind, postBind, reserve bool) (frameworkruntime.Registry, schedulerconfig.KubeSchedulerProfile) {
	t.Helper()
	registry := frameworkruntime.Registry{}
	registry.Register(plugin.Name(), schedulerutils.NewPlugin(plugin))

	pls := &configv1.Plugins{}
	entry := configv1.Plugin{Name: plugin.Name()}
	if preBind {
		pls.PreBind.Enabled = append(pls.PreBind.Enabled, entry)
	}
	if postBind {
		pls.PostBind.Enabled = append(pls.PostBind.Enabled, entry)
	}
	if reserve {
		pls.Reserve.Enabled = append(pls.Reserve.Enabled, entry)
	}
	versionedCfg := configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins:       pls,
		}},
	}
	cfg := configtesting.V1ToInternalWithDefaults(t, versionedCfg)
	return registry, cfg.Profiles[0]
}

// newSchedulerExpectingError builds a scheduler like InitTestSchedulerWithOptions
// but returns the construction error instead of failing the test, so a rejected
// plugin configuration can be asserted.
func newSchedulerExpectingError(t *testing.T, serverCtx ktesting.TContext, registry frameworkruntime.Registry, prof schedulerconfig.KubeSchedulerProfile) error {
	t.Helper()
	ctx, cancel := context.WithCancel(serverCtx)
	t.Cleanup(cancel)
	informerFactory := scheduler.NewInformerFactory(serverCtx.Client(), 0, nil)
	var dynInformerFactory dynamicinformer.DynamicSharedInformerFactory
	if serverCtx.RESTConfig() != nil {
		dynClient := dynamic.NewForConfigOrDie(serverCtx.RESTConfig())
		dynInformerFactory = dynamicinformer.NewFilteredDynamicSharedInformerFactory(dynClient, 0, v1.NamespaceAll, nil)
	}
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: serverCtx.Client().EventsV1()})
	// NewBroadcaster starts an internal watch loop goroutine; shut it down even
	// when scheduler.New fails, otherwise the goroutine leaks.
	t.Cleanup(eventBroadcaster.Shutdown)
	_, err := scheduler.New(
		ctx,
		serverCtx.Client(),
		informerFactory,
		dynInformerFactory,
		profile.NewRecorderFactory(eventBroadcaster),
		scheduler.WithKubeConfig(serverCtx.RESTConfig()),
		scheduler.WithProfiles(prof),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	return err
}

// TestRequiredExtensionPointsSchedulerStartup verifies end to end that the
// scheduler refuses to start when a plugin's required-together extension points
// are only partially enabled, and starts (and schedules a pod) when they are all
// enabled.
func TestRequiredExtensionPointsSchedulerStartup(t *testing.T) {
	testContext := testutils.InitTestAPIServer(t, "required-ep", nil)
	tests := []struct {
		name         string
		preBind      bool
		postBind     bool
		reserve      bool
		wantStartErr string
	}{
		{
			name:     "all required extension points enabled",
			preBind:  true,
			postBind: true,
			reserve:  true,
		},
		{
			name:         "PreBind only, PostBind and Reserve disabled",
			preBind:      true,
			wantStartErr: "enabled together",
		},
		{
			name:         "PreBind and PostBind, Reserve disabled",
			preBind:      true,
			postBind:     true,
			wantStartErr: "enabled together",
		},
		{
			name:         "PreBind and Reserve, PostBind disabled",
			preBind:      true,
			reserve:      true,
			wantStartErr: "enabled together",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plugin := &requiredEPTestPlugin{}
			registry, prof := requiredEPRegistryAndProfile(t, plugin, tc.preBind, tc.postBind, tc.reserve)

			if tc.wantStartErr != "" {
				err := newSchedulerExpectingError(t, testContext, registry, prof)
				if err == nil || !strings.Contains(err.Error(), tc.wantStartErr) {
					t.Fatalf("expected scheduler startup error containing %q, got %v", tc.wantStartErr, err)
				}
				return
			}

			testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 1, true,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer teardown()

			pod, err := testutils.CreatePausePod(testCtx.Client(),
				testutils.InitPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.Namespace()}))
			if err != nil {
				t.Fatalf("error creating test pod: %v", err)
			}
			if err := schedulerutils.WaitForPodToSchedule(testCtx, testCtx.Client(), pod); err != nil {
				t.Fatalf("expected pod to schedule with all required extension points enabled: %v", err)
			}
			if plugin.preBindCalled.Load() == 0 {
				t.Fatalf("expected the plugin's PreBind to run for the scheduled pod")
			}
		})
	}
}
