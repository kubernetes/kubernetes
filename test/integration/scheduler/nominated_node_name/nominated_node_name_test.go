/*
Copyright 2025 The Kubernetes Authors.

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

package nominatednodename

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
)

type FakePermitPlugin struct {
	code fwk.Code
}

type RunForeverPreBindPlugin struct {
	cancel <-chan struct{}
}

type NoNNNPostBindPlugin struct {
	t      *testing.T
	cancel <-chan struct{}
}

func (bp *NoNNNPostBindPlugin) Name() string {
	return "NoNNNPostBindPlugin"
}

func (bp *NoNNNPostBindPlugin) PostBind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) {
	if p.Status.NominatedNodeName != "" {
		bp.t.Fatalf("PostBind should not set .status.nominatedNodeName for pod %v/%v, but it was set to %v", p.Namespace, p.Name, p.Status.NominatedNodeName)
	}
}

// Name returns name of the plugin.
func (pp *FakePermitPlugin) Name() string {
	return "FakePermitPlugin"
}

// Permit implements the permit test plugin.
func (pp *FakePermitPlugin) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if pp.code == fwk.Wait {
		return fwk.NewStatus(pp.code, ""), 10 * time.Minute
	}
	return fwk.NewStatus(pp.code, ""), 0
}

// Name returns name of the plugin.
func (pp *RunForeverPreBindPlugin) Name() string {
	return "RunForeverPreBindPlugin"
}

// PreBindPreFlight is a test function that returns nil for testing.
func (pp *RunForeverPreBindPlugin) PreBindPreFlight(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	return nil
}

// PreBind is a test function that returns (true, nil) or errors for testing.
func (pp *RunForeverPreBindPlugin) PreBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	select {
	case <-ctx.Done():
		return fwk.NewStatus(fwk.Error, "context cancelled")
	case <-pp.cancel:
		return fwk.NewStatus(fwk.Error, "pre-bind cancelled")
	}
}

// Test_PutNominatedNodeNameInBindingCycle makes sure that nominatedNodeName is set in the binding cycle
// when the PreBind or Permit plugin (WaitOnPermit) is going to work.
func Test_PutNominatedNodeNameInBindingCycle(t *testing.T) {
	cancel := make(chan struct{})
	tests := []struct {
		name                    string
		plugin                  framework.Plugin
		expectNominatedNodeName bool
		cleanup                 func()
	}{
		{
			name:                    "NominatedNodeName is put if PreBindPlugin will run",
			plugin:                  &RunForeverPreBindPlugin{cancel: cancel},
			expectNominatedNodeName: true,
			cleanup: func() {
				close(cancel)
			},
		},
		{
			name:                    "NominatedNodeName is put if PermitPlugin will run at WaitOnPermit",
			expectNominatedNodeName: true,
			plugin: &FakePermitPlugin{
				code: fwk.Wait,
			},
		},
		{
			name: "NominatedNodeName is not put if PermitPlugin won't run at WaitOnPermit",
			plugin: &FakePermitPlugin{
				code: fwk.Success,
			},
		},
		{
			name:   "NominatedNodeName is not put if PermitPlugin nor PreBindPlugin will run",
			plugin: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NominatedNodeNameForExpectation, true)

			testContext := testutils.InitTestAPIServer(t, "nnn-test", nil)
			if test.cleanup != nil {
				defer test.cleanup()
			}

			pf := func(plugin framework.Plugin) frameworkruntime.PluginFactory {
				return func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
					return plugin, nil
				}
			}

			plugins := []framework.Plugin{&NoNNNPostBindPlugin{cancel: testContext.Ctx.Done(), t: t}}
			if test.plugin != nil {
				plugins = append(plugins, test.plugin)
			}

			registry, prof := schedulerutils.InitRegistryAndConfig(t, pf, plugins...)

			testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 10, true,
				scheduler.WithProfiles(prof),
				scheduler.WithFrameworkOutOfTreeRegistry(registry))
			defer teardown()

			pod, err := testutils.CreatePausePod(testCtx.ClientSet,
				testutils.InitPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testCtx.NS.Name}))
			if err != nil {
				t.Fatalf("Error while creating a test pod: %v", err)
			}

			if test.expectNominatedNodeName {
				if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
					t.Errorf(".status.nominatedNodeName was not set for pod %v/%v: %v", pod.Namespace, pod.Name, err)
				}
			} else {
				if err := testutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
					t.Errorf("Pod %v/%v was not scheduled: %v", pod.Namespace, pod.Name, err)
				}
			}
		})
	}
}
