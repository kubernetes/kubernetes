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

package scheduler

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// The returned shutdown func will delete created resources and scheduler, resources should be those
// that will affect the scheduling result, like nodes, pods, etc.. Namespaces should not be
// deleted here because it's created together with the apiserver, they should be deleted
// simultaneously or we'll have no namespace.
// This should only be called when you want to kill the scheduler alone, away from apiserver.
// For example, in scheduler integration tests, recreating apiserver is performance consuming,
// then shutdown the scheduler and recreate it between each test case is a better approach.
func InitTestSchedulerForFrameworkTest(t *testing.T, testCtx *testutils.TestContext, nodeCount int, runScheduler bool, opts ...scheduler.Option) (*testutils.TestContext, testutils.ShutdownFunc) {
	testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0, opts...)
	testutils.SyncSchedulerInformerFactory(testCtx)
	if runScheduler {
		go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
	}

	if nodeCount > 0 {
		if _, err := testutils.CreateAndWaitForNodesInCache(testCtx, "test-node", st.MakeNode(), nodeCount); err != nil {
			// Make sure to cleanup the resources when initializing error.
			testutils.CleanupTest(t, testCtx)
			t.Fatal(err)
		}
	}

	teardown := func() {
		err := testCtx.ClientSet.CoreV1().Nodes().DeleteCollection(testCtx.SchedulerCtx, *metav1.NewDeleteOptions(0), metav1.ListOptions{})
		if err != nil {
			t.Errorf("error while deleting all nodes: %v", err)
		}
		err = testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).DeleteCollection(testCtx.SchedulerCtx, *metav1.NewDeleteOptions(0), metav1.ListOptions{})
		if err != nil {
			t.Errorf("error while deleting pod: %v", err)
		}
		// Wait for all pods to be deleted, or will failed to create same name pods
		// required in other test cases.
		err = wait.PollUntilContextTimeout(testCtx.SchedulerCtx, time.Millisecond, wait.ForeverTestTimeout, true,
			testutils.PodsCleanedUp(testCtx.SchedulerCtx, testCtx.ClientSet, testCtx.NS.Name))
		if err != nil {
			t.Errorf("error while waiting for all pods to be deleted: %v", err)
		}
		// Kill the scheduler.
		testCtx.SchedulerCloseFn()
	}

	return testCtx, teardown
}

// NewPlugin returns a plugin factory with specified Plugin.
func NewPlugin(plugin framework.Plugin) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		return plugin, nil
	}
}

// InitRegistryAndConfig returns registry and plugins config based on give plugins.
func InitRegistryAndConfig(t *testing.T, factory func(plugin framework.Plugin) frameworkruntime.PluginFactory, plugins ...framework.Plugin) (frameworkruntime.Registry, schedulerconfig.KubeSchedulerProfile) {
	if len(plugins) == 0 {
		return frameworkruntime.Registry{}, schedulerconfig.KubeSchedulerProfile{}
	}

	if factory == nil {
		factory = NewPlugin
	}

	registry := frameworkruntime.Registry{}
	pls := &configv1.Plugins{}

	for _, p := range plugins {
		registry.Register(p.Name(), factory(p))
		plugin := configv1.Plugin{Name: p.Name()}

		switch p.(type) {
		case framework.QueueSortPlugin:
			pls.QueueSort.Enabled = append(pls.QueueSort.Enabled, plugin)
			// It's intentional to disable the PrioritySort plugin.
			pls.QueueSort.Disabled = []configv1.Plugin{{Name: queuesort.Name}}
		case framework.PreEnqueuePlugin:
			pls.PreEnqueue.Enabled = append(pls.PreEnqueue.Enabled, plugin)
		case framework.PreFilterPlugin:
			pls.PreFilter.Enabled = append(pls.PreFilter.Enabled, plugin)
		case framework.FilterPlugin:
			pls.Filter.Enabled = append(pls.Filter.Enabled, plugin)
		case framework.PreScorePlugin:
			pls.PreScore.Enabled = append(pls.PreScore.Enabled, plugin)
		case framework.ScorePlugin:
			pls.Score.Enabled = append(pls.Score.Enabled, plugin)
		case framework.ReservePlugin:
			pls.Reserve.Enabled = append(pls.Reserve.Enabled, plugin)
		case framework.PreBindPlugin:
			pls.PreBind.Enabled = append(pls.PreBind.Enabled, plugin)
		case framework.BindPlugin:
			pls.Bind.Enabled = append(pls.Bind.Enabled, plugin)
			// It's intentional to disable the DefaultBind plugin. Otherwise, DefaultBinder's failure would fail
			// a pod's scheduling, as well as the test BindPlugin's execution.
			pls.Bind.Disabled = []configv1.Plugin{{Name: defaultbinder.Name}}
		case framework.PostBindPlugin:
			pls.PostBind.Enabled = append(pls.PostBind.Enabled, plugin)
		case framework.PermitPlugin:
			pls.Permit.Enabled = append(pls.Permit.Enabled, plugin)
		}
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
