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

package eventhandler

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/scheduling/corev1"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

var _ framework.FilterPlugin = &fooPlugin{}

type fooPlugin struct {
}

func (pl *fooPlugin) Name() string {
	return "foo"
}

func (pl *fooPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	taints := nodeInfo.Node().Spec.Taints
	if len(taints) == 0 {
		return nil
	}

	if corev1.TolerationsTolerateTaint(pod.Spec.Tolerations, &nodeInfo.Node().Spec.Taints[0]) {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable)
}

func (pl *fooPlugin) EventsToRegister() []framework.ClusterEventWithHint {
	return []framework.ClusterEventWithHint{
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.UpdateNodeTaint}},
	}
}

// newPlugin returns a plugin factory with specified Plugin.
func newPlugin(plugin framework.Plugin) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, fh framework.Handle) (framework.Plugin, error) {
		return plugin, nil
	}
}

func TestUpdateNodeEvent(t *testing.T) {
	testContext := testutils.InitTestAPIServer(t, "test-event", nil)

	taints := []v1.Taint{{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectNoSchedule}}
	nodeWrapper := st.MakeNode().Name("node-0").Label("kubernetes.io/hostname", "node-0").Taints(taints).Obj()
	podWrapper := testutils.InitPausePod(&testutils.PausePodConfig{Name: "test-pod", Namespace: testContext.NS.Name})
	fooPlugin := &fooPlugin{}

	registry := frameworkruntime.Registry{
		fooPlugin.Name(): newPlugin(fooPlugin),
	}

	// Setup plugins for testing.
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To[string](v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				Filter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: fooPlugin.Name()},
					},
					Disabled: []configv1.Plugin{
						{Name: "*"},
					},
				},
			},
		}},
	})

	testCtx, teardown := schedulerutils.InitTestSchedulerForFrameworkTest(t, testContext, 0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	defer teardown()

	node, err := testutils.CreateNode(testCtx.ClientSet, nodeWrapper)
	if err != nil {
		t.Fatalf("Creating node error: %v", err)
	}

	pod, err := testutils.CreatePausePod(testCtx.ClientSet, podWrapper)
	if err != nil {
		t.Fatalf("Creating pod error: %v", err)
	}

	if err := testutils.WaitForPodUnschedulable(testCtx.Ctx, testCtx.ClientSet, pod); err != nil {
		t.Fatalf("Pod %v got scheduled: %v", pod.Name, err)
	}
	node, err = testCtx.ClientSet.CoreV1().Nodes().Get(testCtx.Ctx, node.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error while getting a node: %v", err)
	}

	// Update node label and node taints
	node.Labels["foo"] = "bar"
	node.Spec.Taints = nil

	_, err = testCtx.ClientSet.CoreV1().Nodes().Update(testCtx.Ctx, node, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating the node: %v", err)
	}

	if err := testutils.WaitForPodToSchedule(testCtx.ClientSet, pod); err != nil {
		t.Errorf("Pod %v was not scheduled: %v", pod.Name, err)
	}
}
