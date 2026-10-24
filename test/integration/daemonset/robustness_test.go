/*
Copyright The Kubernetes Authors.

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

package daemonset

import (
	"context"
	"fmt"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/robustness"
)

func TestDaemonSetReconcileRobustnessMatrix(t *testing.T) {
	dsName := "ds-1"

	suite := robustness.NewTestSuite(t, robustness.ControllerProfile{
		Name: "daemonset",
		Root: robustness.ResourceRef{
			Group:     "apps",
			Resource:  "daemonsets",
			Namespace: "default",
			Name:      dsName,
		},
		WritesRootStatus: true,
		UsesExpectations: true,
		Child: &robustness.ChildResource{
			Resource:            "pods",
			CreatedByController: true,
		},
	})

	// Check final status once the controller settles rather than on first match.
	suite.CheckWhenSettled(2 * time.Second)

	suite.SetControllerSetup(func(fixture *robustness.RobustnessTestFixture) {
		t.Logf("Feature StaleControllerConsistencyDaemonSet enabled: %v",
			utilfeature.DefaultFeatureGate.Enabled(features.StaleControllerConsistencyDaemonSet))

		informerFactory := informers.NewSharedInformerFactory(fixture.ClientSet(), 0)
		dsInformer := fixture.WrapDaemonSetInformer(informerFactory.Apps().V1().DaemonSets())
		podInformer := fixture.WrapPodInformer(informerFactory.Core().V1().Pods())
		nodeInformer := fixture.WrapNodeInformer(informerFactory.Core().V1().Nodes())

		dc, err := daemon.NewDaemonSetsController(
			fixture.Context(),
			dsInformer,
			informerFactory.Apps().V1().ControllerRevisions(),
			podInformer,
			nodeInformer,
			fixture.ClientSet(),
			flowcontrol.NewBackOff(100*time.Millisecond, 1*time.Second),
		)
		if err != nil {
			t.Fatalf("Failed to create DaemonSet controller: %v", err)
		}

		fixture.StartInformers(informerFactory)
		go dc.Run(fixture.Context(), 2)
	})

	suite.SetScenarioAction(func(ctx context.Context, fixture *robustness.RobustnessTestFixture) error {
		adminClient := fixture.AdminClientSet()

		node := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "target-node"},
			Status: v1.NodeStatus{
				Conditions: []v1.NodeCondition{
					{Type: v1.NodeReady, Status: v1.ConditionTrue},
				},
			},
		}
		_, err := adminClient.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}

		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: dsName, Namespace: "default"},
			Spec: appsv1.DaemonSetSpec{
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "nginx"}},
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "nginx"}},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Name: "nginx", Image: "nginx"},
						},
					},
				},
			},
		}
		_, err = adminClient.AppsV1().DaemonSets("default").Create(ctx, ds, metav1.CreateOptions{})
		return err
	})

	suite.AddSafetyInvariant("MaxOnePodPerNode", robustness.CountAtMost(1, "Pod",
		func(ctx context.Context, c clientset.Interface) (int, error) {
			pods, err := c.CoreV1().Pods("default").List(ctx, metav1.ListOptions{})
			if err != nil {
				return 0, err
			}
			return len(pods.Items), nil
		}))

	suite.AddLivenessInvariant("DaemonSetStatusUpdated", robustness.ObjectSatisfies(
		func(ctx context.Context, c clientset.Interface) (*appsv1.DaemonSet, error) {
			return c.AppsV1().DaemonSets("default").Get(ctx, dsName, metav1.GetOptions{})
		},
		func(ds *appsv1.DaemonSet) error {
			if ds.Status.CurrentNumberScheduled != 1 {
				return fmt.Errorf("expected Status.CurrentNumberScheduled to be 1, got %d", ds.Status.CurrentNumberScheduled)
			}
			return nil
		}))

	suite.Run()
}
