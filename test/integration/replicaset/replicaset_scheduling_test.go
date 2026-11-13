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

package replicaset

import (
	"context"
	"testing"

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestReplicaSetDelegatedPodGroup verifies that a ReplicaSet selecting a
// template in a Workload via the workload-name and group-template-name
// annotations gets a PodGroup built from that template, controller-owned by
// the ReplicaSet.
func TestReplicaSetDelegatedPodGroup(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: true,
	})

	const (
		workloadName     = "test-workload"
		templateName     = "workers"
		templateMinCount = int32(2)
		replicas         = int32(3)
	)

	tCtx, rm, informers, c := setup(t)
	ns := framework.CreateNamespaceOrDie(c, "rs-delegated-podgroup", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	wl := &schedulingv1beta1.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: workloadName, Namespace: ns.Name},
		Spec: schedulingv1beta1.WorkloadSpec{
			PodGroupTemplates: []schedulingv1beta1.PodGroupTemplate{
				{
					Name: templateName,
					SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
						Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: templateMinCount},
					},
				},
			},
		},
	}
	workloadClient := c.SchedulingV1beta1().Workloads(ns.Name)
	if _, err := workloadClient.Create(tCtx, wl, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Workload: %v", err)
	}

	rs := newRS("delegated-rs", ns.Name, int(replicas))
	rs.Annotations = map[string]string{
		apischeduling.WorkloadNameAnnotation:      workloadName,
		apischeduling.GroupTemplateNameAnnotation: templateName,
	}
	rsClient := c.AppsV1().ReplicaSets(ns.Name)
	created, err := rsClient.Create(tCtx, rs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create ReplicaSet: %v", err)
	}

	stop := runControllerAndInformers(t, rm, informers, 0)
	defer stop()

	pgClient := c.SchedulingV1beta1().PodGroups(ns.Name)
	var podGroup *schedulingv1beta1.PodGroup
	err = wait.PollUntilContextTimeout(tCtx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		pgs, listErr := pgClient.List(ctx, metav1.ListOptions{})
		if listErr != nil {
			return false, listErr
		}
		for i := range pgs.Items {
			for _, ref := range pgs.Items[i].OwnerReferences {
				if ref.UID == created.UID && ref.Kind == "ReplicaSet" && ref.Controller != nil && *ref.Controller {
					podGroup = &pgs.Items[i]
					return true, nil
				}
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed to observe PodGroup owned by ReplicaSet: %v", err)
	}

	if podGroup.Spec.SchedulingPolicy.Gang == nil || podGroup.Spec.SchedulingPolicy.Gang.MinCount != templateMinCount {
		t.Errorf("PodGroup gang minCount = %v, want %d", podGroup.Spec.SchedulingPolicy.Gang, templateMinCount)
	}

	if ref := podGroup.Spec.WorkloadRef; ref == nil || ref.WorkloadName != workloadName || ref.TemplateName != templateName {
		t.Errorf("PodGroup workloadRef = %+v, want workload %q template %q", ref, workloadName, templateName)
	}
}
