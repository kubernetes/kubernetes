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
	"slices"
	"testing"

	apps "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

const replicaSetPodGroupFinalizer = "apps.kubernetes.io/replicaset-podgroup-cleanup"

// TestReplicaSetPodGroupCleanupFinalizerGateDisabled verifies selective cleanup
// finalizer removal when WorkloadWithDeployment is disabled.
func TestReplicaSetPodGroupCleanupFinalizerGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:        true,
		features.WorkloadWithDeployment: false,
	})

	tCtx, rm, informers, c := setup(t)
	ns := framework.CreateNamespaceOrDie(c, "rs-podgroup-cleanup", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	const unrelatedFinalizer = "test.example.com/hold"
	rs := newRS("gang-rs", ns.Name, 0)
	rs.Finalizers = []string{replicaSetPodGroupFinalizer, unrelatedFinalizer}
	created, err := c.AppsV1().ReplicaSets(ns.Name).Create(tCtx, rs, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create ReplicaSet: %v", err)
	}

	stop := runControllerAndInformers(t, rm, informers, 0)
	defer stop()

	if err := c.AppsV1().ReplicaSets(ns.Name).Delete(tCtx, created.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("failed to delete ReplicaSet: %v", err)
	}

	err = wait.PollUntilContextTimeout(tCtx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		got, err := c.AppsV1().ReplicaSets(ns.Name).Get(ctx, created.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return got.DeletionTimestamp != nil && slices.Equal(got.Finalizers, []string{unrelatedFinalizer}), nil
	})
	if err != nil {
		t.Fatalf("failed to observe cleanup finalizer removal on ReplicaSet %q with %q preserved: %v", created.Name, unrelatedFinalizer, err)
	}

	updateRS(t, c.AppsV1().ReplicaSets(ns.Name), created.Name, func(update *apps.ReplicaSet) {
		update.Finalizers = slices.DeleteFunc(update.Finalizers, func(finalizer string) bool {
			return finalizer == unrelatedFinalizer
		})
	})

	err = wait.PollUntilContextTimeout(tCtx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		_, err := c.AppsV1().ReplicaSets(ns.Name).Get(ctx, created.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		t.Fatalf("ReplicaSet %q stranded after cleanup while gate disabled: %v", created.Name, err)
	}
}
