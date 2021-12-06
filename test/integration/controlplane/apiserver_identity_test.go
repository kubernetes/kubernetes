/*
Copyright 2020 The Kubernetes Authors.

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

package controlplane

import (
	"context"
	"strings"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

const (
	testLeaseName = "apiserver-lease-test"
)

func TestCreateLeaseOnStart(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)()
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf(`Waiting the kube-apiserver Lease to be created`)
	if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
		leases, err := kubeclient.
			CoordinationV1().
			Leases(metav1.NamespaceSystem).
			List(context.TODO(), metav1.ListOptions{LabelSelector: controlplane.KubeAPIServerIdentityLeaseLabelSelector})
		if err != nil {
			return false, err
		}
		if leases != nil && len(leases.Items) == 1 && strings.HasPrefix(leases.Items[0].Name, "kube-apiserver-") {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("Failed to see the kube-apiserver lease: %v", err)
	}
}

func TestLeaseGarbageCollection(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)()
	result := kubeapiservertesting.StartTestServerOrDie(t, nil,
		// This shorten the GC check period to make the test run faster.
		// Since we are testing GC behavior on leases we create, what happens to
		// the real apiserver lease doesn't matter.
		[]string{"--identity-lease-duration-seconds=1"},
		framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	expiredLease := newTestLease(time.Now().Add(-2*time.Hour), metav1.NamespaceSystem)
	t.Run("expired apiserver lease should be garbage collected",
		testLeaseGarbageCollected(t, kubeclient, expiredLease))

	freshLease := newTestLease(time.Now().Add(-2*time.Minute), metav1.NamespaceSystem)
	t.Run("fresh apiserver lease should not be garbage collected",
		testLeaseNotGarbageCollected(t, kubeclient, freshLease))

	expiredLease.Labels = nil
	t.Run("expired non-identity lease should not be garbage collected",
		testLeaseNotGarbageCollected(t, kubeclient, expiredLease))

	// identity leases (with k8s.io/component label) created in user namespaces should not be GC'ed
	expiredNonKubeSystemLease := newTestLease(time.Now().Add(-2*time.Hour), metav1.NamespaceDefault)
	t.Run("expired non-system identity lease should not be garbage collected",
		testLeaseNotGarbageCollected(t, kubeclient, expiredNonKubeSystemLease))
}

func testLeaseGarbageCollected(t *testing.T, client kubernetes.Interface, lease *coordinationv1.Lease) func(t *testing.T) {
	return func(t *testing.T) {
		ns := lease.Namespace
		if _, err := client.CoordinationV1().Leases(ns).Create(context.TODO(), lease, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Unexpected error creating lease: %v", err)
		}
		if err := wait.PollImmediate(500*time.Millisecond, 5*time.Second, func() (bool, error) {
			_, err := client.CoordinationV1().Leases(ns).Get(context.TODO(), lease.Name, metav1.GetOptions{})
			if err == nil {
				return false, nil
			}
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}); err != nil {
			t.Fatalf("Failed to see the expired lease garbage collected: %v", err)
		}
	}
}

func testLeaseNotGarbageCollected(t *testing.T, client kubernetes.Interface, lease *coordinationv1.Lease) func(t *testing.T) {
	return func(t *testing.T) {
		ns := lease.Namespace
		if _, err := client.CoordinationV1().Leases(ns).Create(context.TODO(), lease, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Unexpected error creating lease: %v", err)
		}
		if err := wait.PollImmediate(500*time.Millisecond, 5*time.Second, func() (bool, error) {
			_, err := client.CoordinationV1().Leases(ns).Get(context.TODO(), lease.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}); err == nil {
			t.Fatalf("Unexpected valid lease getting garbage collected")
		}
		if _, err := client.CoordinationV1().Leases(ns).Get(context.TODO(), lease.Name, metav1.GetOptions{}); err != nil {
			t.Fatalf("Failed to retrieve valid lease: %v", err)
		}
		if err := client.CoordinationV1().Leases(ns).Delete(context.TODO(), lease.Name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to clean up valid lease: %v", err)
		}
	}
}

func newTestLease(acquireTime time.Time, namespace string) *coordinationv1.Lease {
	return &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testLeaseName,
			Namespace: namespace,
			Labels: map[string]string{
				controlplane.IdentityLeaseComponentLabelKey: controlplane.KubeAPIServer,
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       pointer.StringPtr(testLeaseName),
			LeaseDurationSeconds: pointer.Int32Ptr(3600),
			AcquireTime:          &metav1.MicroTime{Time: acquireTime},
			RenewTime:            &metav1.MicroTime{Time: acquireTime},
		},
	}
}
