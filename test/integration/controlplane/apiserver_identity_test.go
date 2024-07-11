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
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/cryptobyte"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

const (
	testLeaseName = "apiserver-lease-test"
)

func expectedAPIServerIdentity(t *testing.T, hostname string) string {
	b := cryptobyte.NewBuilder(nil)
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte(hostname))
	})
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes([]byte("kube-apiserver"))
	})
	hashData, err := b.Bytes()
	if err != nil {
		t.Fatalf("error building hash data for apiserver identity: %v", err)
	}

	hash := sha256.Sum256(hashData)
	return "apiserver-" + strings.ToLower(base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(hash[:16]))
}

func TestCreateLeaseOnStart(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	hostname, err := os.Hostname()
	if err != nil {
		t.Fatalf("Unexpected error getting apiserver hostname: %v", err)
	}

	t.Logf(`Waiting the kube-apiserver Lease to be created`)
	if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
		leases, err := kubeclient.
			CoordinationV1().
			Leases(metav1.NamespaceSystem).
			List(context.TODO(), metav1.ListOptions{LabelSelector: controlplaneapiserver.IdentityLeaseComponentLabelKey + "=" + controlplane.KubeAPIServer})
		if err != nil {
			return false, err
		}

		if leases == nil {
			return false, nil
		}

		if len(leases.Items) != 1 {
			return false, nil
		}

		lease := leases.Items[0]
		if lease.Name != expectedAPIServerIdentity(t, hostname) {
			return false, fmt.Errorf("unexpected apiserver identity, got: %v, expected: %v", lease.Name, expectedAPIServerIdentity(t, hostname))
		}

		if lease.Labels[corev1.LabelHostname] != hostname {
			return false, fmt.Errorf("unexpected hostname label, got: %v, expected: %v", lease.Labels[corev1.LabelHostname], hostname)
		}

		return true, nil
	}); err != nil {
		t.Fatalf("Failed to see the kube-apiserver lease: %v", err)
	}
}

func TestLeaseGarbageCollection(t *testing.T) {
	oldIdentityLeaseDurationSeconds := controlplaneapiserver.IdentityLeaseDurationSeconds
	oldIdentityLeaseGCPeriod := controlplaneapiserver.IdentityLeaseGCPeriod
	oldIdentityLeaseRenewIntervalPeriod := controlplaneapiserver.IdentityLeaseRenewIntervalPeriod
	defer func() {
		// reset the default values for leases after this test
		controlplaneapiserver.IdentityLeaseDurationSeconds = oldIdentityLeaseDurationSeconds
		controlplaneapiserver.IdentityLeaseGCPeriod = oldIdentityLeaseGCPeriod
		controlplaneapiserver.IdentityLeaseRenewIntervalPeriod = oldIdentityLeaseRenewIntervalPeriod
	}()

	// Shorten lease parameters so GC behavior can be exercised in integration tests
	controlplaneapiserver.IdentityLeaseDurationSeconds = 1
	controlplaneapiserver.IdentityLeaseGCPeriod = time.Second
	controlplaneapiserver.IdentityLeaseRenewIntervalPeriod = time.Second

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
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

	// identity leases (with apiserver.kubernetes.io/identity label) created in user namespaces should not be GC'ed
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
				controlplaneapiserver.IdentityLeaseComponentLabelKey: controlplane.KubeAPIServer,
			},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       pointer.String(testLeaseName),
			LeaseDurationSeconds: pointer.Int32(3600),
			AcquireTime:          &metav1.MicroTime{Time: acquireTime},
			RenewTime:            &metav1.MicroTime{Time: acquireTime},
		},
	}
}
