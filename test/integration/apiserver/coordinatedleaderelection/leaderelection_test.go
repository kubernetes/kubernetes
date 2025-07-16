/*
Copyright 2024 The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"fmt"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1beta1 "k8s.io/api/coordination/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubernetes "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controlplane/controller/leaderelection"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestCoordinatedLeaderElectionLeaseTransfer(t *testing.T) {
	// Reset the coordinated leader election variables after the test
	defaultLeaseDuration := leaderelection.LeaseDuration
	defaultRenewDeadline := leaderelection.RenewDeadline
	defaultRetryPeriod := leaderelection.RetryPeriod
	defer func() {
		leaderelection.LeaseDuration = defaultLeaseDuration
		leaderelection.RenewDeadline = defaultRenewDeadline
		leaderelection.RetryPeriod = defaultRetryPeriod
	}()

	// Use shorter interval for lease duration in integration test
	leaderelection.LeaseDuration = 5 * time.Second
	leaderelection.RenewDeadline = 3 * time.Second
	leaderelection.RetryPeriod = 2 * time.Second

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.CoordinatedLeaderElection, true)
	etcd := framework.SharedEtcd()

	flags := []string{fmt.Sprintf("--runtime-config=%s=true", v1beta1.SchemeGroupVersion)}
	server := apiservertesting.StartTestServerOrDie(t, apiservertesting.NewDefaultTestServerOptions(), flags, etcd)
	defer server.TearDownFn()

	config := server.ClientConfig
	clientset := kubernetes.NewForConfigOrDie(config)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err := wait.PollUntilContextTimeout(ctx, 1000*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := clientset.CoordinationV1().Leases("kube-system").Get(ctx, "leader-election-controller", metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		return lease.Spec.HolderIdentity != nil, nil
	})

	if err != nil {
		t.Fatalf("timeout waiting for Lease %s %s err: %v", "leader-election-controller", "kube-system", err)
	}

	lease, err := clientset.CoordinationV1().Leases("kube-system").Get(ctx, "leader-election-controller", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	leaseName := *lease.Spec.HolderIdentity

	server2 := apiservertesting.StartTestServerOrDie(t, apiservertesting.NewDefaultTestServerOptions(), flags, etcd)
	vap := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "cle-block-renewal"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistrationv1.RuleWithOperations{
							Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
							Rule:       admissionregistrationv1.Rule{APIGroups: []string{"coordination.k8s.io"}, APIVersions: []string{"v1"}, Resources: []string{"leases"}},
						},
					},
				},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: "object.spec.holderIdentity != '" + leaseName + "'",
			}},
		},
	}
	_, err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(ctx, vap, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	vapBinding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "cle-block-renewal"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "cle-block-renewal",
			ValidationActions: []admissionregistrationv1.ValidationAction{
				admissionregistrationv1.Deny,
			},
		},
	}

	_, err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(ctx, vapBinding, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Wait until the first apiserver releases the lease and second apiserver takes over the lock
	err = wait.PollUntilContextTimeout(ctx, 1000*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := clientset.CoordinationV1().Leases("kube-system").Get(ctx, "leader-election-controller", metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		return lease.Spec.HolderIdentity != nil && *lease.Spec.HolderIdentity != leaseName, nil
	})
	if err != nil {
		t.Error("Expected the cle lease lock to transition to the second apiserver")
	}

	// Shutdown the second apiserver
	server2.TearDownFn()

	// Allow writes again from the first apiserver
	err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(ctx, vap.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	err = clientset.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Delete(ctx, vapBinding.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Ensure that the first apiserver is able to reacquire the CLE leader lease
	err = wait.PollUntilContextTimeout(ctx, 1000*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (done bool, err error) {
		lease, err := clientset.CoordinationV1().Leases("kube-system").Get(ctx, "leader-election-controller", metav1.GetOptions{})
		if err != nil {
			fmt.Println(err)
			return false, nil
		}
		return *lease.Spec.HolderIdentity == leaseName, nil
	})
	if err != nil {
		t.Error("Expected the cle lease lock to transition to the first apiserver")
	}
}
