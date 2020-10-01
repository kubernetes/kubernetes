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

package master

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCreateNamespaceAndLeaseOnStart(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.APIServerIdentity, true)()
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf(`Waiting the kube-apiserver lease Namespace to be created`)
	if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
		if _, err := kubeclient.CoreV1().Namespaces().Get(context.TODO(), v1.NamespaceAPIServerLease, metav1.GetOptions{}); err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed to see the kube-apiserver lease Namespace: %v", err)
	}
	t.Logf(`Waiting the kube-apiserver Lease to be created`)
	if err := wait.PollImmediate(500*time.Millisecond, 10*time.Second, func() (bool, error) {
		leases, err := kubeclient.CoordinationV1().Leases(v1.NamespaceAPIServerLease).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if leases != nil && len(leases.Items) == 1 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("Failed to see the kube-apiserver lease: %v", err)
	}
}
