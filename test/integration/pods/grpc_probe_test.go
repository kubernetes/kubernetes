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

package pods

import (
	"context"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func grpcProbePod(name string, mode *v1.GRPCProbeMode) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "container",
				Image: "fakeimage",
				LivenessProbe: &v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						GRPC: &v1.GRPCAction{
							Port: 8443,
							Mode: mode,
						},
					},
				},
			}},
		},
	}
}

func grpcProbeDeployment(name string, mode *v1.GRPCProbeMode) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To[int32](1),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": name},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "container",
						Image: "fakeimage",
						LivenessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								GRPC: &v1.GRPCAction{
									Port: 8443,
									Mode: mode,
								},
							},
						},
					}},
				},
			},
		},
	}
}

func grpcProbeMode(t *v1.Pod) *v1.GRPCProbeMode {
	return t.Spec.Containers[0].LivenessProbe.GRPC.Mode
}

func TestGRPCContainerProbeTLSPod(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-grpc-probe-tls", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	t.Run("TLS mode is persisted when the feature gate is enabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, true)

		created, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("tls-enabled", ptr.To(v1.GRPCProbeModeTLS)), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if got := grpcProbeMode(created); got == nil || *got != v1.GRPCProbeModeTLS {
			t.Fatalf("expected mode TLS to be persisted, got %v", got)
		}
	})

	t.Run("Plaintext mode is persisted when the feature gate is enabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, true)

		created, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("plaintext-enabled", ptr.To(v1.GRPCProbeModePlaintext)), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if got := grpcProbeMode(created); got == nil || *got != v1.GRPCProbeModePlaintext {
			t.Fatalf("expected mode Plaintext to be persisted, got %v", got)
		}
	})

	t.Run("mode is dropped when the feature gate is disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, false)

		created, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("tls-disabled", ptr.To(v1.GRPCProbeModeTLS)), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if got := grpcProbeMode(created); got != nil {
			t.Fatalf("expected mode to be dropped, got %v", *got)
		}

		stored, err := client.CoreV1().Pods(ns.Name).Get(context.TODO(), created.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get pod: %v", err)
		}
		if got := grpcProbeMode(stored); got != nil {
			t.Fatalf("expected mode to be dropped from storage, got %v", *got)
		}
	})

	t.Run("unsupported mode is rejected when the feature gate is enabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, true)

		_, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("unsupported-enabled", ptr.To(v1.GRPCProbeMode("Verify"))), metav1.CreateOptions{})
		assertUnsupportedGRPCModeError(t, err)
	})

	t.Run("unsupported mode is rejected when the feature gate is disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, false)

		_, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("unsupported-disabled", ptr.To(v1.GRPCProbeMode("Verify"))), metav1.CreateOptions{})
		assertUnsupportedGRPCModeError(t, err)
	})

	t.Run("mode is preserved on update when the feature gate is disabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, true)

		created, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), grpcProbePod("update-preserve", ptr.To(v1.GRPCProbeModeTLS)), metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}

		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, false)

		created.Labels = map[string]string{"updated": "true"}
		updated, err := client.CoreV1().Pods(ns.Name).Update(context.TODO(), created, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update pod: %v", err)
		}
		if got := grpcProbeMode(updated); got == nil || *got != v1.GRPCProbeModeTLS {
			t.Fatalf("expected mode TLS to be preserved on update, got %v", got)
		}
	})
}

func TestGRPCContainerProbeTLSDeployment(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "deployment-grpc-probe-tls", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, true)

	created, err := client.AppsV1().Deployments(ns.Name).Create(context.TODO(), grpcProbeDeployment("tls-enabled", ptr.To(v1.GRPCProbeModeTLS)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}
	if got := created.Spec.Template.Spec.Containers[0].LivenessProbe.GRPC.Mode; got == nil || *got != v1.GRPCProbeModeTLS {
		t.Fatalf("expected mode TLS to be persisted, got %v", got)
	}

	// Disabling the gate must not drop a mode that is already in use by the
	// existing pod template.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GRPCContainerProbeTLS, false)

	created.Labels = map[string]string{"updated": "true"}
	updated, err := client.AppsV1().Deployments(ns.Name).Update(context.TODO(), created, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to update deployment: %v", err)
	}
	if got := updated.Spec.Template.Spec.Containers[0].LivenessProbe.GRPC.Mode; got == nil || *got != v1.GRPCProbeModeTLS {
		t.Fatalf("expected mode TLS to be preserved on update, got %v", got)
	}

	// A brand-new deployment created while the gate is disabled cannot set
	// the field at all.
	dropped, err := client.AppsV1().Deployments(ns.Name).Create(context.TODO(), grpcProbeDeployment("tls-disabled", ptr.To(v1.GRPCProbeModeTLS)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}
	if got := dropped.Spec.Template.Spec.Containers[0].LivenessProbe.GRPC.Mode; got != nil {
		t.Fatalf("expected mode to be dropped, got %v", *got)
	}

	// Unsupported values are still rejected for new deployments, whether or
	// not the gate is enabled.
	_, err = client.AppsV1().Deployments(ns.Name).Create(context.TODO(), grpcProbeDeployment("unsupported-disabled", ptr.To(v1.GRPCProbeMode("Verify"))), metav1.CreateOptions{})
	assertUnsupportedGRPCModeError(t, err)
}

func assertUnsupportedGRPCModeError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected object to be rejected, but it was accepted")
	}
	if !apierrors.IsInvalid(err) {
		t.Fatalf("expected an Invalid error, got %v", err)
	}
	if !strings.Contains(err.Error(), "Unsupported value") || !strings.Contains(err.Error(), "Verify") {
		t.Fatalf("expected an unsupported value error for Verify, got %v", err)
	}
}
