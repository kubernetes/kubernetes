/*
Copyright 2018 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"testing"
	"time"

	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

var (
	brokenWebhookName    = "integration-broken-webhook-test-webhook-config"
	deploymentNamePrefix = "integration-broken-webhook-test-deployment"
)

func TestBrokenWebhook(t *testing.T) {
	var tearDownFn kubeapiservertesting.TearDownFunc
	defer func() {
		if tearDownFn != nil {
			tearDownFn()
		}
	}()

	etcdConfig := framework.SharedEtcd()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Creating Deployment to ensure apiserver is functional")
	_, err = client.AppsV1beta1().Deployments("default").Create(exampleDeployment(generateDeploymentName(0)))
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}

	t.Logf("Creating Broken Webhook that will block all operations on all objects")
	_, err = client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations().Create(brokenWebhookConfig(brokenWebhookName))
	if err != nil {
		t.Fatalf("Failed to register broken webhook: %v", err)
	}

	// The webhook configuration is honored in 10s.
	time.Sleep(10 * time.Second)

	// test whether the webhook blocks requests
	t.Logf("Attempt to create Deployment which should fail due to the webhook")
	_, err = client.AppsV1beta1().Deployments("default").Create(exampleDeployment(generateDeploymentName(1)))
	if err == nil {
		t.Fatalf("Expected to broken webhook to cause creating a deployment to fail, but it succeeded.")
	}

	t.Logf("Restarting apiserver")
	tearDownFn = nil
	server.TearDownFn()
	server = kubeapiservertesting.StartTestServerOrDie(t, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err = kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the webhook still blocks requests after restarting
	t.Logf("Attempt again to create Deployment which should fail due to the webhook")
	_, err = client.AppsV1beta1().Deployments("default").Create(exampleDeployment(generateDeploymentName(2)))
	if err == nil {
		t.Fatalf("Expected to broken webhook to cause creating a deployment to fail, but it succeeded.")
	}

	t.Logf("Deleting the broken webhook to fix the cluster")
	err = client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations().Delete(brokenWebhookName, nil)
	if err != nil {
		t.Fatalf("Failed to delete broken webhook: %v", err)
	}

	// The webhook deletion is honored in 10s.
	time.Sleep(10 * time.Second)

	// test if the deleted webhook no longer blocks requests
	t.Logf("Creating Deployment to ensure webhook is deleted")
	_, err = client.AppsV1beta1().Deployments("default").Create(exampleDeployment(generateDeploymentName(3)))
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

func generateDeploymentName(suffix int) string {
	return fmt.Sprintf("%v-%v", deploymentNamePrefix, suffix)
}

func exampleDeployment(name string) *appsv1beta1.Deployment {
	var replicas int32 = 1
	return &appsv1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      name,
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: &replicas,
			Strategy: appsv1beta1.DeploymentStrategy{
				Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "foo",
							Image: "foo",
						},
					},
				},
			},
		},
	}
}

func brokenWebhookConfig(name string) *admissionregistrationv1beta1.ValidatingWebhookConfiguration {
	var path string
	var failurePolicy admissionregistrationv1beta1.FailurePolicyType = admissionregistrationv1beta1.Fail
	return &admissionregistrationv1beta1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Webhooks: []admissionregistrationv1beta1.Webhook{
			{
				Name: "broken-webhook.k8s.io",
				Rules: []admissionregistrationv1beta1.RuleWithOperations{{
					Operations: []admissionregistrationv1beta1.OperationType{admissionregistrationv1beta1.OperationAll},
					Rule: admissionregistrationv1beta1.Rule{
						APIGroups:   []string{"*"},
						APIVersions: []string{"*"},
						Resources:   []string{"*/*"},
					},
				}},
				// This client config references a non existent service
				// so it should always fail.
				ClientConfig: admissionregistrationv1beta1.WebhookClientConfig{
					Service: &admissionregistrationv1beta1.ServiceReference{
						Namespace: "default",
						Name:      "invalid-webhook-service",
						Path:      &path,
					},
					CABundle: nil,
				},
				FailurePolicy: &failurePolicy,
			},
		},
	}
}
