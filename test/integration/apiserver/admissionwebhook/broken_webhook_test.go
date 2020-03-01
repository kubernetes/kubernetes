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

package admissionwebhook

import (
	"context"
	"fmt"
	"testing"
	"time"

	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
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
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Creating Deployment to ensure apiserver is functional")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeployment(generateDeploymentName(0)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}

	t.Logf("Creating Broken Webhook that will block all operations on all objects")
	_, err = client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations().Create(context.TODO(), brokenWebhookConfig(brokenWebhookName), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to register broken webhook: %v", err)
	}

	// There is no guarantee on how long it takes the apiserver to honor the configuration and there is
	// no API to determine if the configuration is being honored, so we will just wait 10s, which is long enough
	// in most cases.
	time.Sleep(10 * time.Second)

	// test whether the webhook blocks requests
	t.Logf("Attempt to create Deployment which should fail due to the webhook")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeployment(generateDeploymentName(1)), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected the broken webhook to cause creating a deployment to fail, but it succeeded.")
	}

	t.Logf("Restarting apiserver")
	tearDownFn = nil
	server.TearDownFn()
	server = kubeapiservertesting.StartTestServerOrDie(t, nil, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err = kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the webhook still blocks requests after restarting
	t.Logf("Attempt again to create Deployment which should fail due to the webhook")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeployment(generateDeploymentName(2)), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected the broken webhook to cause creating a deployment to fail, but it succeeded.")
	}

	t.Logf("Deleting the broken webhook to fix the cluster")
	err = client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations().Delete(context.TODO(), brokenWebhookName, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete broken webhook: %v", err)
	}

	// The webhook deletion is honored in 10s.
	time.Sleep(10 * time.Second)

	// test if the deleted webhook no longer blocks requests
	t.Logf("Creating Deployment to ensure webhook is deleted")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeployment(generateDeploymentName(3)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

func generateDeploymentName(suffix int) string {
	return fmt.Sprintf("%v-%v", deploymentNamePrefix, suffix)
}

func exampleDeployment(name string) *appsv1.Deployment {
	var replicas int32 = 1
	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      name,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
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
	failurePolicy := admissionregistrationv1beta1.Fail
	return &admissionregistrationv1beta1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Webhooks: []admissionregistrationv1beta1.ValidatingWebhook{
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
