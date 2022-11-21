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
	"os"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

var (
	webhookNameExclusionRules    = "integration-exclusion-rules-test-webhook-config"
	deploymentNameExclusionRules = "integration-exclusion-rules-test-deployment"
)

func TestWebhookExclusionRules(t *testing.T) {
	var tearDownFn kubeapiservertesting.TearDownFunc
	defer func() {
		if tearDownFn != nil {
			tearDownFn()
		}
	}()

	etcdConfig := framework.SharedEtcd()
	admissionConfigFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())

	var configFile = fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    exclusionRules:
      - apiGroups: ["*"]
        apiVersions: ["*"]
        kind: "Deployment"
        namespace: "default"
        name: "%v"
`, deploymentNameExclusionRules)

	if err := os.WriteFile(admissionConfigFile.Name(), []byte(configFile), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--admission-control-config-file=" + admissionConfigFile.Name(),
	}, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Creating Broken Webhook that will block all operations on all objects")
	_, err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.TODO(), brokenWebhookConfigExclusionRules(webhookNameExclusionRules), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to register broken webhook: %v", err)
	}

	// There is no guarantee on how long it takes the apiserver to honor the configuration and there is
	// no API to determine if the configuration is being honored, so we will just wait 10s, which is long enough
	// in most cases.
	time.Sleep(10 * time.Second)

	t.Logf("Creating Deployment which should be allowed due to exclusion rules")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeploymentExclusionRules(deploymentNameExclusionRules), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}

	t.Logf("Creating Deployment with different name which should be blocked due to webhook")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeploymentExclusionRules("test-different-name"), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}

	t.Logf("Restarting apiserver")
	tearDownFn = nil
	server.TearDownFn()
	server = kubeapiservertesting.StartTestServerOrDie(t, nil, nil, etcdConfig)
	tearDownFn = server.TearDownFn

	client, err = kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Expected the broken webhook to cause creating a deployment to fail, but it succeeded.")
	}

	// test whether the webhook still blocks requests after restarting
	t.Logf("Attempt again to create Deployment which should fail due to the webhook without exclusion rules")
	_, err = client.AppsV1().Deployments("default").Create(context.TODO(), exampleDeploymentExclusionRules(deploymentNameExclusionRules), metav1.CreateOptions{})
	if err == nil {
		t.Fatalf("Expected the broken webhook to cause creating a deployment to fail, but it succeeded.")
	}
}

func exampleDeploymentExclusionRules(name string) *appsv1.Deployment {
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

func brokenWebhookConfigExclusionRules(name string) *admissionregistrationv1.ValidatingWebhookConfiguration {
	var path string
	failurePolicy := admissionregistrationv1.Fail
	return &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "broken-webhook.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"*"},
						APIVersions: []string{"*"},
						Resources:   []string{"*/*"},
					},
				}},
				// This client config references a non existent service
				// so it should always fail.
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: "default",
						Name:      "invalid-webhook-service",
						Path:      &path,
					},
					CABundle: nil,
				},
				FailurePolicy:           &failurePolicy,
				SideEffects:             &noSideEffects,
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}
}
