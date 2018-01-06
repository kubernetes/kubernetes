/*
Copyright 2017 The Kubernetes Authors.

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
	"encoding/json"
	"strings"
	"testing"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestRun(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the server is really healthy after /healthz told us so
	t.Logf("Creating Deployment directly after being healthy")
	var replicas int32 = 1
	_, err = client.AppsV1beta1().Deployments("default").Create(&appsv1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test",
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
	})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

// TestOpenAPIDelegationChainPlumbing is a smoke test that checks for
// the existence of some representative paths from the
// apiextensions-server and the kube-aggregator server, both part of
// the delegation chain in kube-apiserver.
func TestOpenAPIDelegationChainPlumbing(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := kubeclient.RESTClient().Get().AbsPath("/swagger.json").Do()
	status := 0
	result.StatusCode(&status)
	if status != 200 {
		t.Fatalf("GET /swagger.json failed: expected status=%d, got=%d", 200, status)
	}

	raw, err := result.Raw()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	type openAPISchema struct {
		Paths map[string]interface{} `json:"paths"`
	}

	var doc openAPISchema
	err = json.Unmarshal(raw, &doc)
	if err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	matchedExtension := false
	extensionsPrefix := "/apis/" + apiextensions.GroupName

	matchedRegistration := false
	registrationPrefix := "/apis/" + apiregistration.GroupName

	for path := range doc.Paths {
		if strings.HasPrefix(path, extensionsPrefix) {
			matchedExtension = true
		}
		if strings.HasPrefix(path, registrationPrefix) {
			matchedRegistration = true
		}
		if matchedExtension && matchedRegistration {
			return
		}
	}

	if !matchedExtension {
		t.Errorf("missing path: %q", extensionsPrefix)
	}

	if !matchedRegistration {
		t.Errorf("missing path: %q", registrationPrefix)
	}
}
