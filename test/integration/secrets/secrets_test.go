/*
Copyright 2015 The Kubernetes Authors.

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

package secrets

// This file tests use of the secrets API resource.

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func deleteSecretOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.Core().Secrets(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete secret %v: %v", name, err)
	}
}

// TestSecrets tests apiserver-side behavior of creation of secret objects and their use by pods.
func TestSecrets(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()}})

	ns := framework.CreateTestingNamespace("secret", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	DoTestSecrets(t, client, ns)
}

// DoTestSecrets test secrets for one api version.
func DoTestSecrets(t *testing.T, client clientset.Interface, ns *v1.Namespace) {
	// Make a secret object.
	s := v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret",
			Namespace: ns.Name,
		},
		Data: map[string][]byte{
			"data": []byte("value1\n"),
		},
	}

	if _, err := client.Core().Secrets(s.Namespace).Create(&s); err != nil {
		t.Errorf("unable to create test secret: %v", err)
	}
	defer deleteSecretOrErrorf(t, client, s.Namespace, s.Name)

	// Template for pods that use a secret.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "XXX",
			Namespace: ns.Name,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "secvol",
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{
							SecretName: "secret",
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "secvol",
							MountPath: "/fake/path",
							ReadOnly:  true,
						},
					},
				},
			},
		},
	}

	// Create a pod to consume secret.
	pod.ObjectMeta.Name = "uses-secret"
	if _, err := client.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)

	// Create a pod that consumes non-existent secret.
	pod.ObjectMeta.Name = "uses-non-existent-secret"
	if _, err := client.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	// This pod may fail to run, but we don't currently prevent this, and this
	// test can't check whether the kubelet actually pulls the secret.

	// Verifying contents of the volumes is out of scope for a
	// apiserver<->kubelet integration test.  It is covered by an e2e test.
}
