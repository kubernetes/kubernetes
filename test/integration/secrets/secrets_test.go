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
	"context"
	"testing"

	"encoding/json"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func deleteSecretOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.CoreV1().Secrets(ns).Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
		t.Errorf("unable to delete secret %v: %v", name, err)
	}
}

// TestSecrets tests apiserver-side behavior of creation of secret objects and their use by pods.
func TestSecrets(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "secret", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	DoTestSecrets(t, client, ns)
	DoTestSecretsImmutableWithEmptyValue(t, client, ns)
}

// DoTestSecretsImmutableWithEmptyValue Test whether apiserver will judge the secret data inconsistency
// if the value of map in secret data is the empty string when patch secret,
// and if the Immutable value of secret is true, the patch request is rejected.
func DoTestSecretsImmutableWithEmptyValue(t *testing.T, client clientset.Interface, ns *v1.Namespace) {
	// Make a secret object. Make a secret object. the map in the secret data contains the value of the empty string
	// make Immutable true, so that if the apiserver judge that the patch secret is inconsistent, it will refuse the patch
	trueVal := true
	s := v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret",
			Namespace: ns.Name,
		},
		Immutable: &trueVal,
		Data: map[string][]byte{
			"emptyData": {},
		},
	}

	if _, err := client.CoreV1().Secrets(s.Namespace).Create(context.TODO(), &s, metav1.CreateOptions{}); err != nil {
		t.Errorf("unable to create test secret: %v", err)
	}
	defer deleteSecretOrErrorf(t, client, s.Namespace, s.Name)

	// Make a patch secret object
	patchSecret := v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "secret",
			Namespace: ns.Name,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Immutable: &trueVal,
		Data: map[string][]byte{
			"emptyData": {},
		},
	}

	secretPatch, err := json.Marshal(patchSecret)
	if err != nil {
		t.Errorf("unable to marshal test secret: %v", err)
	}

	// Patch secret object, expect patch to succeed,
	// more detailed discussion is at #119229
	if _, err := client.CoreV1().Secrets(s.Namespace).Patch(context.TODO(), patchSecret.Name, types.StrategicMergePatchType, secretPatch, metav1.PatchOptions{}); err != nil {
		t.Errorf("unable to patch test secret: %v", err)
	}
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

	if _, err := client.CoreV1().Secrets(s.Namespace).Create(context.TODO(), &s, metav1.CreateOptions{}); err != nil {
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
	if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)

	// Create a pod that consumes non-existent secret.
	pod.ObjectMeta.Name = "uses-non-existent-secret"
	if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	// This pod may fail to run, but we don't currently prevent this, and this
	// test can't check whether the kubelet actually pulls the secret.

	// Verifying contents of the volumes is out of scope for a
	// apiserver<->kubelet integration test.  It is covered by an e2e test.
}
