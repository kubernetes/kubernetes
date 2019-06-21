/*
Copyright 2014 The Kubernetes Authors.

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

package common

import (
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = ginkgo.Describe("[sig-api-machinery] Secrets", func() {
	f := framework.NewDefaultFramework("secrets")

	/*
		Release : v1.9
		Testname: Secrets, pod environment field
		Description: Create a secret. Create a Pod with Container that declares a environment variable which references the secret created to extract a key value from the secret. Pod MUST have the environment variable that contains proper value for the key to the secret.
	*/
	framework.ConformanceIt("should be consumable from pods in env vars [NodeConformance]", func() {
		name := "secret-test-" + string(uuid.NewUUID())
		secret := secretForTest(f.Namespace.Name, name)

		ginkgo.By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		var err error
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
			e2elog.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-secrets-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "secret-env-test",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						Env: []v1.EnvVar{
							{
								Name: "SECRET_DATA",
								ValueFrom: &v1.EnvVarSource{
									SecretKeyRef: &v1.SecretKeySelector{
										LocalObjectReference: v1.LocalObjectReference{
											Name: name,
										},
										Key: "data-1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume secrets", pod, 0, []string{
			"SECRET_DATA=value-1",
		})
	})

	/*
		Release : v1.9
		Testname: Secrets, pod environment from source
		Description: Create a secret. Create a Pod with Container that declares a environment variable using ‘EnvFrom’ which references the secret created to extract a key value from the secret. Pod MUST have the environment variable that contains proper value for the key to the secret.
	*/
	framework.ConformanceIt("should be consumable via the environment [NodeConformance]", func() {
		name := "secret-test-" + string(uuid.NewUUID())
		secret := newEnvFromSecret(f.Namespace.Name, name)
		ginkgo.By(fmt.Sprintf("creating secret %v/%v", f.Namespace.Name, secret.Name))
		var err error
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
			e2elog.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						EnvFrom: []v1.EnvFromSource{
							{
								SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								Prefix:    "p_",
								SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume secrets", pod, 0, []string{
			"data_1=value-1", "data_2=value-2", "data_3=value-3",
			"p_data_1=value-1", "p_data_2=value-2", "p_data_3=value-3",
		})
	})

	/*
	   Release : v1.15
	   Testname: Secrets, with empty-key
	   Description: Attempt to create a Secret with an empty key. The creation MUST fail.
	*/
	framework.ConformanceIt("should fail to create secret due to empty secret key", func() {
		secret, err := createEmptyKeySecretForTest(f)
		framework.ExpectError(err, "created secret %q with empty key in namespace %q", secret.Name, f.Namespace.Name)
	})
})

func newEnvFromSecret(namespace, name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string][]byte{
			"data_1": []byte("value-1\n"),
			"data_2": []byte("value-2\n"),
			"data_3": []byte("value-3\n"),
		},
	}
}

func createEmptyKeySecretForTest(f *framework.Framework) (*v1.Secret, error) {
	secretName := "secret-emptykey-test-" + string(uuid.NewUUID())
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      secretName,
		},
		Data: map[string][]byte{
			"": []byte("value-1\n"),
		},
	}
	ginkgo.By(fmt.Sprintf("Creating projection with secret that has name %s", secret.Name))
	return f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret)
}
