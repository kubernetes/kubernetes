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
	"context"
	"encoding/json"
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"encoding/base64"
	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/types"
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
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
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
		Description: Create a secret. Create a Pod with Container that declares a environment variable using 'EnvFrom' which references the secret created to extract a key value from the secret. Pod MUST have the environment variable that contains proper value for the key to the secret.
	*/
	framework.ConformanceIt("should be consumable via the environment [NodeConformance]", func() {
		name := "secret-test-" + string(uuid.NewUUID())
		secret := newEnvFromSecret(f.Namespace.Name, name)
		ginkgo.By(fmt.Sprintf("creating secret %v/%v", f.Namespace.Name, secret.Name))
		var err error
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
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

	/*
			   Release : v1.18
			   Testname: Secret patching
			   Description: A Secret is created.
		           Listing all Secrets MUST return an empty list.
		           Given the patching and fetching of the Secret, the fields MUST equal the new values.
		           The Secret is deleted by it's static Label.
		           Secrets are listed finally, the list MUST NOT include the originally created Secret.
	*/
	framework.ConformanceIt("should patch a secret", func() {
		ginkgo.By("creating a secret")

		secretTestName := "test-secret-" + string(uuid.NewUUID())

		// create a secret in the test namespace
		_, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name: secretTestName,
				Labels: map[string]string{
					"testsecret-constant": "true",
				},
			},
			Data: map[string][]byte{
				"key": []byte("value"),
			},
			Type: "Opaque",
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create secret")

		ginkgo.By("listing secrets in all namespaces to ensure that there are more than zero")
		// list all secrets in all namespaces to ensure endpoint coverage
		secretsList, err := f.ClientSet.CoreV1().Secrets("").List(context.TODO(), metav1.ListOptions{
			LabelSelector: "testsecret-constant=true",
		})
		framework.ExpectNoError(err, "failed to list secrets")
		framework.ExpectNotEqual(len(secretsList.Items), 0, "no secrets found")

		foundCreatedSecret := false
		var secretCreatedName string
		for _, val := range secretsList.Items {
			if val.ObjectMeta.Name == secretTestName && val.ObjectMeta.Namespace == f.Namespace.Name {
				foundCreatedSecret = true
				secretCreatedName = val.ObjectMeta.Name
				break
			}
		}
		framework.ExpectEqual(foundCreatedSecret, true, "unable to find secret by its value")

		ginkgo.By("patching the secret")
		// patch the secret in the test namespace
		secretPatchNewData := base64.StdEncoding.EncodeToString([]byte("value1"))
		secretPatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{"testsecret": "true"},
			},
			"data": map[string][]byte{"key": []byte(secretPatchNewData)},
		})
		framework.ExpectNoError(err, "failed to marshal JSON")
		_, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Patch(context.TODO(), secretCreatedName, types.StrategicMergePatchType, []byte(secretPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch secret")

		secret, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Get(context.TODO(), secretCreatedName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get secret")

		secretDecodedstring, err := base64.StdEncoding.DecodeString(string(secret.Data["key"]))
		framework.ExpectNoError(err, "failed to decode secret from Base64")

		framework.ExpectEqual(string(secretDecodedstring), "value1", "found secret, but the data wasn't updated from the patch")

		ginkgo.By("deleting the secret using a LabelSelector")
		err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "testsecret=true",
		})
		framework.ExpectNoError(err, "failed to delete patched secret")

		ginkgo.By("listing secrets in all namespaces, searching for label name and value in patch")
		// list all secrets in all namespaces
		secretsList, err = f.ClientSet.CoreV1().Secrets("").List(context.TODO(), metav1.ListOptions{
			LabelSelector: "testsecret-constant=true",
		})
		framework.ExpectNoError(err, "failed to list secrets")

		foundCreatedSecret = false
		for _, val := range secretsList.Items {
			if val.ObjectMeta.Name == secretTestName && val.ObjectMeta.Namespace == f.Namespace.Name {
				foundCreatedSecret = true
				break
			}
		}
		framework.ExpectEqual(foundCreatedSecret, false, "secret was not deleted successfully")
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
	return f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{})
}
