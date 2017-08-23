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

package auth

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var serviceAccountTokenNamespaceVersion = utilversion.MustParseSemantic("v1.2.0")

var serviceAccountTokenAutomountVersion = utilversion.MustParseSemantic("v1.6.0-alpha.2")

var _ = SIGDescribe("ServiceAccounts", func() {
	f := framework.NewDefaultFramework("svcaccounts")

	It("should ensure a single API token exists", func() {
		// wait for the service account to reference a single secret
		var secrets []v1.ObjectReference
		framework.ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*10, func() (bool, error) {
			By("waiting for a single token reference")
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				framework.Logf("default service account was not found")
				return false, nil
			}
			if err != nil {
				framework.Logf("error getting default service account: %v", err)
				return false, err
			}
			switch len(sa.Secrets) {
			case 0:
				framework.Logf("default service account has no secret references")
				return false, nil
			case 1:
				framework.Logf("default service account has a single secret reference")
				secrets = sa.Secrets
				return true, nil
			default:
				return false, fmt.Errorf("default service account has too many secret references: %#v", sa.Secrets)
			}
		}))

		// make sure the reference doesn't flutter
		{
			By("ensuring the single token reference persists")
			time.Sleep(2 * time.Second)
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(sa.Secrets).To(Equal(secrets))
		}

		// delete the referenced secret
		By("deleting the service account token")
		framework.ExpectNoError(f.ClientSet.Core().Secrets(f.Namespace.Name).Delete(secrets[0].Name, nil))

		// wait for the referenced secret to be removed, and another one autocreated
		framework.ExpectNoError(wait.Poll(time.Millisecond*500, framework.ServiceAccountProvisionTimeout, func() (bool, error) {
			By("waiting for a new token reference")
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			if err != nil {
				framework.Logf("error getting default service account: %v", err)
				return false, err
			}
			switch len(sa.Secrets) {
			case 0:
				framework.Logf("default service account has no secret references")
				return false, nil
			case 1:
				if sa.Secrets[0] == secrets[0] {
					framework.Logf("default service account still has the deleted secret reference")
					return false, nil
				}
				framework.Logf("default service account has a new single secret reference")
				secrets = sa.Secrets
				return true, nil
			default:
				return false, fmt.Errorf("default service account has too many secret references: %#v", sa.Secrets)
			}
		}))

		// make sure the reference doesn't flutter
		{
			By("ensuring the single token reference persists")
			time.Sleep(2 * time.Second)
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(sa.Secrets).To(Equal(secrets))
		}

		// delete the reference from the service account
		By("deleting the reference to the service account token")
		{
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			framework.ExpectNoError(err)
			sa.Secrets = nil
			_, updateErr := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Update(sa)
			framework.ExpectNoError(updateErr)
		}

		// wait for another one to be autocreated
		framework.ExpectNoError(wait.Poll(time.Millisecond*500, framework.ServiceAccountProvisionTimeout, func() (bool, error) {
			By("waiting for a new token to be created and added")
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			if err != nil {
				framework.Logf("error getting default service account: %v", err)
				return false, err
			}
			switch len(sa.Secrets) {
			case 0:
				framework.Logf("default service account has no secret references")
				return false, nil
			case 1:
				framework.Logf("default service account has a new single secret reference")
				secrets = sa.Secrets
				return true, nil
			default:
				return false, fmt.Errorf("default service account has too many secret references: %#v", sa.Secrets)
			}
		}))

		// make sure the reference doesn't flutter
		{
			By("ensuring the single token reference persists")
			time.Sleep(2 * time.Second)
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(sa.Secrets).To(Equal(secrets))
		}
	})

	It("should mount an API token into pods [Conformance]", func() {
		var tokenContent string
		var rootCAContent string

		// Standard get, update retry loop
		framework.ExpectNoError(wait.Poll(time.Millisecond*500, framework.ServiceAccountProvisionTimeout, func() (bool, error) {
			By("getting the auto-created API token")
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get("default", metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				framework.Logf("default service account was not found")
				return false, nil
			}
			if err != nil {
				framework.Logf("error getting default service account: %v", err)
				return false, err
			}
			if len(sa.Secrets) == 0 {
				framework.Logf("default service account has no secret references")
				return false, nil
			}
			for _, secretRef := range sa.Secrets {
				secret, err := f.ClientSet.Core().Secrets(f.Namespace.Name).Get(secretRef.Name, metav1.GetOptions{})
				if err != nil {
					framework.Logf("Error getting secret %s: %v", secretRef.Name, err)
					continue
				}
				if secret.Type == v1.SecretTypeServiceAccountToken {
					tokenContent = string(secret.Data[v1.ServiceAccountTokenKey])
					rootCAContent = string(secret.Data[v1.ServiceAccountRootCAKey])
					return true, nil
				}
			}

			framework.Logf("default service account has no secret references to valid service account tokens")
			return false, nil
		}))

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "pod-service-account-" + string(uuid.NewUUID()) + "-",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "token-test",
						Image: "gcr.io/google_containers/mounttest:0.8",
						Args: []string{
							fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountTokenKey),
						},
					},
					{
						Name:  "root-ca-test",
						Image: "gcr.io/google_containers/mounttest:0.8",
						Args: []string{
							fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountRootCAKey),
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		supportsTokenNamespace, _ := framework.ServerVersionGTE(serviceAccountTokenNamespaceVersion, f.ClientSet.Discovery())
		if supportsTokenNamespace {
			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name:  "namespace-test",
				Image: "gcr.io/google_containers/mounttest:0.8",
				Args: []string{
					fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountNamespaceKey),
				},
			})
		}

		f.TestContainerOutput("consume service account token", pod, 0, []string{
			fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountTokenKey, tokenContent),
		})
		f.TestContainerOutput("consume service account root CA", pod, 1, []string{
			fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountRootCAKey, rootCAContent),
		})

		if supportsTokenNamespace {
			f.TestContainerOutput("consume service account namespace", pod, 2, []string{
				fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, v1.ServiceAccountNamespaceKey, f.Namespace.Name),
			})
		}
	})

	It("should allow opting out of API token automount [Conformance]", func() {
		framework.SkipUnlessServerVersionGTE(serviceAccountTokenAutomountVersion, f.ClientSet.Discovery())

		var err error
		trueValue := true
		falseValue := false
		mountSA := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "mount"}, AutomountServiceAccountToken: &trueValue}
		nomountSA := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "nomount"}, AutomountServiceAccountToken: &falseValue}
		mountSA, err = f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Create(mountSA)
		framework.ExpectNoError(err)
		nomountSA, err = f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Create(nomountSA)
		framework.ExpectNoError(err)

		// Standard get, update retry loop
		framework.ExpectNoError(wait.Poll(time.Millisecond*500, framework.ServiceAccountProvisionTimeout, func() (bool, error) {
			By("getting the auto-created API token")
			sa, err := f.ClientSet.Core().ServiceAccounts(f.Namespace.Name).Get(mountSA.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				framework.Logf("mount service account was not found")
				return false, nil
			}
			if err != nil {
				framework.Logf("error getting mount service account: %v", err)
				return false, err
			}
			if len(sa.Secrets) == 0 {
				framework.Logf("mount service account has no secret references")
				return false, nil
			}
			for _, secretRef := range sa.Secrets {
				secret, err := f.ClientSet.Core().Secrets(f.Namespace.Name).Get(secretRef.Name, metav1.GetOptions{})
				if err != nil {
					framework.Logf("Error getting secret %s: %v", secretRef.Name, err)
					continue
				}
				if secret.Type == v1.SecretTypeServiceAccountToken {
					return true, nil
				}
			}

			framework.Logf("default service account has no secret references to valid service account tokens")
			return false, nil
		}))

		testcases := []struct {
			PodName            string
			ServiceAccountName string
			AutomountPodSpec   *bool
			ExpectTokenVolume  bool
		}{
			{
				PodName:            "pod-service-account-defaultsa",
				ServiceAccountName: "default",
				AutomountPodSpec:   nil,
				ExpectTokenVolume:  true, // default is true
			},
			{
				PodName:            "pod-service-account-mountsa",
				ServiceAccountName: mountSA.Name,
				AutomountPodSpec:   nil,
				ExpectTokenVolume:  true,
			},
			{
				PodName:            "pod-service-account-nomountsa",
				ServiceAccountName: nomountSA.Name,
				AutomountPodSpec:   nil,
				ExpectTokenVolume:  false,
			},

			// Make sure pod spec trumps when opting in
			{
				PodName:            "pod-service-account-defaultsa-mountspec",
				ServiceAccountName: "default",
				AutomountPodSpec:   &trueValue,
				ExpectTokenVolume:  true,
			},
			{
				PodName:            "pod-service-account-mountsa-mountspec",
				ServiceAccountName: mountSA.Name,
				AutomountPodSpec:   &trueValue,
				ExpectTokenVolume:  true,
			},
			{
				PodName:            "pod-service-account-nomountsa-mountspec",
				ServiceAccountName: nomountSA.Name,
				AutomountPodSpec:   &trueValue,
				ExpectTokenVolume:  true, // pod spec trumps
			},

			// Make sure pod spec trumps when opting out
			{
				PodName:            "pod-service-account-defaultsa-nomountspec",
				ServiceAccountName: "default",
				AutomountPodSpec:   &falseValue,
				ExpectTokenVolume:  false, // pod spec trumps
			},
			{
				PodName:            "pod-service-account-mountsa-nomountspec",
				ServiceAccountName: mountSA.Name,
				AutomountPodSpec:   &falseValue,
				ExpectTokenVolume:  false, // pod spec trumps
			},
			{
				PodName:            "pod-service-account-nomountsa-nomountspec",
				ServiceAccountName: nomountSA.Name,
				AutomountPodSpec:   &falseValue,
				ExpectTokenVolume:  false, // pod spec trumps
			},
		}

		for _, tc := range testcases {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: tc.PodName},
				Spec: v1.PodSpec{
					Containers:                   []v1.Container{{Name: "token-test", Image: "gcr.io/google_containers/mounttest:0.8"}},
					RestartPolicy:                v1.RestartPolicyNever,
					ServiceAccountName:           tc.ServiceAccountName,
					AutomountServiceAccountToken: tc.AutomountPodSpec,
				},
			}
			createdPod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			framework.ExpectNoError(err)
			framework.Logf("created pod %s", tc.PodName)

			hasServiceAccountTokenVolume := false
			for _, c := range createdPod.Spec.Containers {
				for _, vm := range c.VolumeMounts {
					if vm.MountPath == serviceaccount.DefaultAPITokenMountPath {
						hasServiceAccountTokenVolume = true
					}
				}
			}

			if hasServiceAccountTokenVolume != tc.ExpectTokenVolume {
				framework.Failf("%s: expected volume=%v, got %v (%#v)", tc.PodName, tc.ExpectTokenVolume, hasServiceAccountTokenVolume, createdPod)
			} else {
				framework.Logf("pod %s service account token volume mount: %v", tc.PodName, hasServiceAccountTokenVolume)
			}
		}
	})
})
