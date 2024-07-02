/*
Copyright 2024 The Kubernetes Authors.

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
	"context"
	"embed"
	"strings"

	g "github.com/onsi/ginkgo/v2"
	o "github.com/onsi/gomega"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var (
	//go:embed e2edata
	e2eData embed.FS
)

var _ = SIGDescribe("ValidatingAdmissionPolicy", framework.WithFeatureGate(features.ServiceAccountTokenNodeBindingValidation), func() {
	defer g.GinkgoRecover()
	f := framework.NewDefaultFramework("node-authn")
	f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

	g.It("can restrict access by-node", func() {
		// Read manifests that we leave as yaml to make it clear to users how to these permissions are created.
		// These will match future docs.
		perNodeCheckValidatingAdmissionPolicy, err := e2eData.ReadFile("e2edata/per_node_validatingadmissionpolicy.yaml")
		framework.ExpectNoError(err)
		perNodeCheckValidatingAdmissionPolicyBinding, err := e2eData.ReadFile("e2edata/per_node_validatingadmissionpolicybinding.yaml")
		framework.ExpectNoError(err)
		admission := strings.ReplaceAll(string(perNodeCheckValidatingAdmissionPolicy), "e2e-ns", f.Namespace.Name)
		admissionToCreate := readValidatingAdmissionPolicyV1OrDie([]byte(admission))
		admissionBinding := strings.ReplaceAll(string(perNodeCheckValidatingAdmissionPolicyBinding), "e2e-ns", f.Namespace.Name)
		admissionBindingToCreate := readValidatingAdmissionPolicyBindingV1OrDie([]byte(admissionBinding))

		saTokenRoleBinding := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      "sa-token",
			},
			Subjects: []rbacv1.Subject{
				{
					Kind:      "ServiceAccount",
					Name:      "default",
					Namespace: f.Namespace.Name,
				},
			},
			RoleRef: rbacv1.RoleRef{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "ClusterRole",
				Name:     "edit",
			},
		}

		agnhost := imageutils.GetConfig(imageutils.Agnhost)
		sleeperPod := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      "sa-token",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "sleeper",
						Image:   agnhost.GetE2EImage(),
						Command: []string{"sleep"},
						Args:    []string{"1200"},
						SecurityContext: &v1.SecurityContext{
							AllowPrivilegeEscalation: ptr.To(false),
							Capabilities: &v1.Capabilities{
								Drop: []v1.Capability{"ALL"},
							},
						},
					},
				},
			},
			Status: v1.PodStatus{},
		})

		ctx := context.Background()

		// cleanup the ValidatingAdmissionPolicy.
		defer func() {
			_ = f.ClientSet.AdmissionregistrationV1().ValidatingAdmissionPolicies().Delete(context.Background(), admissionToCreate.Name, metav1.DeleteOptions{})
			_ = f.ClientSet.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Delete(context.Background(), admissionBindingToCreate.Name, metav1.DeleteOptions{})
		}()

		_, err = f.ClientSet.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(context.Background(), admissionToCreate, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = f.ClientSet.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(context.Background(), admissionBindingToCreate, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// create permissions that will allow unrestricted access to mutate configmaps in this namespace.
		// We limited these permissions in the step above.
		// This means the admission policy must fail closed or permissions will be too broad.
		_, err = f.ClientSet.RbacV1().RoleBindings(f.Namespace.Name).Create(context.Background(), saTokenRoleBinding, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		// run an actual pod to prove that the token is injected, not just creatable via the API
		actualPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, sleeperPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, actualPod.Name, actualPod.Namespace)
		framework.ExpectNoError(err)
		// need the pod that contains the node name
		actualPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, actualPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// get the actual projected token from the pod.
		nodeScopedSAToken, err := utils.ExecInPodWithResult(
			ctx,
			f.ClientSet.CoreV1(),
			f.ClientConfig(),
			actualPod.Namespace,
			actualPod.Name,
			"sleeper",
			[]string{"cat", "/var/run/secrets/kubernetes.io/serviceaccount/token"},
		)
		framework.ExpectNoError(err)

		// make a kubeconfig with the token and confirm the kube-apiserver has the expected claims
		nodeScopedClientConfig := rest.AnonymousClientConfig(f.ClientConfig())
		nodeScopedClientConfig.BearerToken = nodeScopedSAToken
		nodeScopedClient, err := kubernetes.NewForConfig(nodeScopedClientConfig)
		framework.ExpectNoError(err)
		saUser, err := nodeScopedClient.AuthenticationV1().SelfSubjectReviews().Create(ctx, &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		expectedUser := serviceaccount.MakeUsername(f.Namespace.Name, "default")
		o.Expect(saUser.Status.UserInfo.Username).To(o.Equal(expectedUser))
		expectedNode := authenticationv1.ExtraValue([]string{actualPod.Spec.NodeName})
		o.Expect(saUser.Status.UserInfo.Extra["authentication.kubernetes.io/node-name"]).To(o.Equal(expectedNode))

		allowedConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      actualPod.Spec.NodeName,
			},
		}
		disallowedConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      "unlikely-node",
			},
		}

		actualAllowedConfigMap, err := nodeScopedClient.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, allowedConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = nodeScopedClient.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, disallowedConfigMap, metav1.CreateOptions{})
		o.Expect(err).To(o.HaveOccurred())
		o.Expect(err.Error()).To(o.ContainSubstring("this user may only modify configmaps that belong to the node the pod is running on"))

		// now create so we can see the update cases
		actualDisallowedConfigMap, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, disallowedConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		actualAllowedConfigMap, err = nodeScopedClient.CoreV1().ConfigMaps(f.Namespace.Name).Update(ctx, actualAllowedConfigMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		_, err = nodeScopedClient.CoreV1().ConfigMaps(f.Namespace.Name).Update(ctx, actualDisallowedConfigMap, metav1.UpdateOptions{})
		o.Expect(err).To(o.HaveOccurred())
		o.Expect(err.Error()).To(o.ContainSubstring("this user may only modify configmaps that belong to the node the pod is running on"))

		// ensure that if the node claim is missing from the restricted service-account user, we reject the request
		impersonatingConfig := rest.CopyConfig(f.ClientConfig())
		impersonatingConfig.Impersonate.UserName = saUser.Status.UserInfo.Username
		impersonatingConfig.Impersonate.UID = saUser.Status.UserInfo.UID
		impersonatingConfig.Impersonate.Groups = saUser.Status.UserInfo.Groups
		impersonatingConfig.Impersonate.Extra = map[string][]string{}
		for k, v := range saUser.Status.UserInfo.Extra {
			if k == "authentication.kubernetes.io/node-name" {
				continue
			}
			currVal := append([]string{}, v...)
			impersonatingConfig.Impersonate.Extra[k] = currVal
		}
		impersonatingClient, err := kubernetes.NewForConfig(impersonatingConfig)
		framework.ExpectNoError(err)

		_, err = impersonatingClient.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, actualDisallowedConfigMap, metav1.CreateOptions{})
		o.Expect(err).To(o.HaveOccurred())
		o.Expect(err.Error()).To(o.ContainSubstring("this user must have a \"authentication.kubernetes.io/node-name\" claim"))
		_, err = impersonatingClient.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, actualAllowedConfigMap, metav1.CreateOptions{})
		o.Expect(err).To(o.HaveOccurred())
		o.Expect(err.Error()).To(o.ContainSubstring("this user must have a \"authentication.kubernetes.io/node-name\" claim"))
	})
})

var (
	readScheme = runtime.NewScheme()
	readCodecs = serializer.NewCodecFactory(readScheme)
)

func init() {
	utilruntime.Must(admissionregistrationv1.AddToScheme(readScheme))
}

func readValidatingAdmissionPolicyV1OrDie(objBytes []byte) *admissionregistrationv1.ValidatingAdmissionPolicy {
	requiredObj, err := runtime.Decode(readCodecs.UniversalDecoder(admissionregistrationv1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*admissionregistrationv1.ValidatingAdmissionPolicy)
}

func readValidatingAdmissionPolicyBindingV1OrDie(objBytes []byte) *admissionregistrationv1.ValidatingAdmissionPolicyBinding {
	requiredObj, err := runtime.Decode(readCodecs.UniversalDecoder(admissionregistrationv1.SchemeGroupVersion), objBytes)
	if err != nil {
		panic(err)
	}
	return requiredObj.(*admissionregistrationv1.ValidatingAdmissionPolicyBinding)
}
