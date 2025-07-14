/*
Copyright 2025 The Kubernetes Authors.

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
	o "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"
	"time"
)

var _ = SIGDescribe(framework.WithFeatureGate(features.ConstrainedImpersonation), framework.WithFeatureGate(features.ConstrainedImpersonation), func() {
	const commonName = "tester-impersonation"

	f := framework.NewDefaultFramework("constrainedimperonsation")

	/*
	   Release: v1.34
	   Testname: Service Account Impersonates Node It Is Running On
	   Description: Ensure that Service Account with correct constrained impersonation permission
	   is able to impersonate the node the service account is on.
	*/
	framework.ConformanceIt("should impersonate the scheduled node", func(ctx context.Context) {
		// run an actual pod to get sa token
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

		actualPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, sleeperPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, actualPod.Name, actualPod.Namespace)
		framework.ExpectNoError(err)
		// need the pod that contains the node name
		actualPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, actualPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// get the actual projected token from the pod.
		nodeScopedSAToken, stderr, err := e2epod.ExecWithOptionsContext(ctx, f, e2epod.ExecOptions{
			Command:            []string{"cat", "/var/run/secrets/kubernetes.io/serviceaccount/token"},
			Namespace:          actualPod.Namespace,
			PodName:            actualPod.Name,
			ContainerName:      actualPod.Spec.Containers[0].Name,
			CaptureStdout:      true,
			CaptureStderr:      true,
			PreserveWhitespace: true,
		})
		framework.ExpectNoError(err)
		o.Expect(stderr).To(o.BeEmpty(), "stderr from cat")

		nodeScopedClientConfig := rest.AnonymousClientConfig(f.ClientConfig())
		nodeScopedClientConfig.BearerToken = nodeScopedSAToken
		nodeScopedClientConfig.Impersonate.UserName = "system:node:" + actualPod.Spec.NodeName
		nodeScopedClientConfig.Impersonate.Groups = []string{user.NodesGroup}
		nodeScopedClient, err := kubernetes.NewForConfig(nodeScopedClientConfig)
		framework.ExpectNoError(err)

		_, err = nodeScopedClient.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{
			FieldSelector: "spec.nodeName=" + actualPod.Spec.NodeName,
		})
		o.Expect(apierrors.IsForbidden(err)).To(o.BeTrue())

		// Grant permissions to the service account
		clusterRole, err := f.ClientSet.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			Rules: []rbacv1.PolicyRule{
				{Verbs: []string{"impersonate:scheduled-node"}, APIGroups: []string{"authentication.k8s.io"}, Resources: []string{"nodes"}},
				{Verbs: []string{"impersonate-on:list"}, APIGroups: []string{""}, Resources: []string{"pods"}},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			framework.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				framework.ExpectNoError(f.ClientSet.RbacV1().ClusterRoles().Delete(ctx, clusterRole.Name, metav1.DeleteOptions{}))
			}()
		}

		clusterRoleBinding, err := f.ClientSet.RbacV1().ClusterRoleBindings().Create(ctx, &rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{GenerateName: commonName + "-"},
			RoleRef:    rbacv1.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole", Name: clusterRole.Name},
			Subjects:   []rbacv1.Subject{{Kind: "ServiceAccount", Name: "default", Namespace: f.Namespace.Name}},
		}, metav1.CreateOptions{})
		if err != nil {
			// Tolerate RBAC not being enabled
			framework.Logf("error granting permissions to %s, create certificatesigningrequests permissions must be granted out of band: %v", commonName, err)
		} else {
			defer func() {
				framework.ExpectNoError(f.ClientSet.RbacV1().ClusterRoleBindings().Delete(ctx, clusterRoleBinding.Name, metav1.DeleteOptions{}))
			}()
		}

		framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, 1*time.Second, time.Minute, false, func(waitCtx context.Context) (bool, error) {
			// pod list is allowed with correct permission.
			_, err = nodeScopedClient.CoreV1().Pods(f.Namespace.Name).List(waitCtx, metav1.ListOptions{
				FieldSelector: "spec.nodeName=" + actualPod.Spec.NodeName,
			})
			if apierrors.IsForbidden(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
			return true, nil
		}))

		// pod watch is not allowed
		_, err = nodeScopedClient.CoreV1().Pods(f.Namespace.Name).Watch(ctx, metav1.ListOptions{
			FieldSelector: "spec.nodeName=" + actualPod.Spec.NodeName,
		})
		o.Expect(apierrors.IsForbidden(err)).To(o.BeTrue())
	})
})
