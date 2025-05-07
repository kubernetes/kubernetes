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

package node

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	authorizationv1 "k8s.io/api/authorization/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe(framework.WithFeatureGate(features.KubeletFineGrainedAuthz), func() {
	f := framework.NewDefaultFramework("kubelet-authz-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("when calling kubelet API", func() {
		ginkgo.It("check /healthz enpoint is accessible via nodes/healthz RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "healthz")
			gomega.Expect(sc).To(gomega.Equal("200"))
		})
		ginkgo.It("check /healthz enpoint is accessible via nodes/proxy RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "proxy")
			gomega.Expect(sc).To(gomega.Equal("200"))
		})
		ginkgo.It("check /healthz enpoint is not accessible via nodes/configz RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "configz")
			gomega.Expect(sc).To(gomega.Equal("403"))
		})
	})
})

func runKubeletAuthzTest(ctx context.Context, f *framework.Framework, endpoint, authzSubresource string) string {
	ns := f.Namespace.Name
	saName := authzSubresource
	verb := "get"
	resource := "nodes"

	ginkgo.By(fmt.Sprintf("Creating Service Account %s/%s", ns, saName))

	_, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Create(ctx, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: ns,
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating ClusterRole with prefix %s with for %s/%s", authzSubresource, resource, authzSubresource))

	clusterRole, err := f.ClientSet.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: authzSubresource + "-",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Verbs:     []string{verb},
				Resources: []string{resource + "/" + authzSubresource},
			},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		ginkgo.By(fmt.Sprintf("Destroying ClusterRoles %q for this suite.", clusterRole.Name))
		framework.ExpectNoError(f.ClientSet.RbacV1().ClusterRoles().Delete(ctx, clusterRole.Name, metav1.DeleteOptions{}))
	}()

	subject := rbacv1.Subject{
		Kind:      rbacv1.ServiceAccountKind,
		Namespace: ns,
		Name:      saName,
	}

	ginkgo.By(fmt.Sprintf("Creating ClusterRoleBinding with ClusterRole %s with subject %s/%s", clusterRole.Name, ns, saName))

	cleanupFunc, err := e2eauth.BindClusterRole(ctx, f.ClientSet.RbacV1(), clusterRole.Name, ns, subject)
	framework.ExpectNoError(err)
	defer cleanupFunc(ctx)

	ginkgo.By("Waiting for Authorization Update.")

	err = e2eauth.WaitForAuthzUpdate(ctx, f.ClientSet.AuthorizationV1(),
		serviceaccount.MakeUsername(ns, saName),
		&authorizationv1.ResourceAttributes{
			Namespace:   ns,
			Verb:        verb,
			Resource:    resource,
			Subresource: authzSubresource,
		},
		true,
	)
	framework.ExpectNoError(err)

	pod := e2epod.NewAgnhostPod(ns, fmt.Sprintf("agnhost-pod-%s", authzSubresource), nil, nil, nil)
	pod.Spec.ServiceAccountName = saName
	pod.Spec.Containers[0].Env = []v1.EnvVar{
		{
			Name: "NODE_IP",
			ValueFrom: &v1.EnvVarSource{
				FieldRef: &v1.ObjectFieldSelector{
					FieldPath: "status.hostIP",
				},
			},
		},
	}

	ginkgo.By(fmt.Sprintf("Creating Pod %s in namespace %s with serviceaccount %s", pod.Name, pod.Namespace, pod.Spec.ServiceAccountName))

	_ = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("Running command in Pod")

	var hostWarpStart, hostWarpEnd string
	// IPv6 host must be wrapped within [] if you specify a port.
	if framework.TestContext.ClusterIsIPv6() {
		hostWarpStart = "["
		hostWarpEnd = "]"
	}

	result := e2eoutput.RunHostCmdOrDie(ns,
		pod.Name,
		fmt.Sprintf("curl -XGET -sIk -o /dev/null -w '%s' --header \"Authorization: Bearer `%s`\" https://%s$NODE_IP%s:%d/%s",
			"%{http_code}",
			"cat /var/run/secrets/kubernetes.io/serviceaccount/token",
			hostWarpStart,
			hostWarpEnd,
			ports.KubeletPort,
			endpoint))

	return result
}
