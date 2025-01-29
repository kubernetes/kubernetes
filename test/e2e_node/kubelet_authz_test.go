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

package e2enode

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	authenticationv1 "k8s.io/api/authentication/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
)

var _ = SIGDescribe("Kubelet Authz", feature.KubeletFineGrainedAuthz, func() {
	f := framework.NewDefaultFramework("kubelet-authz-test")
	ginkgo.Context("when calling kubelet API", func() {
		ginkgo.It("check /healthz enpoint is accessible via nodes/healthz RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "healthz")
			gomega.Expect(sc).To(gomega.Equal(http.StatusOK))
		})
		ginkgo.It("check /healthz enpoint is accessible via nodes/proxy RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "proxy")
			gomega.Expect(sc).To(gomega.Equal(http.StatusOK))
		})
		ginkgo.It("check /healthz enpoint is not accessible via nodes/configz RBAC", func(ctx context.Context) {
			sc := runKubeletAuthzTest(ctx, f, "healthz", "configz")
			gomega.Expect(sc).To(gomega.Equal(http.StatusUnauthorized))
		})
	})
})

func runKubeletAuthzTest(ctx context.Context, f *framework.Framework, endpoint, authzSubresource string) int {
	ns := f.Namespace.Name
	saName := authzSubresource
	crName := authzSubresource
	verb := "get"
	resource := "nodes"

	ginkgo.By(fmt.Sprintf("Creating Service Account:%s/%s", ns, saName))

	_, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Create(ctx, &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      saName,
			Namespace: ns,
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating ClusterRole %s with for %s/%s", crName, resource, authzSubresource))

	_, err = f.ClientSet.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: crName,
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

	subject := rbacv1.Subject{
		Kind:      rbacv1.ServiceAccountKind,
		Namespace: ns,
		Name:      saName,
	}

	ginkgo.By(fmt.Sprintf("Creating ClusterRoleBinding with ClusterRole %s with subject %s/%s", crName, ns, saName))

	err = e2eauth.BindClusterRole(ctx, f.ClientSet.RbacV1(), crName, ns, subject)
	framework.ExpectNoError(err)

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

	ginkgo.By(fmt.Sprintf("Getting token for ServiceAccount %s/%s.", ns, saName))

	tr, err := f.ClientSet.CoreV1().ServiceAccounts(ns).CreateToken(ctx, saName, &authenticationv1.TokenRequest{}, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	resp, err := healthCheck(fmt.Sprintf("https://127.0.0.1:%d/%s", ports.KubeletPort, endpoint), tr.Status.Token)
	framework.ExpectNoError(err)
	return resp.StatusCode
}

func healthCheck(url, token string) (*http.Response, error) {
	insecureTransport := http.DefaultTransport.(*http.Transport).Clone()
	insecureTransport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	insecureHTTPClient := &http.Client{
		Transport: insecureTransport,
	}

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	return insecureHTTPClient.Do(req)
}
