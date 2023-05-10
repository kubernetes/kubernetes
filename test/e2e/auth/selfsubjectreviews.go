/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1alpha1 "k8s.io/api/authentication/v1alpha1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("SelfSubjectReview", func() {
	f := framework.NewDefaultFramework("selfsubjectreviews")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
			Release: v1.28
			Testname: SelfSubjectReview API
			Description:
			The authentication.k8s.io API group MUST exist in the /apis discovery document.
			The authentication.k8s.io/v1alpha1 API group/version MUST exist in the /apis/mode.k8s.io discovery document.
		    The authentication.k8s.io/v1beta1 API group/version MUST exist in the /apis/mode.k8s.io discovery document.
			The authentication.k8s.io/v1 API group/version MUST exist in the /apis/mode.k8s.io discovery document.
			The selfsubjectreviews resource MUST exist in the /apis/authentication.k8s.io/v1alpha1 discovery document.
			The selfsubjectreviews resource MUST exist in the /apis/authentication.k8s.io/v1beta1 discovery document.
			The selfsubjectreviews resource MUST exist in the /apis/authentication.k8s.io/v1 discovery document.
			The selfsubjectreviews resource MUST support create.
	*/
	ginkgo.DescribeTable(
		"testing SSR in different API groups",
		func(ctx context.Context, apiVersion, gv string) {
			// Discovery
			ginkgo.By("getting /apis")
			{
				discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
				framework.ExpectNoError(err)
				found := false
				for _, group := range discoveryGroups.Groups {
					if group.Name == authenticationv1alpha1.GroupName {
						for _, version := range group.Versions {
							if version.Version == apiVersion {
								found = true
								break
							}
						}
					}
				}
				if !found {
					ginkgo.Skip(fmt.Sprintf("expected SelfSubjectReview API group/version, got %#v", discoveryGroups.Groups))
				}
			}

			ginkgo.By("getting /apis/authentication.k8s.io")
			{
				group := &metav1.APIGroup{}
				err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/authentication.k8s.io").Do(ctx).Into(group)
				framework.ExpectNoError(err)
				found := false
				for _, version := range group.Versions {
					if version.Version == apiVersion {
						found = true
						break
					}
				}
				if !found {
					ginkgo.Skip(fmt.Sprintf("expected SelfSubjectReview API version, got %#v", group.Versions))
				}
			}

			ginkgo.By("getting /apis/authentication.k8s.io/" + apiVersion)
			{
				resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(gv)
				framework.ExpectNoError(err)
				found := false
				for _, resource := range resources.APIResources {
					switch resource.Name {
					case "selfsubjectreviews":
						found = true
					}
				}
				if !found {
					ginkgo.Skip(fmt.Sprintf("expected selfsubjectreviews, got %#v", resources.APIResources))
				}
			}
		},
		ginkgo.Entry("authentication/v1alpha1", "v1alpha1", authenticationv1alpha1.SchemeGroupVersion.String()),
		ginkgo.Entry("authentication/v1beta1", "v1beta1", authenticationv1beta1.SchemeGroupVersion.String()),
		ginkgo.Entry("authentication/v1", "v1", authenticationv1.SchemeGroupVersion.String()),
	)

	ginkgo.It("should support SelfSubjectReview API operations", func(ctx context.Context) {
		// Check creating
		ginkgo.By("creating SSR authentication/v1alpha1")
		{
			// Use impersonate to make user attributes predictable
			config := restConfig(f)

			ssrClient := kubernetes.NewForConfigOrDie(config).AuthenticationV1alpha1().SelfSubjectReviews()
			res, err := ssrClient.Create(ctx, &authenticationv1alpha1.SelfSubjectReview{}, metav1.CreateOptions{})
			if apierrors.IsNotFound(err) {
				return // Alpha API is disabled
			}

			framework.ExpectNoError(err)
			gomega.Expect(config.Impersonate.UserName).To(gomega.Equal(res.Status.UserInfo.Username))
			gomega.Expect(config.Impersonate.UID).To(gomega.Equal(res.Status.UserInfo.UID))
			gomega.Expect(config.Impersonate.Groups).To(gomega.Equal(res.Status.UserInfo.Groups))

			extra := make(map[string][]string, len(res.Status.UserInfo.Extra))
			for k, v := range res.Status.UserInfo.Extra {
				extra[k] = v
			}

			gomega.Expect(config.Impersonate.Extra).To(gomega.Equal(extra))
		}

		ginkgo.By("creating SSR authentication/v1beta1")
		{
			config := restConfig(f)

			ssrClient := kubernetes.NewForConfigOrDie(config).AuthenticationV1beta1().SelfSubjectReviews()
			res, err := ssrClient.Create(ctx, &authenticationv1beta1.SelfSubjectReview{}, metav1.CreateOptions{})
			if apierrors.IsNotFound(err) {
				return // Beta API is disabled
			}

			framework.ExpectNoError(err)
			gomega.Expect(config.Impersonate.UserName).To(gomega.Equal(res.Status.UserInfo.Username))
			gomega.Expect(config.Impersonate.UID).To(gomega.Equal(res.Status.UserInfo.UID))
			gomega.Expect(config.Impersonate.Groups).To(gomega.Equal(res.Status.UserInfo.Groups))

			extra := make(map[string][]string, len(res.Status.UserInfo.Extra))
			for k, v := range res.Status.UserInfo.Extra {
				extra[k] = v
			}

			gomega.Expect(config.Impersonate.Extra).To(gomega.Equal(extra))
		}

		ginkgo.By("creating SSR authentication/v1")
		{
			config := restConfig(f)

			ssrClient := kubernetes.NewForConfigOrDie(config).AuthenticationV1().SelfSubjectReviews()
			res, err := ssrClient.Create(ctx, &authenticationv1.SelfSubjectReview{}, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			gomega.Expect(config.Impersonate.UserName).To(gomega.Equal(res.Status.UserInfo.Username))
			gomega.Expect(config.Impersonate.UID).To(gomega.Equal(res.Status.UserInfo.UID))
			gomega.Expect(config.Impersonate.Groups).To(gomega.Equal(res.Status.UserInfo.Groups))

			extra := make(map[string][]string, len(res.Status.UserInfo.Extra))
			for k, v := range res.Status.UserInfo.Extra {
				extra[k] = v
			}

			gomega.Expect(config.Impersonate.Extra).To(gomega.Equal(extra))
		}
	})
})

func restConfig(f *framework.Framework) *rest.Config {
	// Use impersonate to make user attributes predictable
	config := f.ClientConfig()
	config.Impersonate.UserName = "jane-doe"
	config.Impersonate.UID = "uniq-id"
	config.Impersonate.Groups = []string{"system:authenticated", "developers"}
	config.Impersonate.Extra = map[string][]string{
		"known-languages": {"python", "javascript"},
	}

	return config
}
