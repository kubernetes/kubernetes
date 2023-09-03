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

	authorizationv1 "k8s.io/api/authorization/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("SubjectReview", func() {
	f := framework.NewDefaultFramework("subjectreview")

	/*
		Release: v1.27
		Testname: SubjectReview, API Operations
		Description: A ServiceAccount is created which MUST succeed.
		A clientset is created to impersonate the ServiceAccount.
		A SubjectAccessReview is created for the ServiceAccount which
		MUST succeed. The allowed status for the SubjectAccessReview
		MUST match the expected allowed for the impersonated client
		call. A LocalSubjectAccessReviews is created for the ServiceAccount
		which MUST succeed. The allowed status for the LocalSubjectAccessReview
		MUST match the expected allowed for the impersonated client call.
	*/
	framework.ConformanceIt("should support SubjectReview API operations", func() {

		AuthClient := f.ClientSet.AuthorizationV1()
		ns := f.Namespace.Name
		saName := "e2e"

		ginkgo.By(fmt.Sprintf("Creating a Serviceaccount %q in namespace %q", saName, ns))

		sa, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Create(context.TODO(), &v1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{
				Name: saName,
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Unable to create serviceaccount %q", saName)

		saUsername := serviceaccount.MakeUsername(sa.Namespace, sa.Name)
		framework.Logf("saUsername: %q", saUsername)
		saGroups := []string{"system:authenticated"}
		saGroups = append(saGroups, serviceaccount.MakeGroupNames(sa.Namespace)...)
		framework.Logf("saGroups: %#v", saGroups)
		saUID := string(sa.UID)
		framework.Logf("saUID: %q", saUID)

		ginkgo.By(fmt.Sprintf("Creating clientset to impersonate %q", saUsername))
		config := rest.CopyConfig(f.ClientConfig())
		config.Impersonate = rest.ImpersonationConfig{
			UserName: saUsername,
			UID:      saUID,
			Groups:   saGroups,
		}

		impersonatedClientSet, err := kubernetes.NewForConfig(config)
		framework.ExpectNoError(err, "Could not load config, %v", err)

		ginkgo.By(fmt.Sprintf("Creating SubjectAccessReview for %q", saUsername))

		sar := &authorizationv1.SubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{
					Verb:      "list",
					Resource:  "configmaps",
					Namespace: ns,
					Version:   "v1",
				},
				User:   saUsername,
				Groups: saGroups,
				UID:    saUID,
			},
		}

		sarResponse, err := AuthClient.SubjectAccessReviews().Create(context.TODO(), sar, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Unable to create a SubjectAccessReview, %#v", err)
		framework.Logf("sarResponse Status: %#v", sarResponse.Status)
		expectedAllowed := sarResponse.Status.Allowed

		ginkgo.By(fmt.Sprintf("Verifying as %q api 'list' configmaps in %q namespace", saUsername, ns))
		_, requestErr := impersonatedClientSet.CoreV1().ConfigMaps(ns).List(context.TODO(), metav1.ListOptions{})

		actuallyAllowed := false
		switch {
		case apierrors.IsForbidden(requestErr):
			actuallyAllowed = false
		case requestErr != nil:
			framework.Fail("Unexpected error")
		default:
			actuallyAllowed = true
		}

		if actuallyAllowed != expectedAllowed {
			framework.Fail(fmt.Sprintf("Could not verify SubjectAccessReview for %q in namespace %q: SubjectAccessReview allowed: %v, request allowed: %v, request error: %v", saUsername, ns, expectedAllowed, actuallyAllowed, requestErr))
		}
		framework.Logf("SubjectAccessReview has been verified")

		ginkgo.By(fmt.Sprintf("Creating a LocalSubjectAccessReview for %q", saUsername))

		lsar := &authorizationv1.LocalSubjectAccessReview{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
			},
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{
					Verb:      "list",
					Resource:  "configmaps",
					Namespace: ns,
					Version:   "v1",
				},
				User:   saName,
				Groups: saGroups,
				UID:    saUID,
			},
		}

		lsarResponse, err := AuthClient.LocalSubjectAccessReviews(ns).Create(context.TODO(), lsar, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Unable to create a LocalSubjectAccessReview, %#v", err)
		framework.Logf("lsarResponse Status: %#v", lsarResponse.Status)
		expectedAllowed = lsarResponse.Status.Allowed

		if actuallyAllowed != expectedAllowed {
			framework.Fail(fmt.Sprintf("Could not verify LocalSubjectAccessReview for %q in namespace %q: LocalSubjectAccessReview allowed: %v, request allowed: %v, request error: %v", saUsername, ns, expectedAllowed, actuallyAllowed, requestErr))
		}
		framework.Logf("LocalSubjectAccessReview has been verified")
	})
})
