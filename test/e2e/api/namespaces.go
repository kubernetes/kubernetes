/*
Copyright 2018 The Kubernetes Authors.

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

package api

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	namespacesRootUrl  string = "/api/v1/namespaces/"
	originalGeneration int64  = 0
)

var _ = SIGDescribe("api-namespaces-list", func() {
	f := framework.NewDefaultFramework("api-namespaces-list")

	Context("starting from an any environment (might be empty or not)", func() {
		// explicitly not asking for a namespace not being created for the test
		f.SkipNamespaceCreation = true

		/*
		   Testname: api-namespaces-list-default
		   Description: Check that the namespaces returned match the expected well known default namespaces.
		*/
		framework.ConformanceIt("list with empty options should show built-in by default namespaces", func() {

			defaultNamespaces := make(map[string]bool)
			defaultNamespaces["default"] = false
			defaultNamespaces["kube-system"] = false
			defaultNamespaces["kube-public"] = false

			By("Retrieving namespaces")
			options := metav1.ListOptions{}
			namespaces, err := f.ClientSet.CoreV1().Namespaces().List(options)
			Expect(err).NotTo(HaveOccurred())

			// checking default namespaces are present in the result
			for _, namespace := range namespaces.Items {
				_, present := defaultNamespaces[namespace.ObjectMeta.Name]
				if present {
					delete(defaultNamespaces, namespace.ObjectMeta.Name)
				}
			}
			// if all were present, map should end up empty
			Expect(len(defaultNamespaces)).To(Equal(0))
		})
	})

})

var _ = SIGDescribe("api-namespaces-create", func() {
	f := framework.NewDefaultFramework("api-namespaces-create")

	Context("starting from an any environment (might be empty or not)", func() {
		// explicitly not asking for a namespace not being created for the test
		f.SkipNamespaceCreation = true

		/*
		   Testname: api-namespaces-create-empty
		   Description: Check that creating a namespace with empty parameters fails.
		*/
		framework.ConformanceIt("creating a namespace with empty parameters should fail", func() {
			namespaceCreationData := &v1.Namespace{}
			_, err := f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).To(HaveOccurred())
		})

		/*
		   Testname: api-namespaces-create-name-valid
		   Description: Check that creating a namespace with valid name parameter succeeds.
		*/
		framework.ConformanceIt("creating a namespace with only (valid) name parameter should succeed", func() {
			namespaceName :=
				strings.Join([]string{"testbasename", string(uuid.NewUUID())}, "")

			By(fmt.Sprintf(" creating namespaceName:[%v]\n", namespaceName))

			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespaceName,
				},
			}

			namespace, err := f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				err := f.ClientSet.CoreV1().Namespaces().Delete(namespace.ObjectMeta.Name, nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			v := newDefaultVerifier()
			v.verifyName = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.Name).To(Equal(namespaceName)) // this should match our given namespaceName
			}
			v.verifySelfLink = func(namespace *v1.Namespace) {
				expectedSelfLink := strings.Join([]string{namespacesRootUrl, namespaceName}, "")
				Expect(namespace.ObjectMeta.SelfLink).To(Equal(expectedSelfLink))
			}
			v.verifyAll(namespace)

		})

		/*
		   Testname: api-namespaces-create-generatename-invalid
		   Description: Check that creating a namespace with invalid generatename parameter fails.
		*/
		framework.ConformanceIt("creating a namespace with invalid generateName (containing uppercase) should fail", func() {
			generateName :=
				strings.Join([]string{"thisHasUppercaseCharacters", string(uuid.NewUUID())}, "")

			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: generateName,
					Name:         "",
				},
			}

			_, err := f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).To(HaveOccurred())
		})

		/*
		   Testname: api-namespaces-create-generatename-valid
		   Description: Check that creating a namespace with valid generatename parameter succeeds.
		*/
		framework.ConformanceIt("creating a namespace with valid generateName (all lowercase)", func() {
			generateName :=
				strings.Join([]string{"lowercasecharacters", string(uuid.NewUUID())}, "")

			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: generateName,
					Name:         "",
				},
			}

			namespace, err := f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				err := f.ClientSet.CoreV1().Namespaces().Delete(namespace.ObjectMeta.Name, nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			v := newDefaultVerifier()
			v.verifyGenerateName = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.GenerateName).To(Equal(generateName))
			}
			v.verifyName = func(namespace *v1.Namespace) {
				Expect(strings.HasPrefix(namespace.ObjectMeta.Name, generateName)).To(BeTrue())
			}
			v.verifySelfLink = func(namespace *v1.Namespace) {
				expectedSelfLinkBase := strings.Join([]string{namespacesRootUrl, generateName}, "")
				Expect(strings.HasPrefix(namespace.ObjectMeta.SelfLink, expectedSelfLinkBase)).To(BeTrue())
			}
			v.verifyAll(namespace)
		})

		/*
		   Testname: api-namespaces-create-generatename-valid-labels
		   Description: Check that creating a namespace with valid generatename and labels parameters succeeds.
		*/
		framework.ConformanceIt("creating a namespace with valid generateName (all lowercase) and labels set", func() {
			generateName :=
				strings.Join([]string{"lowercasecharacters", string(uuid.NewUUID())}, "")

			labels := map[string]string{}
			labels["e2e-run"] = string(uuid.NewUUID())

			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: generateName,
					Name:         "",
					Labels:       labels,
				},
			}

			namespace, err := f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				err := f.ClientSet.CoreV1().Namespaces().Delete(namespace.ObjectMeta.Name, nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			v := newDefaultVerifier()
			v.verifyGenerateName = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.GenerateName).To(Equal(generateName))
			}
			v.verifyName = func(namespace *v1.Namespace) {
				Expect(strings.HasPrefix(namespace.ObjectMeta.Name, generateName)).To(BeTrue())
			}
			v.verifySelfLink = func(namespace *v1.Namespace) {
				expectedSelfLinkBase := strings.Join([]string{namespacesRootUrl, generateName}, "")
				Expect(strings.HasPrefix(namespace.ObjectMeta.SelfLink, expectedSelfLinkBase)).To(BeTrue())
			}
			v.verifyLabels = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.Labels).To(Equal(labels))
			}
			v.verifyAll(namespace)
		})
	})
})

// VerifyFunc is a type meant to hold functions that operate on namespaces

type VerifyFunc func(namespace *v1.Namespace)

// NamespaceVerifier is an object meant to hold functions that operate on namespaces that will be later called to verify expectations.

type NamespaceVerifier struct {
	verifyKind       VerifyFunc
	verifyAPIVersion VerifyFunc

	verifyName         VerifyFunc
	verifyGenerateName VerifyFunc

	verifySelfLink VerifyFunc
	verifyUID      VerifyFunc

	verifyGeneration                 VerifyFunc
	verifyCreationTimestamp          VerifyFunc
	verifyDeletionTimestamp          VerifyFunc
	verifyDeletionGracePeriodSeconds VerifyFunc

	verifyLabels      VerifyFunc
	verifyAnnotations VerifyFunc

	verifyInitializers VerifyFunc
	verifyFinalizers   VerifyFunc
}

// method invokeable from a namespaceVerifier object, that verifies all conditions by calling the functions
func (verifier NamespaceVerifier) verifyAll(namespace *v1.Namespace) {
	// Metadata
	verifier.verifyKind(namespace)
	verifier.verifyAPIVersion(namespace)

	// ObjectMeta
	verifier.verifyName(namespace)
	verifier.verifyGenerateName(namespace)

	verifier.verifySelfLink(namespace)
	verifier.verifyUID(namespace)
	verifier.verifyGeneration(namespace)

	verifier.verifyCreationTimestamp(namespace)
	verifier.verifyDeletionTimestamp(namespace)
	verifier.verifyDeletionGracePeriodSeconds(namespace)

	verifier.verifyLabels(namespace)
	verifier.verifyAnnotations(namespace)

	verifier.verifyInitializers(namespace)
	verifier.verifyFinalizers(namespace)
}

// creation of a default verifier that helps us defining basic expectations throughout different scenarios.
// Allows override of specific areas that we want to verify against
func newDefaultVerifier() NamespaceVerifier {
	v := NamespaceVerifier{

		verifyKind: func(namespace *v1.Namespace) {
			Expect(namespace.TypeMeta.Kind).To(Equal(""))
		},
		verifyAPIVersion: func(namespace *v1.Namespace) {
			Expect(namespace.TypeMeta.APIVersion).To(Equal(""))
		},

		verifyName: func(namespace *v1.Namespace) {
			Expect(namespace.ObjectMeta.Name).To(Equal(""))
		},
		verifyGenerateName: func(namespace *v1.Namespace) {
			Expect(namespace.ObjectMeta.GenerateName).To(Equal(""))
		},

		verifySelfLink: func(namespace *v1.Namespace) {

		},
		verifyUID: func(namespace *v1.Namespace) {
			// TODO / Question
			// How to work against UID? wanted to check is not empty
			//Expect(len(strings.TrimSpace(namespace.ObjectMeta.UID)) != 0).To(BeTrue())
		},
		verifyGeneration: func(namespace *v1.Namespace) {
			Expect(namespace.ObjectMeta.Generation).To(Equal(originalGeneration)) // this is brand new
		},

		verifyCreationTimestamp: func(namespace *v1.Namespace) {
			Expect(namespace.ObjectMeta.CreationTimestamp).NotTo(BeNil())
		},
		verifyDeletionTimestamp: func(namespace *v1.Namespace) {
			// TODO / Question
			// resolve Time package import
			//var defaultDeletionTimestamp *Time = nil
			//Expect(namespace.ObjectMeta.DeletionTimestamp).To(Equal(defaultDeletionTimestamp))
		},
		verifyDeletionGracePeriodSeconds: func(namespace *v1.Namespace) {
			var defaultDeletionGracePeriodSeconds *int64 = nil
			Expect(namespace.ObjectMeta.DeletionGracePeriodSeconds).To(Equal(defaultDeletionGracePeriodSeconds))
		},

		verifyLabels: func(namespace *v1.Namespace) {
			Expect(len(namespace.ObjectMeta.Labels)).To(Equal(0))
		},
		verifyAnnotations: func(namespace *v1.Namespace) {
			Expect(len(namespace.ObjectMeta.Annotations)).To(Equal(0))
		},

		verifyInitializers: func(namespace *v1.Namespace) {
			// TODO / Question
			// resolve Initializers package import
			//var defaultInitializers *Initializers = nil
			//Expect(namespace.ObjectMeta.Initializers).To(Equal(defaultInitializers))
		},
		verifyFinalizers: func(namespace *v1.Namespace) {
			Expect(len(namespace.ObjectMeta.Finalizers)).To(Equal(0))
		},
	}
	return v
}
