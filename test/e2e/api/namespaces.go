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
	"time"

	v1 "k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	namespacesRootURL  string = "/api/v1/namespaces/"
	originalGeneration int64  = 0
)

var _ = SIGDescribe("api-namespaces", func() {
	f := framework.NewDefaultFramework("api-namespaces")

	Context("starting from an any environment (might be empty or not)", func() {
		// explicitly not asking for a namespace not being created for the test
		f.SkipNamespaceCreation = true

		/*
		   Testname: api-namespaces-scenario-get-post-delete-get
		   Description:
		     1) check that the namespaceName does not already exist: get by name should not find it
		     2) post it
		     3) check that the newly created namespace is brought when get by name is invoked.
		     4) check that the newly created namespace appears on the list.
		     5) delete the namespace, waiting until 
		     6) check that the newly created namespace is NOT brought when get by name is invoked.
		     7) check that the newly created namespace does NOT appear on the list.
		*/
		framework.ConformanceIt("scenario: get: not-found > post > list : found > get by name: found-and-matches >  delete >  list: not-found > get: not-found", func() {
			namespaceName :=
				strings.Join([]string{"testbasename", string(uuid.NewUUID())}, "")

			By(fmt.Sprintf("namespaceName:[%v] should not be found\n", namespaceName))
			getOptions := metav1.GetOptions{}
			namespace, err := f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)
			Expect(err).To(HaveOccurred())

			By(fmt.Sprintf("creating namespaceName:[%v]\n", namespaceName))
			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespaceName,
				},
			}
			namespace, err = f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).NotTo(HaveOccurred())

			v := newDefaultVerifier()
			v.verifyName = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.Name).To(Equal(namespaceName)) // this should match our given namespaceName
			}
			v.verifySelfLink = func(namespace *v1.Namespace) {
				expectedSelfLink := strings.Join([]string{namespacesRootURL, namespaceName}, "")
				Expect(namespace.ObjectMeta.SelfLink).To(Equal(expectedSelfLink))
			}

			By(fmt.Sprintf("running verifications\n"))
			v.verifyAll(namespace)

			By(fmt.Sprintf("listing all namespaces, should return the just created namespace\n"))
			listOptions := metav1.ListOptions{}
			var namespaces *v1.NamespaceList = nil
			namespaces, err = f.ClientSet.CoreV1().Namespaces().List(listOptions)
			Expect(err).NotTo(HaveOccurred())

			found := false
			for _, namespaceItem := range namespaces.Items {
				found = (namespaceItem.ObjectMeta.Name == namespaceName)
				if found {
					break
				}
			}
			Expect(found).To(BeTrue())

			By(fmt.Sprintf("looking for that specific namespace again\n"))
			namespace, err = f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("running verifications on returned by name namespace\n"))
			v.verifyAll(namespace)

			By(fmt.Sprintf("deleting namespace\n"))
			err = f.ClientSet.CoreV1().Namespaces().Delete(namespace.ObjectMeta.Name, nil)
			Expect(err).NotTo(HaveOccurred())

			// looking for that specific namespace again, should return not found
			maxSeconds := 10
			framework.ExpectNoError(wait.Poll(time.Second, time.Duration(maxSeconds)*time.Second,
				func() (bool, error) {
					namespace, err = f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)

					if err != nil {
						By(fmt.Sprintf("namespace was probably already deleted\n"))
						return true, nil
					}

					if namespace != nil && namespace.Status.Phase == "Terminating" {
						By(fmt.Sprintf("namespace in terminating state,  wait\n"))
						return false, nil
					}
					return true, nil
				}))

			By(fmt.Sprintf("listing should not return the deleted namespace\n"))
			namespaces, err = f.ClientSet.CoreV1().Namespaces().List(listOptions)
			Expect(err).NotTo(HaveOccurred())

			found = false
			for _, namespaceItem := range namespaces.Items {
				found = (namespaceItem.ObjectMeta.Name == namespaceName)
				if found {
					break
				}
			}
			Expect(found).To(BeFalse())
		})

		/*
		   Testname: api-namespaces-scenario-get-post-patch-get
		   Description:
		     1) check that the namespaceName does not already exist: get by name should not find it
		     2) post it
		     3) check that the newly created namespace is brought when get by name is invoked.
		     4) patch the namespace using labels
		     5) check that the newly created namespace is brought when get by name is invoked.
		     6) delete the namespace.
		*/
		framework.ConformanceIt("scenario: get:not found > post >  get by name: found-and-matches >  patch  > get: found > delete", func() {
			namespaceName :=
				strings.Join([]string{"testbasename", string(uuid.NewUUID())}, "")

			By(fmt.Sprintf("namespaceName:[%v] should not be found\n", namespaceName))
			getOptions := metav1.GetOptions{}
			namespace, err := f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)
			Expect(err).To(HaveOccurred())

			By(fmt.Sprintf(" creating namespaceName:[%v]\n", namespaceName))
			namespaceCreationData := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: namespaceName,
				},
			}
			namespace, err = f.ClientSet.CoreV1().Namespaces().Create(namespaceCreationData)
			Expect(err).NotTo(HaveOccurred())

			v := newDefaultVerifier()
			v.verifyName = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.Name).To(Equal(namespaceName)) // this should match our given namespaceName
			}
			v.verifySelfLink = func(namespace *v1.Namespace) {
				expectedSelfLink := strings.Join([]string{namespacesRootURL, namespaceName}, "")
				Expect(namespace.ObjectMeta.SelfLink).To(Equal(expectedSelfLink))
			}
			By(fmt.Sprintf("running verifications\n"))
			v.verifyAll(namespace)

			By(fmt.Sprintf("looking for that specific namespace again\n"))
			namespace, err = f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("running verifications on returned by name namespace\n"))
			v.verifyAll(namespace)

			By(fmt.Sprintf("patching namespace with labels\n"))
			labelName := "e2e-run"
			labelValue := string(uuid.NewUUID())

			labels := map[string]string{}
			labels[labelName] = labelValue

			patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, labelName, labelValue)

			namespace, err = f.ClientSet.CoreV1().Namespaces().Patch(namespaceName, types.StrategicMergePatchType, []byte(patch))
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("running verifications on returned by name namespace, labels should be there\n"))
			v.verifyLabels = func(namespace *v1.Namespace) {
				Expect(namespace.ObjectMeta.Labels).To(Equal(labels))
			}
			v.verifyAll(namespace)

			By(fmt.Sprintf("looking for that specific namespace again\n"))
			namespace, err = f.ClientSet.CoreV1().Namespaces().Get(namespaceName, getOptions)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("deleting namespace\n"))
			err = f.ClientSet.CoreV1().Namespaces().Delete(namespace.ObjectMeta.Name, nil)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

// VerifyFunc is a type meant to hold functions that operate on namespaces
type VerifyFunc func(namespace *v1.Namespace)

// NamespaceVerifier is an object meant to hold functions that operate on namespaces that will be
//later called to verify expectations.
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
			By(fmt.Sprintf(">>>namespace.ObjectMeta.Generation:[%v]\n", namespace.ObjectMeta.Generation))
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
			var defaultDeletionGracePeriodSeconds *int64
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
