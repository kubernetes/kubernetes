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

package node

import (
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Nodes [Disruptive]", func() {
	var cs clientset.Interface
	f := framework.NewDefaultFramework("nodes")

	BeforeEach(func() {
		cs = f.ClientSet
	})

	/*
		Release : v1.13
		Testname: Nodes
		Description: Delete the fist node from the nodeList, and then Create the same node.
	*/
	It("should be deletable and recreatable", func() {
		By("Getting nodes")
		nodeList, err := cs.CoreV1().Nodes().List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		firstNode := nodeList.Items[0]

		By("Deleting the first node")
		err = cs.CoreV1().Nodes().Delete(firstNode.GetName(), &metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		By("Check if the first node is deleted")
		_, err = cs.CoreV1().Nodes().Get(firstNode.GetName(), metav1.GetOptions{})
		Expect(errors.IsNotFound(err)).To(BeTrue())

		By("Creating the node")
		firstNode.ObjectMeta.ResourceVersion = "" // resourceVersion should not be set on objects to be created
		_, err = cs.CoreV1().Nodes().Create(&firstNode)
		framework.ExpectNoError(err)
		By("Check if a recreated node is exist")
		_, err = cs.CoreV1().Nodes().Get(firstNode.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)
	})
})
