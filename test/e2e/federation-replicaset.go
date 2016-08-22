/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

const (
	FederationReplicaSetName = "federation-replicaset"
)

// Create/delete replicaset api objects
var _ = framework.KubeDescribe("Federation replicasets [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-replicaset")

	Describe("ReplicaSet objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)

			// Delete registered replicasets.
			nsName := f.FederationNamespace.Name
			replicasetList, err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, replicaset := range replicasetList.Items {
				err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).Delete(replicaset.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			nsName := f.FederationNamespace.Name
			replicaset := createReplicaSetOrFail(f.FederationClientset_1_4, nsName)
			By(fmt.Sprintf("Creation of replicaset %q in namespace %q succeeded.  Deleting replicaset.", replicaset.Name, nsName))
			// Cleanup
			err := f.FederationClientset_1_4.Extensions().ReplicaSets(nsName).Delete(replicaset.Name, &api.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting replicaset %q in namespace %q", replicaset.Name, replicaset.Namespace)
			By(fmt.Sprintf("Deletion of replicaset %q in namespace %q succeeded.", replicaset.Name, nsName))
		})

	})
})

func createReplicaSetOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1beta1.ReplicaSet {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createReplicaSetOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))

	replicas := int32(5)
	replicaset := &v1beta1.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name:      FederationReplicaSetName,
			Namespace: namespace,
		},
		Spec: v1beta1.ReplicaSetSpec{
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{"name": "myrs"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "nginx",
						},
					},
				},
			},
		},
	}

	_, err := clientset.Extensions().ReplicaSets(namespace).Create(replicaset)
	framework.ExpectNoError(err, "Creating replicaset %q in namespace %q", replicaset.Name, namespace)
	By(fmt.Sprintf("Successfully created federation replicaset %q in namespace %q", FederationReplicaSetName, namespace))
	return replicaset
}
