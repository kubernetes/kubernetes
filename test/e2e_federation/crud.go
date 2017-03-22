/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_federation

import (
	. "github.com/onsi/ginkgo"

	"k8s.io/kubernetes/federation/pkg/typeadapters"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
)

var _ = framework.KubeDescribe("Federated types [Feature:Federation][Experimental] ", func() {
	var clusterClients []kubeclientset.Interface

	f := fedframework.NewDefaultFederatedFramework("federated-type")

	resourceTypes := map[string]typeadapters.FederatedTypeAdapter{
		"Secrets": typeadapters.NewSecretAdapter(f.FederationClientset),
	}
	for pluralName, adapter := range resourceTypes {
		Describe(pluralName, func() {
			It("should be created, read, updated and deleted successfully", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

				// Only need to load clients once
				if clusterClients == nil {
					clusterClients = f.GetClusterClients()
				}

				crudtester := fedframework.NewFederatedTypeCRUDTester(adapter, clusterClients)
				obj := adapter.NewTestObject(f.FederationNamespace.Name)
				crudtester.CheckLifecycle(obj)
			})
		})
	}
})
