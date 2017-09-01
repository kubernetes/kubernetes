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
	"fmt"

	. "github.com/onsi/ginkgo"

	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
)

var _ = framework.KubeDescribe("Federated types [Feature:Federation][Experimental] ", func() {
	var clusterClients []kubeclientset.Interface

	f := fedframework.NewDefaultFederatedFramework("federated-types")

	fedTypes := federatedtypes.FederatedTypes()
	for name := range fedTypes {
		fedType := fedTypes[name]
		Describe(fmt.Sprintf("Federated %q resources", name), func() {
			It("should be created, read, updated and deleted successfully", func() {
				fedframework.SkipUnlessFederated(f.ClientSet)

				// Load clients only if not skipping to avoid doing
				// unnecessary work.  Assume clients can be shared
				// across tests.
				if clusterClients == nil {
					clusterClients = f.GetClusterClients()
				}
				adapter := fedType.AdapterFactory(f.FederationClientset, f.FederationConfig, nil)
				crudTester := fedframework.NewFederatedTypeCRUDTester(adapter, clusterClients)
				obj := adapter.NewTestObject(f.FederationNamespace.Name)
				crudTester.CheckLifecycle(obj)
			})
		})
	}
})
