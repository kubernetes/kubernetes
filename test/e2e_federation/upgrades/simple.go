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

package upgrades

import (
	"fmt"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	crudtester "k8s.io/kubernetes/federation/pkg/federatedtypes/crudtester"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
)

// SimpleUpgradeTest validates that a federated resource remains
// propagated before and after a control plane upgrade
type SimpleUpgradeTest struct {
	kind           string
	adapterFactory federatedtypes.AdapterFactory
	crudTester     *crudtester.FederatedTypeCRUDTester
	obj            pkgruntime.Object
}

// Setup creates a resource and validates its propagation to member clusters
func (ut *SimpleUpgradeTest) Setup(f *fedframework.Framework) {
	adapter := ut.adapterFactory(f.FederationClientset)
	clients := f.GetClusterClients()
	ut.crudTester = fedframework.NewFederatedTypeCRUDTester(adapter, clients)

	By(fmt.Sprintf("Creating a resource of kind %q and validating propagation to member clusters", ut.kind))
	obj := adapter.NewTestObject(f.Namespace.Name)
	ut.obj = ut.crudTester.CheckCreate(obj)
}

// Test validates that a resource remains propagated post-upgrade
func (ut *SimpleUpgradeTest) Test(f *fedframework.Framework, done <-chan struct{}, upgrade FederationUpgradeType) {
	<-done
	By(fmt.Sprintf("Validating that a resource of kind %q remains propagated to member clusters after upgrade", ut.kind))
	ut.crudTester.CheckPropagation(ut.obj)
}

// Teardown cleans up remaining resources
func (ut *SimpleUpgradeTest) Teardown(f *fedframework.Framework) {
	// Rely on the namespace deletion to clean up everything
}

// SimpleUpgradeTests collects simple upgrade tests for registered federated types
func SimpleUpgradeTests() []Test {
	tests := []Test{}
	for kind, fedType := range federatedtypes.FederatedTypes() {
		tests = append(tests, &SimpleUpgradeTest{
			kind:           kind,
			adapterFactory: fedType.AdapterFactory,
		})
	}
	return tests
}
