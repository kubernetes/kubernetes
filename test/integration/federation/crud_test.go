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

package federation

import (
	"fmt"
	"testing"

	"github.com/pborman/uuid"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federatedtypes/crudtester"
	"k8s.io/kubernetes/test/integration/federation/framework"
)

// TestFederationCRUD validates create/read/update/delete operations for federated resource types.
func TestFederationCRUD(t *testing.T) {
	fedFixture := framework.FederationFixture{DesiredClusterCount: 2}
	fedFixture.SetUp(t)
	defer fedFixture.TearDown(t)

	federatedTypes := federatedtypes.FederatedTypes()
	for kind, fedType := range federatedTypes {
		t.Run(kind, func(t *testing.T) {
			fixture, crudTester, obj, _ := initCRUDTest(t, &fedFixture, fedType.AdapterFactory, kind)
			defer fixture.TearDown(t)

			crudTester.CheckLifecycle(obj)
		})
	}

	// The following tests target a single type since the underlying logic is common across all types.
	kind := federatedtypes.SecretKind
	adapterFactory := federatedtypes.NewSecretAdapter

	// Validate deletion handling where orphanDependents is true or nil
	orphanedDependents := true
	testCases := map[string]*bool{
		"Resource should not be deleted from underlying clusters when OrphanDependents is true": &orphanedDependents,
		"Resource should not be deleted from underlying clusters when OrphanDependents is nil":  nil,
	}
	for testName, orphanDependents := range testCases {
		t.Run(testName, func(t *testing.T) {
			fixture, crudTester, obj, _ := initCRUDTest(t, &fedFixture, adapterFactory, kind)
			defer fixture.TearDown(t)

			updatedObj := crudTester.CheckCreate(obj)
			crudTester.CheckDelete(updatedObj, orphanDependents)
		})
	}

	t.Run("Resource should be propagated to a newly added cluster", func(t *testing.T) {
		fixture, crudTester, obj, _ := initCRUDTest(t, &fedFixture, adapterFactory, kind)
		defer fixture.TearDown(t)

		updatedObj := crudTester.CheckCreate(obj)
		// Start a new cluster and validate that the resource is propagated to it.
		fedFixture.StartCluster(t)
		// Check propagation to the new cluster by providing the updated set of clients
		objectExpected := true
		crudTester.CheckPropagationForClients(updatedObj, fedFixture.ClusterClients, objectExpected)
	})

	t.Run("Resource should only be propagated to the cluster with a matching selector", func(t *testing.T) {
		fixture, crudTester, obj, adapter := initCRUDTest(t, &fedFixture, adapterFactory, kind)
		defer fixture.TearDown(t)

		// Set an annotation to specify that the object is isolated to cluster 1.
		federatedtypes.SetAnnotation(adapter, obj, federationapi.FederationClusterSelectorAnnotation, `[{"key": "cluster", "operator": "==", "values": ["1"]}]`)

		updatedObj := crudTester.Create(obj)

		// Check propagation to the first cluster
		objectExpected := true
		crudTester.CheckPropagationForClients(updatedObj, fedFixture.ClusterClients[0:1], objectExpected)

		// Verify the object is not sent to the second cluster
		objectExpected = false
		crudTester.CheckPropagationForClients(updatedObj, fedFixture.ClusterClients[1:2], objectExpected)

	})
}

// initCRUDTest initializes common elements of a crud test
func initCRUDTest(t *testing.T, fedFixture *framework.FederationFixture, adapterFactory federatedtypes.AdapterFactory, kind string) (
	*framework.ControllerFixture, *crudtester.FederatedTypeCRUDTester, pkgruntime.Object, federatedtypes.FederatedTypeAdapter) {
	config := fedFixture.APIFixture.NewConfig()
	fixture := framework.NewControllerFixture(t, kind, adapterFactory, config)

	client := fedFixture.APIFixture.NewClient(fmt.Sprintf("crud-test-%s", kind))
	adapter := adapterFactory(client)

	crudTester := framework.NewFederatedTypeCRUDTester(t, adapter, fedFixture.ClusterClients)

	obj := adapter.NewTestObject(uuid.New())

	return fixture, crudTester, obj, adapter
}
