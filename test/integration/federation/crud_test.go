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

	"k8s.io/kubernetes/federation/pkg/federatedtypes"
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
			config := fedFixture.APIFixture.NewConfig()
			fixture := framework.NewControllerFixture(t, kind, fedType.AdapterFactory, config)
			defer fixture.TearDown(t)

			client := fedFixture.APIFixture.NewClient(fmt.Sprintf("crud-test-%s", kind))
			adapter := fedType.AdapterFactory(client)

			crudtester := framework.NewFederatedTypeCRUDTester(t, adapter, fedFixture.ClusterClients)
			obj := adapter.NewTestObject(uuid.New())
			crudtester.CheckLifecycle(obj)
		})
	}

	// Validate deletion handling where orphanDependents is true or nil for a single resource type since the
	// underlying logic is common across all types.
	orphanedDependents := true
	testCases := map[string]*bool{
		"Resources should not be deleted from underlying clusters when OrphanDependents is true": &orphanedDependents,
		"Resources should not be deleted from underlying clusters when OrphanDependents is nil":  nil,
	}
	kind := federatedtypes.SecretKind
	adapterFactory := federatedtypes.NewSecretAdapter
	for testName, orphanDependents := range testCases {
		t.Run(testName, func(t *testing.T) {
			config := fedFixture.APIFixture.NewConfig()
			fixture := framework.NewControllerFixture(t, kind, adapterFactory, config)
			defer fixture.TearDown(t)

			client := fedFixture.APIFixture.NewClient(fmt.Sprintf("deletion-test-%s", kind))
			adapter := adapterFactory(client)

			crudtester := framework.NewFederatedTypeCRUDTester(t, adapter, fedFixture.ClusterClients)
			obj := adapter.NewTestObject(uuid.New())
			updatedObj := crudtester.CheckCreate(obj)
			crudtester.CheckDelete(updatedObj, orphanDependents)
		})
	}
}
