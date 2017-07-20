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

package sync

import (
	"errors"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	fedtest "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"

	"github.com/stretchr/testify/require"
)

var awfulError error = errors.New("Something bad happened")

func TestSyncToClusters(t *testing.T) {
	adapter := &federatedtypes.SecretAdapter{}
	obj := adapter.NewTestObject("foo")

	testCases := map[string]struct {
		clusterError    bool
		operationsError bool
		executionError  bool
		operations      []util.FederatedOperation
		status          reconciliationStatus
	}{
		"Error listing clusters redelivers with cluster delay": {
			clusterError: true,
			status:       statusNotSynced,
		},
		"Error retrieving cluster operations redelivers": {
			operationsError: true,
			status:          statusError,
		},
		"No operations returns ok": {
			status: statusAllOK,
		},
		"Execution error redelivers": {
			executionError: true,
			operations:     []util.FederatedOperation{{}},
			status:         statusError,
		},
		"Successful update indicates recheck": {
			operations: []util.FederatedOperation{{}},
			status:     statusNeedsRecheck,
		},
	}

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			status := syncToClusters(
				func() ([]*federationapi.Cluster, error) {
					if testCase.clusterError {
						return nil, awfulError
					}
					return nil, nil
				},
				func(federatedtypes.FederatedTypeAdapter, []*federationapi.Cluster, []*federationapi.Cluster, pkgruntime.Object, interface{}) ([]util.FederatedOperation, error) {
					if testCase.operationsError {
						return nil, awfulError
					}
					return testCase.operations, nil
				},
				func(objMeta *metav1.ObjectMeta, selector func(map[string]string, map[string]string) (bool, error), clusters []*federationapi.Cluster) ([]*federationapi.Cluster, []*federationapi.Cluster, error) {
					return clusters, []*federationapi.Cluster{}, nil
				},
				func([]util.FederatedOperation) error {
					if testCase.executionError {
						return awfulError
					}
					return nil
				},
				adapter,
				nil,
				obj,
			)
			require.Equal(t, testCase.status, status, "Unexpected status!")
		})
	}
}

func TestSelectedClusters(t *testing.T) {
	clusterOne := fedtest.NewCluster("cluster1", apiv1.ConditionTrue)
	clusterOne.Labels = map[string]string{"name": "cluster1"}
	clusterTwo := fedtest.NewCluster("cluster2", apiv1.ConditionTrue)
	clusterTwo.Labels = map[string]string{"name": "cluster2"}

	clusters := []*federationapi.Cluster{clusterOne, clusterTwo}
	testCases := map[string]struct {
		expectedSelectorError      bool
		clusterOneSelected         bool
		clusterTwoSelected         bool
		expectedSelectedClusters   []*federationapi.Cluster
		expectedUnselectedClusters []*federationapi.Cluster
	}{
		"Selector returned error": {
			expectedSelectorError: true,
		},
		"All clusters selected": {
			clusterOneSelected:         true,
			clusterTwoSelected:         true,
			expectedSelectedClusters:   clusters,
			expectedUnselectedClusters: []*federationapi.Cluster{},
		},
		"One cluster selected": {
			clusterOneSelected:         true,
			expectedSelectedClusters:   []*federationapi.Cluster{clusterOne},
			expectedUnselectedClusters: []*federationapi.Cluster{clusterTwo},
		},
		"No clusters selected": {
			expectedSelectedClusters:   []*federationapi.Cluster{},
			expectedUnselectedClusters: clusters,
		},
	}

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			selectedClusters, unselectedClusters, err := selectedClusters(&metav1.ObjectMeta{}, func(labels map[string]string, annotations map[string]string) (bool, error) {
				if testCase.expectedSelectorError {
					return false, awfulError
				}
				if labels["name"] == "cluster1" {
					return testCase.clusterOneSelected, nil
				}
				if labels["name"] == "cluster2" {
					return testCase.clusterTwoSelected, nil
				}
				t.Errorf("Unexpected cluster")
				return false, nil
			}, clusters)

			if testCase.expectedSelectorError {
				require.Error(t, err, "An error was expected")
			} else {
				require.NoError(t, err, "An error was not expected")
			}
			require.Equal(t, testCase.expectedSelectedClusters, selectedClusters, "Expected the correct clusters to be selected.")
			require.Equal(t, testCase.expectedUnselectedClusters, unselectedClusters, "Expected the correct clusters to be unselected.")
		})
	}
}

func TestClusterOperations(t *testing.T) {
	adapter := &federatedtypes.SecretAdapter{}
	obj := adapter.NewTestObject("foo")
	differingObj := adapter.Copy(obj)
	federatedtypes.SetAnnotation(adapter, differingObj, "foo", "bar")

	testCases := map[string]struct {
		clusterObject pkgruntime.Object
		expectedErr   bool
		sendToCluster bool

		operationType util.FederatedOperationType
	}{
		"Accessor error returned": {
			expectedErr: true,
		},
		"Missing cluster object should result in add operation": {
			operationType: util.OperationTypeAdd,
			sendToCluster: true,
		},
		"Differing cluster object should result in update operation": {
			clusterObject: differingObj,
			operationType: util.OperationTypeUpdate,
			sendToCluster: true,
		},
		"Matching object and not matching ClusterSelector should result in delete operation": {
			clusterObject: obj,
			operationType: util.OperationTypeDelete,
			sendToCluster: false,
		},
		"Matching cluster object should not result in an operation": {
			clusterObject: obj,
			sendToCluster: true,
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			clusters := []*federationapi.Cluster{fedtest.NewCluster("cluster1", apiv1.ConditionTrue)}
			key := federatedtypes.ObjectKey(adapter, obj)

			var selectedClusters, unselectedClusters []*federationapi.Cluster
			if testCase.sendToCluster {
				selectedClusters = clusters
				unselectedClusters = []*federationapi.Cluster{}
			} else {
				selectedClusters = []*federationapi.Cluster{}
				unselectedClusters = clusters
			}
			// TODO: Tests for ScheduleObject on type adapter
			operations, err := clusterOperations(adapter, selectedClusters, unselectedClusters, obj, key, nil, func(string) (interface{}, bool, error) {
				if testCase.expectedErr {
					return nil, false, awfulError
				}
				return testCase.clusterObject, (testCase.clusterObject != nil), nil
			})
			if testCase.expectedErr {
				require.Error(t, err, "An error was expected")
			} else {
				require.NoError(t, err, "An error was not expected")
			}
			if len(testCase.operationType) == 0 {
				require.True(t, len(operations) == 0, "An operation was not expected")
			} else {
				require.True(t, len(operations) == 1, "A single operation was expected")
				require.Equal(t, testCase.operationType, operations[0].Type, "Unexpected operation returned")
			}
		})
	}
}
