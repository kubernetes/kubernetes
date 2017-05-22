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

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	fedtest "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"

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
				func(federatedtypes.FederatedTypeAdapter, []*federationapi.Cluster, pkgruntime.Object) ([]util.FederatedOperation, error) {
					if testCase.operationsError {
						return nil, awfulError
					}
					return testCase.operations, nil
				},
				func([]util.FederatedOperation) error {
					if testCase.executionError {
						return awfulError
					}
					return nil
				},
				adapter,
				obj,
			)
			require.Equal(t, testCase.status, status, "Unexpected status!")
		})
	}
}

func TestClusterOperations(t *testing.T) {
	adapter := &federatedtypes.SecretAdapter{}
	obj := adapter.NewTestObject("foo")
	differingObj := adapter.Copy(obj)
	federatedtypes.SetAnnotation(adapter, differingObj, "foo", "bar")

	testCases := map[string]struct {
		clusterObject   pkgruntime.Object
		expectedErr     bool
		expectedSendErr bool
		sendToCluster   bool

		operationType util.FederatedOperationType
	}{
		"Accessor error returned": {
			expectedErr: true,
		},
		"sendToCluster error returned": {
			expectedSendErr: true,
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

			operations, err := clusterOperations(adapter, clusters, obj, key, func(string) (interface{}, bool, error) {
				if testCase.expectedErr {
					return nil, false, awfulError
				}
				return testCase.clusterObject, (testCase.clusterObject != nil), nil
			}, func(map[string]string, map[string]string) (bool, error) {
				if testCase.expectedSendErr {
					return false, awfulError
				}
				return testCase.sendToCluster, nil
			})
			if testCase.expectedErr || testCase.expectedSendErr {
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
