/*
Copyright 2020 The Kubernetes Authors.

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

package state

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
)

const testingCheckpoint = "dramanager_checkpoint_test"

// TODO (https://github.com/kubernetes/kubernetes/issues/123552): reconsider what data gets stored in checkpoints and whether that is really necessary.

func TestCheckpointGetOrCreate(t *testing.T) {
	testCases := []struct {
		description                string
		checkpointContent          string
		expectedError              string
		expectedClaimInfoStateList ClaimInfoStateList
	}{
		{
			description:                "new-checkpoint",
			expectedClaimInfoStateList: nil,
		},
		{
			description:       "single-claim-info-state",
			checkpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}}]}","Checksum":1656016162}`,
			expectedClaimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
		},
		{
			description:       "claim-info-state-with-multiple-devices",
			checkpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]},{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-2\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}}]}","Checksum":3369508096}`,
			expectedClaimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-2",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
		},
		{
			description:       "two-claim-info-states",
			checkpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example-1\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}},{\"ClaimUID\":\"4cf8db2d-06c0-7d70-1a51-e59b25b2c16c\",\"ClaimName\":\"example-2\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-2\",\"RequestNames\":null,\"CDIDeviceIDs\":null}]}}}]}","Checksum":1582256999}`,
			expectedClaimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example-1",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:   "worker-1",
									DeviceName: "dev-2",
								},
							},
						},
					},
					ClaimUID:  "4cf8db2d-06c0-7d70-1a51-e59b25b2c16c",
					ClaimName: "example-2",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
		},
		{
			description:       "incorrect-checksum",
			checkpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"Entries\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example-1\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}},{\"ClaimUID\":\"4cf8db2d-06c0-7d70-1a51-e59b25b2c16c\",\"ClaimName\":\"example-2\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-2\",\"RequestNames\":null,\"CDIDeviceIDs\":null}]}}}]}","Checksum":2930258365}`,
			expectedError:     "checkpoint is corrupted",
		},
		{
			description:       "invalid-JSON",
			checkpointContent: `{`,
			expectedError:     "unexpected end of JSON input",
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "dramanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(testingDir); err != nil {
			t.Fatal(err)
		}
	}()

	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

			// prepare checkpoint for testing
			if strings.TrimSpace(tc.checkpointContent) != "" {
				mock := &testutil.MockCheckpoint{Content: tc.checkpointContent}
				require.NoError(t, cpm.CreateCheckpoint(testingCheckpoint, mock), "could not create testing checkpoint")
			}

			checkpointer, err := NewCheckpointer(testingDir, testingCheckpoint)
			require.NoError(t, err, "could not create testing checkpointer")

			checkpoint, err := checkpointer.GetOrCreate()
			if strings.TrimSpace(tc.expectedError) != "" {
				assert.ErrorContains(t, err, tc.expectedError)
			} else {
				require.NoError(t, err, "unexpected error")
				stateList, err := checkpoint.GetClaimInfoStateList()
				require.NoError(t, err, "could not get data entries from checkpoint")
				require.NoError(t, err)
				assert.Equal(t, tc.expectedClaimInfoStateList, stateList)
			}
		})
	}
}

func TestCheckpointStateStore(t *testing.T) {
	testCases := []struct {
		description               string
		claimInfoStateList        ClaimInfoStateList
		expectedCheckpointContent string
	}{
		{
			description: "single-claim-info-state",
			claimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
			expectedCheckpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}}]}","Checksum":1656016162}`,
		},
		{
			description: "claim-info-state-with-multiple-devices",
			claimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-2",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
			expectedCheckpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]},{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-2\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}}]}","Checksum":3369508096}`,
		},
		{
			description: "two-claim-info-states",
			claimInfoStateList: ClaimInfoStateList{
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
								},
							},
						},
					},
					ClaimUID:  "067798be-454e-4be4-9047-1aa06aea63f7",
					ClaimName: "example-1",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
				{
					DriverState: map[string]DriverState{
						"test-driver.cdi.k8s.io": {
							Devices: []Device{
								{
									PoolName:   "worker-1",
									DeviceName: "dev-2",
								},
							},
						},
					},
					ClaimUID:  "4cf8db2d-06c0-7d70-1a51-e59b25b2c16c",
					ClaimName: "example-2",
					Namespace: "default",
					PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
				},
			},
			expectedCheckpointContent: `{"Data":"{\"kind\":\"DRACheckpoint\",\"apiVersion\":\"checkpoint.dra.kubelet.k8s.io/v1\",\"ClaimInfoStateList\":[{\"ClaimUID\":\"067798be-454e-4be4-9047-1aa06aea63f7\",\"ClaimName\":\"example-1\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-1\",\"RequestNames\":[\"test request\"],\"CDIDeviceIDs\":[\"example.com/example=cdi-example\"]}]}}},{\"ClaimUID\":\"4cf8db2d-06c0-7d70-1a51-e59b25b2c16c\",\"ClaimName\":\"example-2\",\"Namespace\":\"default\",\"PodUIDs\":{\"139cdb46-f989-4f17-9561-ca10cfb509a6\":{}},\"DriverState\":{\"test-driver.cdi.k8s.io\":{\"Devices\":[{\"PoolName\":\"worker-1\",\"DeviceName\":\"dev-2\",\"RequestNames\":null,\"CDIDeviceIDs\":null}]}}}]}","Checksum":1582256999}`,
		},
	}

	// Should return an error, stateDir cannot be an empty string
	if _, err := NewCheckpointer("", testingCheckpoint); err == nil {
		t.Fatal("expected error but got nil")
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "dramanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(testingDir); err != nil {
			t.Fatal(err)
		}
	}()

	// NewCheckpointState with an empty checkpointName should return an error
	if _, err = NewCheckpointer(testingDir, ""); err == nil {
		t.Fatal("expected error but got nil")
	}

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")
	require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

	cs, err := NewCheckpointer(testingDir, testingCheckpoint)
	require.NoError(t, err, "could not create testing checkpointState instance")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			checkpoint, err := NewCheckpoint(tc.claimInfoStateList)
			require.NoError(t, err, "could not create Checkpoint")

			err = cs.Store(checkpoint)
			require.NoError(t, err, "could not store checkpoint")

			err = cpm.GetCheckpoint(testingCheckpoint, checkpoint)
			require.NoError(t, err, "could not get checkpoint")

			checkpointContent, err := checkpoint.MarshalCheckpoint()
			require.NoError(t, err, "could not Marshal Checkpoint")
			assert.Equal(t, tc.expectedCheckpointContent, string(checkpointContent))
		})
	}
}
