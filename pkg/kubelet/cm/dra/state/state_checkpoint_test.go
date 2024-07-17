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
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	cmerrors "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
)

const testingCheckpoint = "dramanager_checkpoint_test"

// assertStateEqual marks provided test as failed if provided states differ
func assertStateEqual(t *testing.T, restoredState, expectedState ClaimInfoStateList) {
	assert.Equal(t, expectedState, restoredState, "expected ClaimInfoState does not equal to restored one")
}

// TODO (https://github.com/kubernetes/kubernetes/issues/123552): reconsider what data gets stored in checkpoints and whether that is really necessary.
//
// As it stands now, a "v1" checkpoint contains data for types like the resourceapi.ResourceHandle
// which may change over time as new fields get added in a backward-compatible way (not unusual
// for API types). That breaks checksuming with pkg/util/hash because it is based on spew output.
// That output includes those new fields.

func TestCheckpointGetOrCreate(t *testing.T) {
	testCases := []struct {
		description       string
		checkpointContent string
		expectedError     string
		expectedState     ClaimInfoStateList
	}{
		{
			description:       "Create non-existing checkpoint",
			checkpointContent: "",
			expectedError:     "",
			expectedState:     []ClaimInfoState{},
		},
		{
			description:       "Restore checkpoint - single claim",
			checkpointContent: `{"version":"v1","entries":[{"ClaimUID":"067798be-454e-4be4-9047-1aa06aea63f7","ClaimName":"example","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-1","RequestNames":["test request"],"CDIDeviceIDs":["example.com/example=cdi-example"]}]}}}],"checksum":1925941526}`,
			expectedState: []ClaimInfoState{
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
			description:       "Restore checkpoint - single claim - multiple devices",
			checkpointContent: `{"version":"v1","entries":[{"ClaimUID":"067798be-454e-4be4-9047-1aa06aea63f7","ClaimName":"example","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-1","RequestNames":["test request"],"CDIDeviceIDs":["example.com/example=cdi-example"]},{"PoolName":"worker-1","DeviceName":"dev-2","RequestNames":["test request2"],"CDIDeviceIDs":["example.com/example=cdi-example2"]}]}}}],"checksum":3560752542}`,
			expectedError:     "",
			expectedState: []ClaimInfoState{
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
									RequestNames: []string{"test request2"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example2"},
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
			description:       "Restore checkpoint - multiple claims",
			checkpointContent: `{"version":"v1","entries":[{"ClaimUID":"067798be-454e-4be4-9047-1aa06aea63f7","ClaimName":"example-1","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-1","RequestNames":["test request"],"CDIDeviceIDs":["example.com/example=cdi-example"]}]}}},{"ClaimUID":"4cf8db2d-06c0-7d70-1a51-e59b25b2c16c","ClaimName":"example-2","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-1","RequestNames":["test request"],"CDIDeviceIDs":["example.com/example=cdi-example"]}]}}}],"checksum":351581974}`,
			expectedError:     "",
			expectedState: []ClaimInfoState{
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
									PoolName:     "worker-1",
									DeviceName:   "dev-1",
									RequestNames: []string{"test request"},
									CDIDeviceIDs: []string{"example.com/example=cdi-example"},
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
			description:       "Restore checkpoint - invalid checksum",
			checkpointContent: `{"version":"v1","entries":[{"DriverName":"test-driver.cdi.k8s.io","ClassName":"class-name","ClaimUID":"067798be-454e-4be4-9047-1aa06aea63f7","ClaimName":"example","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"CDIDevices":{"test-driver.cdi.k8s.io":["example.com/example=cdi-example"]}}],"checksum":1988120168}`,
			expectedError:     "checkpoint is corrupted",
			expectedState:     []ClaimInfoState{},
		},
		{
			description:       "Restore checkpoint with invalid JSON",
			checkpointContent: `{`,
			expectedError:     "unexpected end of JSON input",
			expectedState:     []ClaimInfoState{},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "dramanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	assert.NoError(t, err, "could not create testing checkpoint manager")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			assert.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

			// prepare checkpoint for testing
			if strings.TrimSpace(tc.checkpointContent) != "" {
				checkpoint := &testutil.MockCheckpoint{Content: tc.checkpointContent}
				assert.NoError(t, cpm.CreateCheckpoint(testingCheckpoint, checkpoint), "could not create testing checkpoint")
			}

			var state ClaimInfoStateList

			checkpointState, err := NewCheckpointState(testingDir, testingCheckpoint)

			if err == nil {
				state, err = checkpointState.GetOrCreate()
			}
			if strings.TrimSpace(tc.expectedError) != "" {
				assert.ErrorContains(t, err, tc.expectedError)
			} else {
				requireNoCheckpointError(t, err)
				// compare state after restoration with the one expected
				assertStateEqual(t, state, tc.expectedState)
			}
		})
	}
}

func TestCheckpointStateStore(t *testing.T) {
	claimInfoStateList := ClaimInfoStateList{
		{
			DriverState: map[string]DriverState{
				"test-driver.cdi.k8s.io": {
					Devices: []Device{{
						PoolName:     "worker-1",
						DeviceName:   "dev-1",
						RequestNames: []string{"test request"},
						CDIDeviceIDs: []string{"example.com/example=cdi-example"},
					}},
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
					Devices: []Device{{
						PoolName:   "worker-1",
						DeviceName: "dev-2",
					}},
				},
			},
			ClaimUID:  "4cf8db2d-06c0-7d70-1a51-e59b25b2c16c",
			ClaimName: "example-2",
			Namespace: "default",
			PodUIDs:   sets.New("139cdb46-f989-4f17-9561-ca10cfb509a6"),
		},
	}

	expectedCheckpoint := `{"version":"v1","entries":[{"ClaimUID":"067798be-454e-4be4-9047-1aa06aea63f7","ClaimName":"example-1","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-1","RequestNames":["test request"],"CDIDeviceIDs":["example.com/example=cdi-example"]}]}}},{"ClaimUID":"4cf8db2d-06c0-7d70-1a51-e59b25b2c16c","ClaimName":"example-2","Namespace":"default","PodUIDs":{"139cdb46-f989-4f17-9561-ca10cfb509a6":{}},"DriverState":{"test-driver.cdi.k8s.io":{"Devices":[{"PoolName":"worker-1","DeviceName":"dev-2","RequestNames":null,"CDIDeviceIDs":null}]}}}],"checksum":1191151426}`

	// Should return an error, stateDir cannot be an empty string
	if _, err := NewCheckpointState("", testingCheckpoint); err == nil {
		t.Fatal("expected error but got nil")
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "dramanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	assert.NoError(t, err, "could not create testing checkpoint manager")
	assert.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

	cs, err := NewCheckpointState(testingDir, testingCheckpoint)
	assert.NoError(t, err, "could not create testing checkpointState instance")
	err = cs.Store(claimInfoStateList)
	assert.NoError(t, err, "could not store ClaimInfoState")
	checkpoint := NewDRAManagerCheckpoint()
	cpm.GetCheckpoint(testingCheckpoint, checkpoint)
	checkpointData, err := checkpoint.MarshalCheckpoint()
	assert.NoError(t, err, "could not Marshal Checkpoint")
	assert.Equal(t, expectedCheckpoint, string(checkpointData), "expected ClaimInfoState does not equal to restored one")

	// NewCheckpointState with an empty checkpointName should return an error
	if _, err = NewCheckpointState(testingDir, ""); err == nil {
		t.Fatal("expected error but got nil")
	}
}

func requireNoCheckpointError(t *testing.T, err error) {
	t.Helper()
	var cksumErr *cmerrors.CorruptCheckpointError
	if errors.As(err, &cksumErr) {
		t.Fatalf("unexpected corrupt checkpoint, expected checksum %d, got %d", cksumErr.ExpectedCS, cksumErr.ActualCS)
	} else {
		require.NoError(t, err, "could not restore checkpoint")
	}
}
