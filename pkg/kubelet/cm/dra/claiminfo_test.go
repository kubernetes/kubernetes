/*
Copyright 2024 The Kubernetes Authors.

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

package dra

import (
	"errors"
	"path"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

// ClaimInfo test cases

const (
	namespace     = "test-namespace"
	className     = "test-class"
	driverName    = "test-driver"
	deviceName    = "test-device"                      // name inside ResourceSlice
	cdiDeviceName = "cdi-test-device"                  // name inside CDI spec
	cdiID         = "test-driver/test=cdi-test-device" // CDI device ID
	poolName      = "test-pool"
	requestName   = "test-request"
	claimName     = "test-claim"
	claimUID      = types.UID(claimName + "-uid")
	podUID        = "test-pod-uid"
)

var (
	device = state.Device{
		PoolName:     poolName,
		DeviceName:   deviceName,
		RequestNames: []string{requestName},
		CDIDeviceIDs: []string{cdiID},
	}
	devices = []state.Device{device}
)

func TestNewClaimInfoFromClaim(t *testing.T) {
	for _, test := range []struct {
		description    string
		claim          *resourceapi.ResourceClaim
		expectedResult *ClaimInfo
	}{
		{
			description: "successfully created object",
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					UID:       claimUID,
					Name:      claimName,
					Namespace: namespace,
				},
				Status: resourceapi.ResourceClaimStatus{
					Allocation: &resourceapi.AllocationResult{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: requestName,
									Pool:    poolName,
									Device:  deviceName,
									Driver:  driverName,
								},
							},
						},
					},
				},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Name:            requestName,
								DeviceClassName: className,
							},
						},
					},
				},
			},
			expectedResult: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:  claimUID,
					ClaimName: claimName,
					Namespace: namespace,
					PodUIDs:   sets.New[string](),
					DriverState: map[string]state.DriverState{
						driverName: {},
					},
				},
				prepared: false,
			},
		},
		{
			description: "successfully created object with empty allocation",
			claim: &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					UID:       claimUID,
					Name:      claimName,
					Namespace: namespace,
				},
				Status: resourceapi.ResourceClaimStatus{
					Allocation: &resourceapi.AllocationResult{},
				},
				Spec: resourceapi.ResourceClaimSpec{},
			},
			expectedResult: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:    claimUID,
					ClaimName:   claimName,
					Namespace:   namespace,
					PodUIDs:     sets.New[string](),
					DriverState: map[string]state.DriverState{},
				},
				prepared: false,
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result, err := newClaimInfoFromClaim(test.claim)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedResult, result) {
				t.Errorf("Expected %+v, but got %+v", test.expectedResult, result)
			}
		})
	}
}

func TestNewClaimInfoFromState(t *testing.T) {
	for _, test := range []struct {
		description    string
		state          *state.ClaimInfoState
		expectedResult *ClaimInfo
	}{
		{
			description: "successfully created object",
			state: &state.ClaimInfoState{
				ClaimUID:  claimUID,
				ClaimName: claimName,
				Namespace: namespace,
				PodUIDs:   sets.New[string](podUID),
				DriverState: map[string]state.DriverState{
					driverName: {
						Devices: devices,
					},
				},
			},
			expectedResult: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:  claimUID,
					ClaimName: claimName,
					Namespace: namespace,
					PodUIDs:   sets.New[string](podUID),
					DriverState: map[string]state.DriverState{
						driverName: {
							Devices: devices,
						},
					},
				},
				prepared: false,
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result := newClaimInfoFromState(test.state)
			if !reflect.DeepEqual(result, test.expectedResult) {
				t.Errorf("Expected %+v, but got %+v", test.expectedResult, result)
			}
		})
	}
}

func TestClaimInfoAddDevice(t *testing.T) {
	for _, test := range []struct {
		description string
		claimInfo   *ClaimInfo
		device      state.Device
	}{
		{
			description: "add new device",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:  claimUID,
					ClaimName: claimName,
					Namespace: namespace,
					PodUIDs:   sets.New[string](podUID),
				},
				prepared: false,
			},
			device: device,
		},
		{
			description: "other new device",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:  claimUID,
					ClaimName: claimName,
					Namespace: namespace,
					PodUIDs:   sets.New[string](podUID),
				},
			},
			device: func() state.Device {
				device := device
				device.PoolName += "-2"
				device.DeviceName += "-2"
				device.RequestNames = []string{device.RequestNames[0] + "-2"}
				device.CDIDeviceIDs = []string{device.CDIDeviceIDs[0] + "-2"}
				return device
			}(),
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			test.claimInfo.addDevice(driverName, test.device)
			assert.Equal(t, []state.Device{test.device}, test.claimInfo.DriverState[driverName].Devices)
		})
	}
}

func TestClaimInfoAddPodReference(t *testing.T) {
	for _, test := range []struct {
		description string
		claimInfo   *ClaimInfo
		expectedLen int
	}{
		{
			description: "empty PodUIDs list",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](),
				},
			},
			expectedLen: 1,
		},
		{
			description: "first pod reference",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](podUID),
				},
			},
			expectedLen: 1,
		},
		{
			description: "second pod reference",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string]("pod-uid1"),
				},
			},
			expectedLen: 2,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			test.claimInfo.addPodReference(podUID)
			assert.True(t, test.claimInfo.hasPodReference(podUID))
			assert.Len(t, test.claimInfo.PodUIDs, test.expectedLen)
		})
	}
}

func TestClaimInfoHasPodReference(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfo      *ClaimInfo
		expectedResult bool
	}{
		{
			description: "claim doesn't reference pod",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](),
				},
			},
		},
		{
			description: "claim references pod",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](podUID),
				},
			},
			expectedResult: true,
		},
		{
			description: "empty claim info",
			claimInfo:   &ClaimInfo{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			assert.Equal(t, test.expectedResult, test.claimInfo.hasPodReference(podUID))
		})
	}
}

func TestClaimInfoDeletePodReference(t *testing.T) {
	for _, test := range []struct {
		description string
		claimInfo   *ClaimInfo
	}{
		{
			description: "claim doesn't reference pod",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](),
				},
			},
		},
		{
			description: "claim references pod",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](podUID),
				},
			},
		},
		{
			description: "empty claim info",
			claimInfo:   &ClaimInfo{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			test.claimInfo.deletePodReference(podUID)
			assert.False(t, test.claimInfo.hasPodReference(podUID))
		})
	}
}

func TestClaimInfoSetPrepared(t *testing.T) {
	for _, test := range []struct {
		description string
		claimInfo   *ClaimInfo
	}{
		{
			description: "claim info is not prepared",
			claimInfo: &ClaimInfo{
				prepared: false,
			},
		},
		{
			description: "claim info is prepared",
			claimInfo: &ClaimInfo{
				prepared: true,
			},
		},
		{
			description: "empty claim info",
			claimInfo:   &ClaimInfo{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			test.claimInfo.setPrepared()
			assert.True(t, test.claimInfo.isPrepared())
		})
	}
}

func TestClaimInfoIsPrepared(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfo      *ClaimInfo
		expectedResult bool
	}{
		{
			description: "claim info is not prepared",
			claimInfo: &ClaimInfo{
				prepared: false,
			},
			expectedResult: false,
		},
		{
			description: "claim info is prepared",
			claimInfo: &ClaimInfo{
				prepared: true,
			},
			expectedResult: true,
		},
		{
			description:    "empty claim info",
			claimInfo:      &ClaimInfo{},
			expectedResult: false,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			assert.Equal(t, test.expectedResult, test.claimInfo.isPrepared())
		})
	}
}

// claimInfoCache test cases
func TestNewClaimInfoCache(t *testing.T) {
	for _, test := range []struct {
		description    string
		stateDir       string
		checkpointName string
		wantErr        bool
	}{
		{
			description:    "successfully created cache",
			stateDir:       t.TempDir(),
			checkpointName: "test-checkpoint",
		},
		{
			description: "empty parameters",
			wantErr:     true,
		},
		{
			description: "empty checkpoint name",
			stateDir:    t.TempDir(),
			wantErr:     true,
		},
		{
			description: "incorrect checkpoint name",
			stateDir:    path.Join(t.TempDir(), "incorrect checkpoint"),
			wantErr:     true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			result, err := newClaimInfoCache(tCtx.Logger(), test.stateDir, test.checkpointName)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.NotNil(t, result)
		})
	}
}

func TestClaimInfoCacheWithLock(t *testing.T) {
	for _, test := range []struct {
		description string
		funcGen     func(cache *claimInfoCache) func() error
		wantErr     bool
	}{
		{
			description: "cache is locked inside a function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					if cache.RWMutex.TryLock() {
						return errors.New("Lock succeeded")
					}
					return nil
				}
			},
		},
		{
			description: "cache is Rlocked inside a function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					if cache.RWMutex.TryRLock() {
						return errors.New("RLock succeeded")
					}
					return nil
				}
			},
		},
		{
			description: "successfully called function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					return nil
				}
			},
		},
		{
			description: "erroring function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					return errors.New("test error")
				}
			},
			wantErr: true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			cache, err := newClaimInfoCache(tCtx.Logger(), t.TempDir(), "test-checkpoint")
			require.NoError(t, err)
			assert.NotNil(t, cache)
			err = cache.withLock(test.funcGen(cache))
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestClaimInfoCacheWithRLock(t *testing.T) {
	for _, test := range []struct {
		description string
		funcGen     func(cache *claimInfoCache) func() error
		wantErr     bool
	}{
		{
			description: "RLock-ed cache allows another RLock",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					if !cache.RWMutex.TryRLock() {
						return errors.New("RLock failed")
					}
					return nil
				}
			},
		},
		{
			description: "cache is locked inside a function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					if cache.RWMutex.TryLock() {
						return errors.New("Lock succeeded")
					}
					return nil
				}
			},
		},
		{
			description: "successfully called function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					return nil
				}
			},
		},
		{
			description: "erroring function",
			funcGen: func(cache *claimInfoCache) func() error {
				return func() error {
					return errors.New("test error")
				}
			},
			wantErr: true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			cache, err := newClaimInfoCache(tCtx.Logger(), t.TempDir(), "test-checkpoint")
			require.NoError(t, err)
			assert.NotNil(t, cache)
			err = cache.withRLock(test.funcGen(cache))
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestClaimInfoCacheAdd(t *testing.T) {
	for _, test := range []struct {
		description      string
		initialClaimInfo []*ClaimInfo
		claimInfo        *ClaimInfo
		expectMetrics    string
		expectLog        string
	}{
		{
			description: "claimInfo successfully added",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName,
					Namespace: namespace,
				},
			},
			expectMetrics: `# HELP dra_resource_claims_in_use [ALPHA] The number of ResourceClaims that are currently in use on the node, by driver name (driver_name label value) and across all drivers (special value <any> for driver_name). Note that the sum of all by-driver counts is not the total number of in-use ResourceClaims because the same ResourceClaim might use devices from different drivers. Instead, use the count for the <any> driver_name.
# TYPE dra_resource_claims_in_use gauge
dra_resource_claims_in_use{driver_name="<any>"} 0
`,
		},
		{
			description: "prepared claimInfo",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName,
					Namespace: namespace,
					DriverState: map[string]state.DriverState{
						"test-driver":       {},
						"other-test-driver": {},
					},
				},
				prepared: true,
			},
			expectMetrics: `# HELP dra_resource_claims_in_use [ALPHA] The number of ResourceClaims that are currently in use on the node, by driver name (driver_name label value) and across all drivers (special value <any> for driver_name). Note that the sum of all by-driver counts is not the total number of in-use ResourceClaims because the same ResourceClaim might use devices from different drivers. Instead, use the count for the <any> driver_name.
# TYPE dra_resource_claims_in_use gauge
dra_resource_claims_in_use{driver_name="<any>"} 1
dra_resource_claims_in_use{driver_name="other-test-driver"} 1
dra_resource_claims_in_use{driver_name="test-driver"} 1
`,
			expectLog: `INFO ResourceClaim usage changed claimsInUse=<
	<any>: 1 (+1)
	other-test-driver: 1 (+1)
	test-driver: 1 (+1)
 >
`,
		},
		{
			description: "add more prepared claimInfo",
			initialClaimInfo: []*ClaimInfo{{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName + "-old",
					Namespace: namespace,
					DriverState: map[string]state.DriverState{
						"test-driver": {},
					},
				},
				prepared: true,
			}},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName,
					Namespace: namespace,
					DriverState: map[string]state.DriverState{
						"test-driver":       {},
						"other-test-driver": {},
					},
				},
				prepared: true,
			},
			expectMetrics: `# HELP dra_resource_claims_in_use [ALPHA] The number of ResourceClaims that are currently in use on the node, by driver name (driver_name label value) and across all drivers (special value <any> for driver_name). Note that the sum of all by-driver counts is not the total number of in-use ResourceClaims because the same ResourceClaim might use devices from different drivers. Instead, use the count for the <any> driver_name.
# TYPE dra_resource_claims_in_use gauge
dra_resource_claims_in_use{driver_name="<any>"} 2
dra_resource_claims_in_use{driver_name="other-test-driver"} 1
dra_resource_claims_in_use{driver_name="test-driver"} 2
`,
			expectLog: `INFO ResourceClaim usage changed claimsInUse=<
	<any>: 2 (+1)
	other-test-driver: 1 (+1)
	test-driver: 2 (+1)
 >
`,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t, initoption.BufferLogs(true))
			cache, err := newClaimInfoCache(tCtx.Logger(), t.TempDir(), "test-checkpoint")
			for _, claimInfo := range test.initialClaimInfo {
				cache.add(claimInfo)
			}
			require.NoError(t, err)
			assert.NotNil(t, cache)
			_ = cache.withLock(func() error {
				cache.add(test.claimInfo)
				return nil
			})
			assert.True(t, cache.contains(test.claimInfo.ClaimName, test.claimInfo.Namespace))
			testClaimsInUseMetric(tCtx, cache, test.expectMetrics)
			logOutput := tCtx.Logger().GetSink().(ktesting.Underlier).GetBuffer()
			assert.Equal(t, test.expectLog, logOutput.String())
		})
	}
}

func TestClaimInfoCacheContains(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfo      *ClaimInfo
		claimInfoCache *claimInfoCache
		expectedResult bool
	}{
		{
			description: "cache hit",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					namespace + "/" + claimName: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName,
					Namespace: namespace,
				},
			},
			expectedResult: true,
		},
		{
			description:    "cache miss",
			claimInfoCache: &claimInfoCache{},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: claimName,
					Namespace: namespace,
				},
			},
		},
		{
			description:    "cache miss: empty cache and empty claim info",
			claimInfoCache: &claimInfoCache{},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			assert.Equal(t, test.expectedResult, test.claimInfoCache.contains(test.claimInfo.ClaimName, test.claimInfo.Namespace))
		})
	}
}

func TestClaimInfoCacheGet(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfoCache *claimInfoCache
		expectedNil    bool
		expectedExists bool
	}{
		{
			description: "cache hit",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					namespace + "/" + claimName: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
						},
					},
				},
			},
			expectedExists: true,
		},
		{
			description:    "cache miss",
			claimInfoCache: &claimInfoCache{},
			expectedNil:    true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result, exists := test.claimInfoCache.get(claimName, namespace)
			assert.Equal(t, test.expectedExists, exists)
			assert.Equal(t, test.expectedNil, result == nil)
		})
	}
}

func TestClaimInfoCacheDelete(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfoCache *claimInfoCache
		expectMetrics  string
		expectLog      string
	}{
		{
			description: "item in cache",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					namespace + "/" + claimName: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
							DriverState: map[string]state.DriverState{
								"test-driver": {},
							},
						},
						prepared: true,
					},
					namespace + "/" + claimName + "-old": {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
							DriverState: map[string]state.DriverState{
								"test-driver":       {},
								"other-test-driver": {},
							},
						},
						prepared: true,
					},
				},
			},
			expectMetrics: `# HELP dra_resource_claims_in_use [ALPHA] The number of ResourceClaims that are currently in use on the node, by driver name (driver_name label value) and across all drivers (special value <any> for driver_name). Note that the sum of all by-driver counts is not the total number of in-use ResourceClaims because the same ResourceClaim might use devices from different drivers. Instead, use the count for the <any> driver_name.
# TYPE dra_resource_claims_in_use gauge
dra_resource_claims_in_use{driver_name="<any>"} 1
dra_resource_claims_in_use{driver_name="test-driver"} 1
dra_resource_claims_in_use{driver_name="other-test-driver"} 1
`,
			expectLog: `INFO ResourceClaim usage changed claimsInUse=<
	<any>: 1 (-1)
	other-test-driver: 1 (+0)
	test-driver: 1 (-1)
 >
`,
		},
		{
			description:    "item not in cache",
			claimInfoCache: &claimInfoCache{},
			expectMetrics: `# HELP dra_resource_claims_in_use [ALPHA] The number of ResourceClaims that are currently in use on the node, by driver name (driver_name label value) and across all drivers (special value <any> for driver_name). Note that the sum of all by-driver counts is not the total number of in-use ResourceClaims because the same ResourceClaim might use devices from different drivers. Instead, use the count for the <any> driver_name.
# TYPE dra_resource_claims_in_use gauge
dra_resource_claims_in_use{driver_name="<any>"} 0
`,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t, initoption.BufferLogs(true))
			test.claimInfoCache.logger = tCtx.Logger()
			_ = test.claimInfoCache.withLock(func() error {
				test.claimInfoCache.delete(claimName, namespace)
				return nil
			})
			assert.False(t, test.claimInfoCache.contains(claimName, namespace))
			testClaimsInUseMetric(tCtx, test.claimInfoCache, test.expectMetrics)
			logOutput := tCtx.Logger().GetSink().(ktesting.Underlier).GetBuffer()
			assert.Equal(t, test.expectLog, logOutput.String())
		})
	}
}

func TestClaimInfoCacheHasPodReference(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfoCache *claimInfoCache
		expectedResult bool
	}{
		{
			description: "uid is referenced",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					namespace + "/" + claimName: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
							PodUIDs:   sets.New[string](podUID),
						},
					},
				},
			},
			expectedResult: true,
		},
		{
			description:    "uid is not referenced",
			claimInfoCache: &claimInfoCache{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			assert.Equal(t, test.expectedResult, test.claimInfoCache.hasPodReference(podUID))
		})
	}
}

func TestSyncToCheckpoint(t *testing.T) {
	for _, test := range []struct {
		description    string
		stateDir       string
		checkpointName string
		wantErr        bool
	}{
		{
			description:    "successfully checkpointed cache",
			stateDir:       t.TempDir(),
			checkpointName: "test-checkpoint",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			cache, err := newClaimInfoCache(tCtx.Logger(), test.stateDir, test.checkpointName)
			require.NoError(t, err)
			err = cache.syncToCheckpoint()
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
		})
	}
}
