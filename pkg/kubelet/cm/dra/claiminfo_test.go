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
	"fmt"
	"path"
	"reflect"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ClaimInfo test cases

func TestNewClaimInfoFromClaim(t *testing.T) {
	namespace := "test-namespace"
	className := "test-class"
	driverName := "test-plugin"
	claimUID := types.UID("claim-uid")
	claimName := "test-claim"

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
					DriverName: driverName,
					Allocation: &resourceapi.AllocationResult{
						ResourceHandles: []resourceapi.ResourceHandle{},
					},
				},
				Spec: resourceapi.ResourceClaimSpec{
					ResourceClassName: className,
				},
			},
			expectedResult: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClassName:  className,
					ClaimUID:   claimUID,
					ClaimName:  claimName,
					Namespace:  claimName,
					PodUIDs:    sets.New[string](),
					ResourceHandles: []resourceapi.ResourceHandle{
						{},
					},
					CDIDevices: make(map[string][]string),
				},
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
					DriverName: driverName,
					Allocation: &resourceapi.AllocationResult{},
				},
				Spec: resourceapi.ResourceClaimSpec{
					ResourceClassName: className,
				},
			},
			expectedResult: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClassName:  className,
					ClaimUID:   claimUID,
					ClaimName:  claimName,
					Namespace:  claimName,
					PodUIDs:    sets.New[string](),
					ResourceHandles: []resourceapi.ResourceHandle{
						{},
					},
					CDIDevices: make(map[string][]string),
				},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result := newClaimInfoFromClaim(test.claim)
			if reflect.DeepEqual(result, test.expectedResult) {
				t.Errorf("Expected %v, but got %v", test.expectedResult, result)
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
				DriverName:      "test-driver",
				ClassName:       "test-class",
				ClaimUID:        "test-uid",
				ClaimName:       "test-claim",
				Namespace:       "test-namespace",
				PodUIDs:         sets.New[string]("test-pod-uid"),
				ResourceHandles: []resourceapi.ResourceHandle{},
				CDIDevices:      map[string][]string{},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result := newClaimInfoFromState(test.state)
			if reflect.DeepEqual(result, test.expectedResult) {
				t.Errorf("Expected %v, but got %v", test.expectedResult, result)
			}
		})
	}
}

func TestClaimInfoSetCDIDevices(t *testing.T) {
	claimUID := types.UID("claim-uid")
	pluginName := "test-plugin"
	device := "vendor.com/device=device1"
	annotationName := fmt.Sprintf("cdi.k8s.io/%s_%s", pluginName, claimUID)
	for _, test := range []struct {
		description         string
		claimInfo           *ClaimInfo
		devices             []string
		expectedCDIDevices  map[string][]string
		expectedAnnotations map[string][]kubecontainer.Annotation
		wantErr             bool
	}{
		{
			description: "successfully add one device",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: pluginName,
					ClaimUID:   claimUID,
				},
			},
			devices: []string{device},
			expectedCDIDevices: map[string][]string{
				pluginName: {device},
			},
			expectedAnnotations: map[string][]kubecontainer.Annotation{
				pluginName: {
					{
						Name:  annotationName,
						Value: device,
					},
				},
			},
		},
		{
			description: "empty list of devices",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: pluginName,
					ClaimUID:   claimUID,
				},
			},
			devices:             []string{},
			expectedCDIDevices:  map[string][]string{pluginName: {}},
			expectedAnnotations: map[string][]kubecontainer.Annotation{pluginName: nil},
		},
		{
			description: "incorrect device format",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: pluginName,
					ClaimUID:   claimUID,
				},
			},
			devices: []string{"incorrect"},
			wantErr: true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			err := test.claimInfo.setCDIDevices(pluginName, test.devices)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, test.expectedCDIDevices, test.claimInfo.CDIDevices)
			assert.Equal(t, test.expectedAnnotations, test.claimInfo.annotations)
		})
	}
}

func TestClaimInfoAnnotationsAsList(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfo      *ClaimInfo
		expectedResult []kubecontainer.Annotation
	}{
		{
			description: "empty annotations",
			claimInfo: &ClaimInfo{
				annotations: map[string][]kubecontainer.Annotation{},
			},
		},
		{
			description: "nil annotations",
			claimInfo:   &ClaimInfo{},
		},
		{
			description: "valid annotations",
			claimInfo: &ClaimInfo{
				annotations: map[string][]kubecontainer.Annotation{
					"test-plugin1": {
						{
							Name:  "cdi.k8s.io/test-plugin1_claim-uid1",
							Value: "vendor.com/device=device1",
						},
						{
							Name:  "cdi.k8s.io/test-plugin1_claim-uid2",
							Value: "vendor.com/device=device2",
						},
					},
					"test-plugin2": {
						{
							Name:  "cdi.k8s.io/test-plugin2_claim-uid1",
							Value: "vendor.com/device=device1",
						},
						{
							Name:  "cdi.k8s.io/test-plugin2_claim-uid2",
							Value: "vendor.com/device=device2",
						},
					},
				},
			},
			expectedResult: []kubecontainer.Annotation{
				{
					Name:  "cdi.k8s.io/test-plugin1_claim-uid1",
					Value: "vendor.com/device=device1",
				},
				{
					Name:  "cdi.k8s.io/test-plugin1_claim-uid2",
					Value: "vendor.com/device=device2",
				},
				{
					Name:  "cdi.k8s.io/test-plugin2_claim-uid1",
					Value: "vendor.com/device=device1",
				},
				{
					Name:  "cdi.k8s.io/test-plugin2_claim-uid2",
					Value: "vendor.com/device=device2",
				},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result := test.claimInfo.annotationsAsList()
			sort.Slice(result, func(i, j int) bool {
				return result[i].Name < result[j].Name
			})
			assert.Equal(t, test.expectedResult, result)
		})
	}
}

func TestClaimInfoCDIdevicesAsList(t *testing.T) {
	for _, test := range []struct {
		description    string
		claimInfo      *ClaimInfo
		expectedResult []kubecontainer.CDIDevice
	}{
		{
			description: "empty CDI devices",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					CDIDevices: map[string][]string{},
				},
			},
		},
		{
			description: "nil CDI devices",
			claimInfo:   &ClaimInfo{},
		},
		{
			description: "valid CDI devices",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					CDIDevices: map[string][]string{
						"test-plugin1": {
							"vendor.com/device=device1",
							"vendor.com/device=device2",
						},
						"test-plugin2": {
							"vendor.com/device=device1",
							"vendor.com/device=device2",
						},
					},
				},
			},
			expectedResult: []kubecontainer.CDIDevice{
				{
					Name: "vendor.com/device=device1",
				},
				{
					Name: "vendor.com/device=device1",
				},
				{
					Name: "vendor.com/device=device2",
				},
				{
					Name: "vendor.com/device=device2",
				},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			result := test.claimInfo.cdiDevicesAsList()
			sort.Slice(result, func(i, j int) bool {
				return result[i].Name < result[j].Name
			})
			assert.Equal(t, test.expectedResult, result)
		})
	}
}
func TestClaimInfoAddPodReference(t *testing.T) {
	podUID := types.UID("pod-uid")
	for _, test := range []struct {
		description string
		claimInfo   *ClaimInfo
		expectedLen int
	}{
		{
			description: "successfully add pod reference",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](),
				},
			},
			expectedLen: 1,
		},
		{
			description: "duplicate pod reference",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					PodUIDs: sets.New[string](string(podUID)),
				},
			},
			expectedLen: 1,
		},
		{
			description: "duplicate pod reference",
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
	podUID := types.UID("pod-uid")
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
					PodUIDs: sets.New[string](string(podUID)),
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
			assert.Equal(t, test.claimInfo.hasPodReference(podUID), test.expectedResult)
		})
	}
}

func TestClaimInfoDeletePodReference(t *testing.T) {
	podUID := types.UID("pod-uid")
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
					PodUIDs: sets.New[string](string(podUID)),
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
			assert.Equal(t, test.claimInfo.isPrepared(), test.expectedResult)
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
			result, err := newClaimInfoCache(test.stateDir, test.checkpointName)
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
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
			cache, err := newClaimInfoCache(t.TempDir(), "test-checkpoint")
			assert.NoError(t, err)
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
			cache, err := newClaimInfoCache(t.TempDir(), "test-checkpoint")
			assert.NoError(t, err)
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
		description string
		claimInfo   *ClaimInfo
	}{
		{
			description: "claimInfo successfully added",
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: "test-claim",
					Namespace: "test-namespace",
				},
			},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			cache, err := newClaimInfoCache(t.TempDir(), "test-checkpoint")
			assert.NoError(t, err)
			assert.NotNil(t, cache)
			cache.add(test.claimInfo)
			assert.True(t, cache.contains(test.claimInfo.ClaimName, test.claimInfo.Namespace))
		})
	}
}

func TestClaimInfoCacheContains(t *testing.T) {
	claimName := "test-claim"
	namespace := "test-namespace"
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
	claimName := "test-claim"
	namespace := "test-namespace"
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
	claimName := "test-claim"
	namespace := "test-namespace"
	for _, test := range []struct {
		description    string
		claimInfoCache *claimInfoCache
	}{
		{
			description: "item in cache",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					claimName + namespace: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
						},
					},
				},
			},
		},
		{
			description:    "item not in cache",
			claimInfoCache: &claimInfoCache{},
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			test.claimInfoCache.delete(claimName, namespace)
			assert.False(t, test.claimInfoCache.contains(claimName, namespace))
		})
	}
}

func TestClaimInfoCacheHasPodReference(t *testing.T) {
	claimName := "test-claim"
	namespace := "test-namespace"
	uid := types.UID("test-uid")
	for _, test := range []struct {
		description    string
		claimInfoCache *claimInfoCache
		expectedResult bool
	}{
		{
			description: "uid is referenced",
			claimInfoCache: &claimInfoCache{
				claimInfo: map[string]*ClaimInfo{
					claimName + namespace: {
						ClaimInfoState: state.ClaimInfoState{
							ClaimName: claimName,
							Namespace: namespace,
							PodUIDs:   sets.New[string](string(uid)),
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
			assert.Equal(t, test.expectedResult, test.claimInfoCache.hasPodReference(uid))
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
			cache, err := newClaimInfoCache(test.stateDir, test.checkpointName)
			assert.NoError(t, err)
			err = cache.syncToCheckpoint()
			if test.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
		})
	}
}
