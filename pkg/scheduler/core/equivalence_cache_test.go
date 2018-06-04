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

package core

import (
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

// makeBasicPod returns a Pod object with many of the fields populated.
func makeBasicPod(name string) *v1.Pod {
	isController := true
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "test-ns",
			Labels:    map[string]string{"app": "web", "env": "prod"},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "ReplicationController",
					Name:       "rc",
					UID:        "123",
					Controller: &isController,
				},
			},
		},
		Spec: v1.PodSpec{
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "failure-domain.beta.kubernetes.io/zone",
										Operator: "Exists",
									},
								},
							},
						},
					},
				},
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "db"}},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "web"}},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name:  "init-pause",
					Image: "gcr.io/google_containers/pause",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"cpu": resource.MustParse("1"),
							"mem": resource.MustParse("100Mi"),
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: "gcr.io/google_containers/pause",
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"cpu": resource.MustParse("1"),
							"mem": resource.MustParse("100Mi"),
						},
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nfs",
							MountPath: "/srv/data",
						},
					},
				},
			},
			NodeSelector: map[string]string{"node-type": "awesome"},
			Tolerations: []v1.Toleration{
				{
					Effect:   "NoSchedule",
					Key:      "experimental",
					Operator: "Exists",
				},
			},
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol2",
						},
					},
				},
				{
					Name: "nfs",
					VolumeSource: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server: "nfs.corp.example.com",
						},
					},
				},
			},
		},
	}
}

type predicateItemType struct {
	fit     bool
	reasons []algorithm.PredicateFailureReason
}

// upToDateCache is a fake Cache where IsUpToDate always returns true.
type upToDateCache = schedulertesting.FakeCache

// staleNodeCache is a fake Cache where IsUpToDate always returns false.
type staleNodeCache struct {
	schedulertesting.FakeCache
}

func (c *staleNodeCache) IsUpToDate(*schedulercache.NodeInfo) bool { return false }

// mockPredicate provides an algorithm.FitPredicate with pre-set return values.
type mockPredicate struct {
	fit       bool
	reasons   []algorithm.PredicateFailureReason
	err       error
	callCount int
}

func (p *mockPredicate) predicate(*v1.Pod, algorithm.PredicateMetadata, *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	p.callCount++
	return p.fit, p.reasons, p.err
}

func TestRunPredicate(t *testing.T) {
	tests := []struct {
		name                                        string
		pred                                        mockPredicate
		cache                                       schedulercache.Cache
		expectFit, expectCacheHit, expectCacheWrite bool
		expectedReasons                             []algorithm.PredicateFailureReason
		expectedError                               string
	}{
		{
			name:             "pod fits/cache hit",
			pred:             mockPredicate{},
			cache:            &upToDateCache{},
			expectFit:        true,
			expectCacheHit:   true,
			expectCacheWrite: false,
		},
		{
			name:             "pod fits/cache miss",
			pred:             mockPredicate{fit: true},
			cache:            &upToDateCache{},
			expectFit:        true,
			expectCacheHit:   false,
			expectCacheWrite: true,
		},
		{
			name:             "pod fits/cache miss/no write",
			pred:             mockPredicate{fit: true},
			cache:            &staleNodeCache{},
			expectFit:        true,
			expectCacheHit:   false,
			expectCacheWrite: false,
		},
		{
			name:             "pod doesn't fit/cache miss",
			pred:             mockPredicate{reasons: []algorithm.PredicateFailureReason{predicates.ErrFakePredicate}},
			cache:            &upToDateCache{},
			expectFit:        false,
			expectCacheHit:   false,
			expectCacheWrite: true,
			expectedReasons:  []algorithm.PredicateFailureReason{predicates.ErrFakePredicate},
		},
		{
			name:             "pod doesn't fit/cache hit",
			pred:             mockPredicate{},
			cache:            &upToDateCache{},
			expectFit:        false,
			expectCacheHit:   true,
			expectCacheWrite: false,
			expectedReasons:  []algorithm.PredicateFailureReason{predicates.ErrFakePredicate},
		},
		{
			name:             "predicate error",
			pred:             mockPredicate{err: errors.New("This is expected")},
			cache:            &upToDateCache{},
			expectFit:        false,
			expectCacheHit:   false,
			expectCacheWrite: false,
			expectedError:    "This is expected",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := schedulercache.NewNodeInfo()
			node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n1"}})
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1"}}
			meta := algorithm.EmptyPredicateMetadataProducer(nil, nil)

			ecache := NewEquivalenceCache()
			equivClass := ecache.getEquivalenceClassInfo(pod)
			if test.expectCacheHit {
				ecache.updateResult(pod.Name, "testPredicate", test.expectFit, test.expectedReasons, equivClass.hash, test.cache, node)
			}

			fit, reasons, err := ecache.RunPredicate(test.pred.predicate, "testPredicate", pod, meta, node, equivClass, test.cache)

			if err != nil {
				if err.Error() != test.expectedError {
					t.Errorf("Expected error %v but got %v", test.expectedError, err)
				}
			} else if len(test.expectedError) > 0 {
				t.Errorf("Expected error %v but got nil", test.expectedError)
			}
			if fit && !test.expectFit {
				t.Errorf("pod should not fit")
			}
			if !fit && test.expectFit {
				t.Errorf("pod should fit")
			}
			if len(reasons) != len(test.expectedReasons) {
				t.Errorf("Expected failures: %v but got %v", test.expectedReasons, reasons)
			} else {
				for i, reason := range reasons {
					if reason != test.expectedReasons[i] {
						t.Errorf("Expected failures: %v but got %v", test.expectedReasons, reasons)
						break
					}
				}
			}
			if test.expectCacheHit && test.pred.callCount != 0 {
				t.Errorf("Predicate should not be called")
			}
			if !test.expectCacheHit && test.pred.callCount == 0 {
				t.Errorf("Predicate should be called")
			}
			_, _, invalid := ecache.lookupResult(pod.Name, node.Node().Name, "testPredicate", equivClass.hash)
			if invalid && test.expectCacheWrite {
				t.Errorf("Cache write should happen")
			}
			if !test.expectCacheHit && test.expectCacheWrite && invalid {
				t.Errorf("Cache write should happen")
			}
			if !test.expectCacheHit && !test.expectCacheWrite && !invalid {
				t.Errorf("Cache write should not happen")
			}
		})
	}
}

func TestUpdateResult(t *testing.T) {
	tests := []struct {
		name               string
		pod                string
		predicateKey       string
		nodeName           string
		fit                bool
		reasons            []algorithm.PredicateFailureReason
		equivalenceHash    uint64
		expectPredicateMap bool
		expectCacheItem    HostPredicate
		cache              schedulercache.Cache
	}{
		{
			name:               "test 1",
			pod:                "testPod",
			predicateKey:       "GeneralPredicates",
			nodeName:           "node1",
			fit:                true,
			equivalenceHash:    123,
			expectPredicateMap: false,
			expectCacheItem: HostPredicate{
				Fit: true,
			},
			cache: &upToDateCache{},
		},
		{
			name:               "test 2",
			pod:                "testPod",
			predicateKey:       "GeneralPredicates",
			nodeName:           "node2",
			fit:                false,
			equivalenceHash:    123,
			expectPredicateMap: true,
			expectCacheItem: HostPredicate{
				Fit: false,
			},
			cache: &upToDateCache{},
		},
	}
	for _, test := range tests {
		ecache := NewEquivalenceCache()
		if test.expectPredicateMap {
			ecache.algorithmCache[test.nodeName] = AlgorithmCache{}
			predicateItem := HostPredicate{
				Fit: true,
			}
			ecache.algorithmCache[test.nodeName][test.predicateKey] =
				PredicateMap{
					test.equivalenceHash: predicateItem,
				}
		}

		node := schedulercache.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}})
		ecache.updateResult(
			test.pod,
			test.predicateKey,
			test.fit,
			test.reasons,
			test.equivalenceHash,
			test.cache,
			node,
		)

		cachedMapItem, ok := ecache.algorithmCache[test.nodeName][test.predicateKey]
		if !ok {
			t.Errorf("Failed: %s, can't find expected cache item: %v",
				test.name, test.expectCacheItem)
		} else {
			if !reflect.DeepEqual(cachedMapItem[test.equivalenceHash], test.expectCacheItem) {
				t.Errorf("Failed: %s, expected cached item: %v, but got: %v",
					test.name, test.expectCacheItem, cachedMapItem[test.equivalenceHash])
			}
		}
	}
}

// slicesEqual wraps reflect.DeepEqual, but returns true when comparing nil and empty slice.
func slicesEqual(a, b []algorithm.PredicateFailureReason) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	return reflect.DeepEqual(a, b)
}

func TestLookupResult(t *testing.T) {
	tests := []struct {
		name                              string
		podName                           string
		nodeName                          string
		predicateKey                      string
		equivalenceHashForUpdatePredicate uint64
		equivalenceHashForCalPredicate    uint64
		cachedItem                        predicateItemType
		expectedInvalidPredicateKey       bool
		expectedInvalidEquivalenceHash    bool
		expectedPredicateItem             predicateItemType
		cache                             schedulercache.Cache
	}{
		{
			name:     "test 1",
			podName:  "testPod",
			nodeName: "node1",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
			cache: &upToDateCache{},
		},
		{
			name:     "test 2",
			podName:  "testPod",
			nodeName: "node2",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit: true,
			},
			expectedInvalidPredicateKey: false,
			expectedPredicateItem: predicateItemType{
				fit:     true,
				reasons: []algorithm.PredicateFailureReason{},
			},
			cache: &upToDateCache{},
		},
		{
			name:     "test 3",
			podName:  "testPod",
			nodeName: "node3",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey: false,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			cache: &upToDateCache{},
		},
		{
			name:     "test 4",
			podName:  "testPod",
			nodeName: "node4",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    456,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey:    false,
			expectedInvalidEquivalenceHash: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
			cache: &upToDateCache{},
		},
	}

	for _, test := range tests {
		ecache := NewEquivalenceCache()
		node := schedulercache.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}})
		// set cached item to equivalence cache
		ecache.updateResult(
			test.podName,
			test.predicateKey,
			test.cachedItem.fit,
			test.cachedItem.reasons,
			test.equivalenceHashForUpdatePredicate,
			test.cache,
			node,
		)
		// if we want to do invalid, invalid the cached item
		if test.expectedInvalidPredicateKey {
			predicateKeys := sets.NewString()
			predicateKeys.Insert(test.predicateKey)
			ecache.InvalidateCachedPredicateItem(test.nodeName, predicateKeys)
		}
		// calculate predicate with equivalence cache
		fit, reasons, invalid := ecache.lookupResult(test.podName,
			test.nodeName,
			test.predicateKey,
			test.equivalenceHashForCalPredicate,
		)
		// returned invalid should match expectedInvalidPredicateKey or expectedInvalidEquivalenceHash
		if test.equivalenceHashForUpdatePredicate != test.equivalenceHashForCalPredicate {
			if invalid != test.expectedInvalidEquivalenceHash {
				t.Errorf("Failed: %s, expected invalid: %v, but got: %v",
					test.name, test.expectedInvalidEquivalenceHash, invalid)
			}
		} else {
			if invalid != test.expectedInvalidPredicateKey {
				t.Errorf("Failed: %s, expected invalid: %v, but got: %v",
					test.name, test.expectedInvalidPredicateKey, invalid)
			}
		}
		// returned predicate result should match expected predicate item
		if fit != test.expectedPredicateItem.fit {
			t.Errorf("Failed: %s, expected fit: %v, but got: %v", test.name, test.cachedItem.fit, fit)
		}
		if !slicesEqual(reasons, test.expectedPredicateItem.reasons) {
			t.Errorf("Failed: %s, expected reasons: %v, but got: %v",
				test.name, test.expectedPredicateItem.reasons, reasons)
		}
	}
}

func TestGetEquivalenceHash(t *testing.T) {

	ecache := NewEquivalenceCache()

	pod1 := makeBasicPod("pod1")
	pod2 := makeBasicPod("pod2")

	pod3 := makeBasicPod("pod3")
	pod3.Spec.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: "someEBSVol111",
				},
			},
		},
	}

	pod4 := makeBasicPod("pod4")
	pod4.Spec.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: "someEBSVol222",
				},
			},
		},
	}

	pod5 := makeBasicPod("pod5")
	pod5.Spec.Volumes = []v1.Volume{}

	pod6 := makeBasicPod("pod6")
	pod6.Spec.Volumes = nil

	pod7 := makeBasicPod("pod7")
	pod7.Spec.NodeSelector = nil

	pod8 := makeBasicPod("pod8")
	pod8.Spec.NodeSelector = make(map[string]string)

	type podInfo struct {
		pod         *v1.Pod
		hashIsValid bool
	}

	tests := []struct {
		name         string
		podInfoList  []podInfo
		isEquivalent bool
	}{
		{
			name: "pods with everything the same except name",
			podInfoList: []podInfo{
				{pod: pod1, hashIsValid: true},
				{pod: pod2, hashIsValid: true},
			},
			isEquivalent: true,
		},
		{
			name: "pods that only differ in their PVC volume sources",
			podInfoList: []podInfo{
				{pod: pod3, hashIsValid: true},
				{pod: pod4, hashIsValid: true},
			},
			isEquivalent: false,
		},
		{
			name: "pods that have no volumes, but one uses nil and one uses an empty slice",
			podInfoList: []podInfo{
				{pod: pod5, hashIsValid: true},
				{pod: pod6, hashIsValid: true},
			},
			isEquivalent: true,
		},
		{
			name: "pods that have no NodeSelector, but one uses nil and one uses an empty map",
			podInfoList: []podInfo{
				{pod: pod7, hashIsValid: true},
				{pod: pod8, hashIsValid: true},
			},
			isEquivalent: true,
		},
	}

	var (
		targetPodInfo podInfo
		targetHash    uint64
	)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for i, podInfo := range test.podInfoList {
				testPod := podInfo.pod
				eclassInfo := ecache.getEquivalenceClassInfo(testPod)
				if eclassInfo == nil && podInfo.hashIsValid {
					t.Errorf("Failed: pod %v is expected to have valid hash", testPod)
				}

				if eclassInfo != nil {
					// NOTE(harry): the first element will be used as target so
					// this logic can't verify more than two inequivalent pods
					if i == 0 {
						targetHash = eclassInfo.hash
						targetPodInfo = podInfo
					} else {
						if targetHash != eclassInfo.hash {
							if test.isEquivalent {
								t.Errorf("Failed: pod: %v is expected to be equivalent to: %v", testPod, targetPodInfo.pod)
							}
						}
					}
				}
			}
		})
	}
}

func TestInvalidateCachedPredicateItemOfAllNodes(t *testing.T) {
	testPredicate := "GeneralPredicates"
	// tests is used to initialize all nodes
	tests := []struct {
		podName                           string
		nodeName                          string
		equivalenceHashForUpdatePredicate uint64
		cachedItem                        predicateItemType
		cache                             schedulercache.Cache
	}{
		{
			podName:  "testPod",
			nodeName: "node1",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: false,
				reasons: []algorithm.PredicateFailureReason{
					predicates.ErrPodNotFitsHostPorts,
				},
			},
			cache: &upToDateCache{},
		},
		{
			podName:  "testPod",
			nodeName: "node2",
			equivalenceHashForUpdatePredicate: 456,
			cachedItem: predicateItemType{
				fit: false,
				reasons: []algorithm.PredicateFailureReason{
					predicates.ErrPodNotFitsHostPorts,
				},
			},
			cache: &upToDateCache{},
		},
		{
			podName:  "testPod",
			nodeName: "node3",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: true,
			},
			cache: &upToDateCache{},
		},
	}
	ecache := NewEquivalenceCache()

	for _, test := range tests {
		node := schedulercache.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}})
		// set cached item to equivalence cache
		ecache.updateResult(
			test.podName,
			testPredicate,
			test.cachedItem.fit,
			test.cachedItem.reasons,
			test.equivalenceHashForUpdatePredicate,
			test.cache,
			node,
		)
	}

	// invalidate cached predicate for all nodes
	ecache.InvalidateCachedPredicateItemOfAllNodes(sets.NewString(testPredicate))

	// there should be no cached predicate any more
	for _, test := range tests {
		if algorithmCache, exist := ecache.algorithmCache[test.nodeName]; exist {
			if _, exist := algorithmCache[testPredicate]; exist {
				t.Errorf("Failed: cached item for predicate key: %v on node: %v should be invalidated",
					testPredicate, test.nodeName)
				break
			}
		}
	}
}

func TestInvalidateAllCachedPredicateItemOfNode(t *testing.T) {
	testPredicate := "GeneralPredicates"
	// tests is used to initialize all nodes
	tests := []struct {
		podName                           string
		nodeName                          string
		equivalenceHashForUpdatePredicate uint64
		cachedItem                        predicateItemType
		cache                             schedulercache.Cache
	}{
		{
			podName:  "testPod",
			nodeName: "node1",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			cache: &upToDateCache{},
		},
		{
			podName:  "testPod",
			nodeName: "node2",
			equivalenceHashForUpdatePredicate: 456,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			cache: &upToDateCache{},
		},
		{
			podName:  "testPod",
			nodeName: "node3",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: true,
			},
			cache: &upToDateCache{},
		},
	}
	ecache := NewEquivalenceCache()

	for _, test := range tests {
		node := schedulercache.NewNodeInfo()
		node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}})
		// set cached item to equivalence cache
		ecache.updateResult(
			test.podName,
			testPredicate,
			test.cachedItem.fit,
			test.cachedItem.reasons,
			test.equivalenceHashForUpdatePredicate,
			test.cache,
			node,
		)
	}

	for _, test := range tests {
		// invalidate cached predicate for all nodes
		ecache.InvalidateAllCachedPredicateItemOfNode(test.nodeName)
		if _, exist := ecache.algorithmCache[test.nodeName]; exist {
			t.Errorf("Failed: cached item for node: %v should be invalidated", test.nodeName)
			break
		}
	}
}

func BenchmarkEquivalenceHash(b *testing.B) {
	pod := makeBasicPod("test")
	for i := 0; i < b.N; i++ {
		getEquivalenceHash(pod)
	}
}

// syncingMockCache delegates method calls to an actual Cache,
// but calls to UpdateNodeNameToInfoMap synchronize with the test.
type syncingMockCache struct {
	schedulercache.Cache
	cycleStart, cacheInvalidated chan struct{}
	once                         sync.Once
}

// UpdateNodeNameToInfoMap delegates to the real implementation, but on the first call, it
// synchronizes with the test.
//
// Since UpdateNodeNameToInfoMap is one of the first steps of (*genericScheduler).Schedule, we use
// this point to signal to the test that a scheduling cycle has started.
func (c *syncingMockCache) UpdateNodeNameToInfoMap(infoMap map[string]*schedulercache.NodeInfo) error {
	err := c.Cache.UpdateNodeNameToInfoMap(infoMap)
	c.once.Do(func() {
		c.cycleStart <- struct{}{}
		<-c.cacheInvalidated
	})
	return err
}

// TestEquivalenceCacheInvalidationRace tests that equivalence cache invalidation is correctly
// handled when an invalidation event happens early in a scheduling cycle. Specifically, the event
// occurs after schedulercache is snapshotted and before equivalence cache lock is acquired.
func TestEquivalenceCacheInvalidationRace(t *testing.T) {
	// Create a predicate that returns false the first time and true on subsequent calls.
	podWillFit := false
	var callCount int
	testPredicate := func(pod *v1.Pod,
		meta algorithm.PredicateMetadata,
		nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
		callCount++
		if !podWillFit {
			podWillFit = true
			return false, []algorithm.PredicateFailureReason{predicates.ErrFakePredicate}, nil
		}
		return true, nil, nil
	}

	// Set up the mock cache.
	cache := schedulercache.New(time.Duration(0), wait.NeverStop)
	cache.AddNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1"}})
	mockCache := &syncingMockCache{
		Cache:            cache,
		cycleStart:       make(chan struct{}),
		cacheInvalidated: make(chan struct{}),
	}

	eCache := NewEquivalenceCache()
	// Ensure that equivalence cache invalidation happens after the scheduling cycle starts, but before
	// the equivalence cache would be updated.
	go func() {
		<-mockCache.cycleStart
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "new-pod", UID: "new-pod"},
			Spec:       v1.PodSpec{NodeName: "machine1"}}
		if err := cache.AddPod(pod); err != nil {
			t.Errorf("Could not add pod to cache: %v", err)
		}
		eCache.InvalidateAllCachedPredicateItemOfNode("machine1")
		mockCache.cacheInvalidated <- struct{}{}
	}()

	// Set up the scheduler.
	ps := map[string]algorithm.FitPredicate{"testPredicate": testPredicate}
	predicates.SetPredicatesOrdering([]string{"testPredicate"})
	prioritizers := []algorithm.PriorityConfig{{Map: EqualPriorityMap, Weight: 1}}
	pvcLister := schedulertesting.FakePersistentVolumeClaimLister([]*v1.PersistentVolumeClaim{})
	scheduler := NewGenericScheduler(
		mockCache,
		eCache,
		NewSchedulingQueue(),
		ps,
		algorithm.EmptyPredicateMetadataProducer,
		prioritizers,
		algorithm.EmptyPriorityMetadataProducer,
		nil, nil, pvcLister, true, false)

	// First scheduling attempt should fail.
	nodeLister := schedulertesting.FakeNodeLister(makeNodeList([]string{"machine1"}))
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}}
	machine, err := scheduler.Schedule(pod, nodeLister)
	if machine != "" || err == nil {
		t.Error("First scheduling attempt did not fail")
	}

	// Second scheduling attempt should succeed because cache was invalidated.
	_, err = scheduler.Schedule(pod, nodeLister)
	if err != nil {
		t.Errorf("Second scheduling attempt failed: %v", err)
	}
	if callCount != 2 {
		t.Errorf("Predicate should have been called twice. Was called %d times.", callCount)
	}
}
