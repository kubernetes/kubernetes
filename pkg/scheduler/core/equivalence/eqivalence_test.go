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

package equivalence

import (
	"errors"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
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
		expectFit, expectCacheHit, expectCacheWrite bool
		expectedReasons                             []algorithm.PredicateFailureReason
		expectedError                               string
	}{
		{
			name:             "pod fits/cache hit",
			pred:             mockPredicate{},
			expectFit:        true,
			expectCacheHit:   true,
			expectCacheWrite: false,
		},
		{
			name:             "pod fits/cache miss",
			pred:             mockPredicate{fit: true},
			expectFit:        true,
			expectCacheHit:   false,
			expectCacheWrite: true,
		},
		{
			name:             "pod doesn't fit/cache miss",
			pred:             mockPredicate{reasons: []algorithm.PredicateFailureReason{predicates.ErrFakePredicate}},
			expectFit:        false,
			expectCacheHit:   false,
			expectCacheWrite: true,
			expectedReasons:  []algorithm.PredicateFailureReason{predicates.ErrFakePredicate},
		},
		{
			name:             "pod doesn't fit/cache hit",
			pred:             mockPredicate{},
			expectFit:        false,
			expectCacheHit:   true,
			expectCacheWrite: false,
			expectedReasons:  []algorithm.PredicateFailureReason{predicates.ErrFakePredicate},
		},
		{
			name:             "predicate error",
			pred:             mockPredicate{err: errors.New("This is expected")},
			expectFit:        false,
			expectCacheHit:   false,
			expectCacheWrite: false,
			expectedError:    "This is expected",
		},
	}

	predicatesOrdering := []string{"testPredicate"}
	predicateID := 0
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := schedulercache.NewNodeInfo()
			testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "n1"}}
			node.SetNode(testNode)
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "p1"}}
			meta := algorithm.EmptyPredicateMetadataProducer(nil, nil)

			// Initialize and populate equivalence class cache.
			ecache := NewCache(predicatesOrdering)
			ecache.Snapshot()
			nodeCache, _ := ecache.GetNodeCache(testNode.Name)

			equivClass := NewClass(pod)
			if test.expectCacheHit {
				nodeCache.updateResult(pod.Name, "testPredicate", predicateID, test.expectFit, test.expectedReasons, equivClass.hash, node)
			}

			fit, reasons, err := nodeCache.RunPredicate(test.pred.predicate, "testPredicate", predicateID, pod, meta, node, equivClass)

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
			_, ok := nodeCache.lookupResult(pod.Name, node.Node().Name, "testPredicate", predicateID, equivClass.hash)
			if !ok && test.expectCacheWrite {
				t.Errorf("Cache write should happen")
			}
			if !test.expectCacheHit && test.expectCacheWrite && !ok {
				t.Errorf("Cache write should happen")
			}
			if !test.expectCacheHit && !test.expectCacheWrite && ok {
				t.Errorf("Cache write should not happen")
			}
		})
	}
}

func TestUpdateResult(t *testing.T) {
	predicatesOrdering := []string{"GeneralPredicates"}
	tests := []struct {
		name               string
		pod                string
		predicateKey       string
		predicateID        int
		nodeName           string
		fit                bool
		reasons            []algorithm.PredicateFailureReason
		equivalenceHash    uint64
		expectPredicateMap bool
		expectCacheItem    predicateResult
	}{
		{
			name:               "test 1",
			pod:                "testPod",
			predicateKey:       "GeneralPredicates",
			predicateID:        0,
			nodeName:           "node1",
			fit:                true,
			equivalenceHash:    123,
			expectPredicateMap: false,
			expectCacheItem: predicateResult{
				Fit: true,
			},
		},
		{
			name:               "test 2",
			pod:                "testPod",
			predicateKey:       "GeneralPredicates",
			predicateID:        0,
			nodeName:           "node2",
			fit:                false,
			equivalenceHash:    123,
			expectPredicateMap: true,
			expectCacheItem: predicateResult{
				Fit: false,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := schedulercache.NewNodeInfo()
			testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}
			node.SetNode(testNode)

			// Initialize and populate equivalence class cache.
			ecache := NewCache(predicatesOrdering)
			nodeCache, _ := ecache.GetNodeCache(testNode.Name)

			if test.expectPredicateMap {
				predicateItem := predicateResult{
					Fit: true,
				}
				nodeCache.cache[test.predicateID] =
					resultMap{
						test.equivalenceHash: predicateItem,
					}
			}

			nodeCache.updateResult(
				test.pod,
				test.predicateKey,
				test.predicateID,
				test.fit,
				test.reasons,
				test.equivalenceHash,
				node,
			)

			cachedMapItem := nodeCache.cache[test.predicateID]
			if cachedMapItem == nil {
				t.Errorf("can't find expected cache item: %v", test.expectCacheItem)
			} else {
				if !reflect.DeepEqual(cachedMapItem[test.equivalenceHash], test.expectCacheItem) {
					t.Errorf("expected cached item: %v, but got: %v",
						test.expectCacheItem, cachedMapItem[test.equivalenceHash])
				}
			}
		})
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
	predicatesOrdering := []string{"GeneralPredicates"}
	tests := []struct {
		name                              string
		podName                           string
		nodeName                          string
		predicateKey                      string
		predicateID                       int
		equivalenceHashForUpdatePredicate uint64
		equivalenceHashForCalPredicate    uint64
		cachedItem                        predicateItemType
		expectedPredicateKeyMiss          bool
		expectedEquivalenceHashMiss       bool
		expectedPredicateItem             predicateItemType
	}{
		{
			name:                              "test 1",
			podName:                           "testPod",
			nodeName:                          "node1",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			predicateID:                       0,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedPredicateKeyMiss: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
		{
			name:                              "test 2",
			podName:                           "testPod",
			nodeName:                          "node2",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			predicateID:                       0,
			cachedItem: predicateItemType{
				fit: true,
			},
			expectedPredicateKeyMiss: false,
			expectedPredicateItem: predicateItemType{
				fit:     true,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
		{
			name:                              "test 3",
			podName:                           "testPod",
			nodeName:                          "node3",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			predicateID:                       0,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedPredicateKeyMiss: false,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
		},
		{
			name:                              "test 4",
			podName:                           "testPod",
			nodeName:                          "node4",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    456,
			predicateKey:                      "GeneralPredicates",
			predicateID:                       0,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedPredicateKeyMiss:    false,
			expectedEquivalenceHashMiss: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}

			// Initialize and populate equivalence class cache.
			ecache := NewCache(predicatesOrdering)
			nodeCache, _ := ecache.GetNodeCache(testNode.Name)

			node := schedulercache.NewNodeInfo()
			node.SetNode(testNode)
			// set cached item to equivalence cache
			nodeCache.updateResult(
				test.podName,
				test.predicateKey,
				test.predicateID,
				test.cachedItem.fit,
				test.cachedItem.reasons,
				test.equivalenceHashForUpdatePredicate,
				node,
			)
			// if we want to do invalid, invalid the cached item
			if test.expectedPredicateKeyMiss {
				predicateKeys := sets.NewString()
				predicateKeys.Insert(test.predicateKey)
				ecache.InvalidatePredicatesOnNode(test.nodeName, predicateKeys)
			}
			// calculate predicate with equivalence cache
			result, ok := nodeCache.lookupResult(test.podName,
				test.nodeName,
				test.predicateKey,
				test.predicateID,
				test.equivalenceHashForCalPredicate,
			)
			fit, reasons := result.Fit, result.FailReasons
			// returned invalid should match expectedPredicateKeyMiss or expectedEquivalenceHashMiss
			if test.equivalenceHashForUpdatePredicate != test.equivalenceHashForCalPredicate {
				if ok && test.expectedEquivalenceHashMiss {
					t.Errorf("Failed: %s, expected (equivalence hash) cache miss", test.name)
				}
				if !ok && !test.expectedEquivalenceHashMiss {
					t.Errorf("Failed: %s, expected (equivalence hash) cache hit", test.name)
				}
			} else {
				if ok && test.expectedPredicateKeyMiss {
					t.Errorf("Failed: %s, expected (predicate key) cache miss", test.name)
				}
				if !ok && !test.expectedPredicateKeyMiss {
					t.Errorf("Failed: %s, expected (predicate key) cache hit", test.name)
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
		})
	}
}

func TestGetEquivalenceHash(t *testing.T) {
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
				eclassInfo := NewClass(testPod)
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
	testPredicateID := 0
	predicatesOrdering := []string{testPredicate}
	// tests is used to initialize all nodes
	tests := []struct {
		name                              string
		podName                           string
		nodeName                          string
		equivalenceHashForUpdatePredicate uint64
		cachedItem                        predicateItemType
	}{
		{
			name:                              "hash predicate 123 not fits host ports",
			podName:                           "testPod",
			nodeName:                          "node1",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: false,
				reasons: []algorithm.PredicateFailureReason{
					predicates.ErrPodNotFitsHostPorts,
				},
			},
		},
		{
			name:                              "hash predicate 456 not fits host ports",
			podName:                           "testPod",
			nodeName:                          "node2",
			equivalenceHashForUpdatePredicate: 456,
			cachedItem: predicateItemType{
				fit: false,
				reasons: []algorithm.PredicateFailureReason{
					predicates.ErrPodNotFitsHostPorts,
				},
			},
		},
		{
			name:                              "hash predicate 123 fits",
			podName:                           "testPod",
			nodeName:                          "node3",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: true,
			},
		},
	}
	ecache := NewCache(predicatesOrdering)

	for _, test := range tests {
		node := schedulercache.NewNodeInfo()
		testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}
		node.SetNode(testNode)

		nodeCache, _ := ecache.GetNodeCache(testNode.Name)
		// set cached item to equivalence cache
		nodeCache.updateResult(
			test.podName,
			testPredicate,
			testPredicateID,
			test.cachedItem.fit,
			test.cachedItem.reasons,
			test.equivalenceHashForUpdatePredicate,
			node,
		)
	}

	// invalidate cached predicate for all nodes
	ecache.InvalidatePredicates(sets.NewString(testPredicate))

	// there should be no cached predicate any more
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if nodeCache, exist := ecache.nodeToCache[test.nodeName]; exist {
				if cache := nodeCache.cache[testPredicateID]; cache != nil {
					t.Errorf("Failed: cached item for predicate key: %v on node: %v should be invalidated",
						testPredicate, test.nodeName)
				}
			}
		})
	}
}

func TestInvalidateAllCachedPredicateItemOfNode(t *testing.T) {
	testPredicate := "GeneralPredicates"
	testPredicateID := 0
	predicatesOrdering := []string{testPredicate}
	// tests is used to initialize all nodes
	tests := []struct {
		name                              string
		podName                           string
		nodeName                          string
		equivalenceHashForUpdatePredicate uint64
		cachedItem                        predicateItemType
	}{
		{
			name:                              "hash predicate 123 not fits host ports",
			podName:                           "testPod",
			nodeName:                          "node1",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
		},
		{
			name:                              "hash predicate 456 not fits host ports",
			podName:                           "testPod",
			nodeName:                          "node2",
			equivalenceHashForUpdatePredicate: 456,
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
		},
		{
			name:                              "hash predicate 123 fits host ports",
			podName:                           "testPod",
			nodeName:                          "node3",
			equivalenceHashForUpdatePredicate: 123,
			cachedItem: predicateItemType{
				fit: true,
			},
		},
	}
	ecache := NewCache(predicatesOrdering)

	for _, test := range tests {
		node := schedulercache.NewNodeInfo()
		testNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: test.nodeName}}
		node.SetNode(testNode)

		nodeCache, _ := ecache.GetNodeCache(testNode.Name)
		// set cached item to equivalence cache
		nodeCache.updateResult(
			test.podName,
			testPredicate,
			testPredicateID,
			test.cachedItem.fit,
			test.cachedItem.reasons,
			test.equivalenceHashForUpdatePredicate,
			node,
		)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			oldNodeCache, _ := ecache.GetNodeCache(test.nodeName)
			oldGeneration := oldNodeCache.generation
			// invalidate cached predicate for all nodes
			ecache.InvalidateAllPredicatesOnNode(test.nodeName)
			if n, _ := ecache.GetNodeCache(test.nodeName); oldGeneration == n.generation {
				t.Errorf("Failed: cached item for node: %v should be invalidated", test.nodeName)
			}
		})
	}
}

func BenchmarkEquivalenceHash(b *testing.B) {
	pod := makeBasicPod("test")
	for i := 0; i < b.N; i++ {
		getEquivalencePod(pod)
	}
}
