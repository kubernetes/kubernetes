/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/stretchr/testify/assert"
)

const (
	horizontalpodautoscalers = "horizontalpodautoscalers"
	k8s1                     = "k8s-1"
	k8s2                     = "k8s-2"
	clusters                 = "clusters"
)

func TestAutoscalerController(t *testing.T) {

	cluster1 := NewCluster(k8s1, apiv1.ConditionTrue)
	cluster2 := NewCluster(k8s2, apiv1.ConditionTrue)

	fakeFedClient := &fakefedclientset.Clientset{}
	RegisterFakeList(clusters, &fakeFedClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1}})
	RegisterFakeList(horizontalpodautoscalers, &fakeFedClient.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	fedhpaWatch := RegisterFakeWatch(horizontalpodautoscalers, &fakeFedClient.Fake)
	hpaUpdateChan := RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &fakeFedClient.Fake, fedhpaWatch)
	clusterWatch := RegisterFakeWatch(clusters, &fakeFedClient.Fake)

	cluster1Client := &fakekubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch(horizontalpodautoscalers, &cluster1Client.Fake)
	RegisterFakeList(horizontalpodautoscalers, &cluster1Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate(horizontalpodautoscalers, &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fakekubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch(horizontalpodautoscalers, &cluster2Client.Fake)
	RegisterFakeList(horizontalpodautoscalers, &cluster2Client.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate(horizontalpodautoscalers, &cluster2Client.Fake, cluster2Watch)
	cluster2UpdateChan := RegisterFakeCopyOnUpdate(horizontalpodautoscalers, &cluster2Client.Fake, cluster2Watch)

	fedInformerClientFactory := func(cluster *federationapi.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case k8s1:
			return cluster1Client, nil
		case k8s2:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}

	hpaController := newFederationSyncController(fakeFedClient, federatedtypes.NewHpaAdapter(fakeFedClient))
	hpaController.minimizeLatency()

	hpaFedinformer := ToFederatedInformerForTestOnly(hpaController.informer)
	hpaFedinformer.SetClientFactory(fedInformerClientFactory)

	stopChan := make(chan struct{})
	defer close(stopChan)
	hpaController.Run(stopChan)

	hpa := newHpaWithReplicas(newInt32(1), newInt32(70), 5)
	fedhpaWatch.Add(hpa)

	// 1: deletion finalizers are correctly added to hpa object
	updatedHpa := GetHpaFromChan(hpaUpdateChan)
	AssertHasFinalizer(t, updatedHpa, deletionhelper.FinalizerDeleteFromUnderlyingClusters)
	AssertHasFinalizer(t, updatedHpa, metav1.FinalizerOrphanDependents)

	// 2: the created hpa in federation gets created in registered cluster - cluster1
	hpa1 := GetHpaFromChan(cluster1CreateChan)
	assert.NotNil(t, hpa1)
	assert.Equal(t, hpa.Namespace, hpa1.Namespace)
	assert.Equal(t, hpa.Name, hpa1.Name)

	err := WaitForStoreUpdate(
		hpaController.informer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: hpa1.Namespace, Name: hpa1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "hpa from local cluster should have appeared in the informer store")

	// 3: Adding another cluster to federation gets an hpa in newly added cluster 2 also
	clusterWatch.Add(cluster2)
	hpa1 = GetHpaFromChan(cluster1UpdateChan)
	hpa2 := GetHpaFromChan(cluster2CreateChan)
	checkClusterConditions(t, hpa, hpa1, hpa2)

	// 4: update in the local hpa status gets reflected in federation object
	hpa1.Status.CurrentReplicas = hpa1.Spec.MaxReplicas - int32(3)
	hpa1.Status.DesiredReplicas = hpa1.Spec.MaxReplicas - int32(3)
	var lUtilisation int32
	lUtilisation = *hpa1.Spec.TargetCPUUtilizationPercentage - int32(20)
	hpa1.Status.CurrentCPUUtilizationPercentage = &lUtilisation
	cluster1Watch.Modify(hpa1)

	updatedHpa = GetHpaFromChan(hpaUpdateChan)
	assert.NotNil(t, updatedHpa)
	assert.Equal(t, hpa1.Status.CurrentReplicas, updatedHpa.Status.DesiredReplicas)
	assert.Equal(t, hpa1.Status.DesiredReplicas, updatedHpa.Status.CurrentReplicas)
	assert.Equal(t, *hpa1.Status.CurrentCPUUtilizationPercentage,
		*updatedHpa.Status.CurrentCPUUtilizationPercentage)

	// 5: Update of the federation object reflects in the local clusters
	// To update
	hpa.Spec.MinReplicas = newInt32(4)
	hpa.Spec.MaxReplicas = 10
	hpa.Spec.TargetCPUUtilizationPercentage = newInt32(60)
	fedhpaWatch.Modify(hpa)
	hpa1 = GetHpaFromChan(cluster1UpdateChan)
	hpa2 = GetHpaFromChan(cluster2UpdateChan)
	checkClusterConditions(t, hpa, hpa1, hpa2)

}

type replicas struct {
	min int32
	max int32
}

func TestHpaAdapter_ReconcileHook(t *testing.T) {
	defaultFedHpa := newHpaWithReplicas(newInt32(1), newInt32(70), 10)
	testCases := map[string]struct {
		fedHpa           *autoscalingv1.HorizontalPodAutoscaler
		localHpas        map[string]pkgruntime.Object
		expectedReplicas map[string]*replicas
	}{
		"Distribiutes replicas randomly if no existing hpa in any local cluster": {
			localHpas: func() map[string]pkgruntime.Object {
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = nil
				hpas["c2"] = nil
				return hpas
			}(),
		},
		"Cluster with no hpa gets replicas if other clusters have replicas": {
			localHpas: func() map[string]pkgruntime.Object {
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = newHpaWithReplicas(newInt32(1), newInt32(70), 10)
				hpas["c2"] = nil
				return hpas
			}(),
			expectedReplicas: map[string]*replicas{
				"c1": {
					min: int32(1),
					max: int32(9),
				},
				"c2": {
					min: int32(1),
					max: int32(1),
				},
			},
		},
		"Cluster needing max replicas gets it if there is another cluster to offer max": {
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(newInt32(1), newInt32(70), 7)
				hpa1 = updateHpaStatus(hpa1, newInt32(50), 5, 5, true)
				hpa2 := newHpaWithReplicas(newInt32(1), newInt32(70), 1)
				hpa2 = updateHpaStatus(hpa2, newInt32(70), 1, 1, true)
				// include third object to ensure, it does not break the test
				hpa3 := newHpaWithReplicas(newInt32(1), newInt32(70), 2)
				hpa3 = updateHpaStatus(hpa3, newInt32(70), 1, 1, false)
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = hpa1
				hpas["c2"] = hpa2
				hpas["c3"] = hpa3
				return hpas
			}(),
			expectedReplicas: map[string]*replicas{
				"c1": {
					min: int32(1),
					max: int32(6),
				},
				"c2": {
					min: int32(1),
					max: int32(2),
				},
				"c3": {
					min: int32(1),
					max: int32(2),
				},
			},
		},
		"Cluster needing max replicas does not get it if there is no cluster offerring max": {
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(newInt32(1), newInt32(70), 9)
				hpa1 = updateHpaStatus(hpa1, newInt32(70), 9, 9, false)
				hpa2 := newHpaWithReplicas(newInt32(1), newInt32(70), 1)
				hpa2 = updateHpaStatus(hpa2, newInt32(70), 1, 1, true)
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = hpa1
				hpas["c2"] = hpa2
				return hpas
			}(),
			expectedReplicas: map[string]*replicas{
				"c1": {
					min: int32(1),
					max: int32(9),
				},
				"c2": {
					min: int32(1),
					max: int32(1),
				},
			},
		},
		"Cluster which can increase min replicas gets to increase min if there is a cluster offering min": {
			fedHpa: newHpaWithReplicas(newInt32(4), newInt32(70), 10),
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(newInt32(3), newInt32(70), 6)
				hpa1 = updateHpaStatus(hpa1, newInt32(50), 3, 3, true)
				hpa2 := newHpaWithReplicas(newInt32(1), newInt32(70), 4)
				hpa2 = updateHpaStatus(hpa2, newInt32(50), 3, 3, true)
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = hpa1
				hpas["c2"] = hpa2
				return hpas
			}(),
			expectedReplicas: map[string]*replicas{
				"c1": {
					min: int32(2),
					max: int32(6),
				},
				"c2": {
					min: int32(2),
					max: int32(4),
				},
			},
		},
		"Cluster which can increase min replicas does not increase if there are no clusters offering min": {
			fedHpa: newHpaWithReplicas(newInt32(4), newInt32(70), 10),
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(newInt32(3), newInt32(70), 6)
				hpa1 = updateHpaStatus(hpa1, newInt32(50), 4, 4, true)
				hpa2 := newHpaWithReplicas(newInt32(1), newInt32(70), 4)
				hpa2 = updateHpaStatus(hpa2, newInt32(50), 3, 3, true)
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = hpa1
				hpas["c2"] = hpa2
				return hpas
			}(),
			expectedReplicas: map[string]*replicas{
				"c1": {
					min: int32(3),
					max: int32(6),
				},
				"c2": {
					min: int32(1),
					max: int32(4),
				},
			},
		},
		"Increasing replicas on fed object increases the same on clusters": {
			// original fed min max = 4, 10 increasing to below
			fedHpa: newHpaWithReplicas(newInt32(4), newInt32(70), 14),
			localHpas: func() map[string]pkgruntime.Object {
				// does not matter if scaleability is true
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = newHpaWithReplicas(newInt32(1), newInt32(70), 5)
				hpas["c2"] = newHpaWithReplicas(newInt32(1), newInt32(70), 5)
				return hpas
			}(),
			// We dont know which cluster gets how many, but the resultant total should match
		},
		"Decreasing replicas on fed object decreases the same on clusters": {
			// original fed min max = 4, 16 decreasing to below
			fedHpa: newHpaWithReplicas(newInt32(3), newInt32(70), 8),
			localHpas: func() map[string]pkgruntime.Object {
				// does not matter if scaleability is true
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = newHpaWithReplicas(newInt32(2), newInt32(70), 8)
				hpas["c2"] = newHpaWithReplicas(newInt32(2), newInt32(70), 8)
				return hpas
			}(),
			// We dont know which cluster gets how many, but the resultant total should match
		},
	}

	fakeFedClient := &fakefedclientset.Clientset{}
	//RegisterFakeList(clusters, &fakeFedClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1}})
	RegisterFakeList(horizontalpodautoscalers, &fakeFedClient.Fake, &autoscalingv1.HorizontalPodAutoscalerList{Items: []autoscalingv1.HorizontalPodAutoscaler{}})
	hpaAdapter := federatedtypes.NewHpaAdapter(fakeFedClient).(federatedtypes.ReconcilePlugin)

	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			var lhpas []*autoscalingv1.HorizontalPodAutoscaler
			if testCase.fedHpa == nil {
				testCase.fedHpa = defaultFedHpa
			}
			scheduledObjs, _ := hpaAdapter.ReconcileHook(testCase.fedHpa, testCase.localHpas)
			for _, obj := range scheduledObjs {
				lhpas = append(lhpas, obj.(*autoscalingv1.HorizontalPodAutoscaler))
			}
			checkClusterConditions(t, testCase.fedHpa, lhpas...)
			if testCase.expectedReplicas != nil {
				for cluster, replicas := range testCase.expectedReplicas {
					scheduledHpa := scheduledObjs[cluster].(*autoscalingv1.HorizontalPodAutoscaler)
					assert.Equal(t, replicas.min, *scheduledHpa.Spec.MinReplicas)
					assert.Equal(t, replicas.max, scheduledHpa.Spec.MaxReplicas)
				}
			}
		})
	}
}

func checkClusterConditions(t *testing.T, fedHpa *autoscalingv1.HorizontalPodAutoscaler, hpas ...*autoscalingv1.HorizontalPodAutoscaler) {
	minTotal := int32(0)
	maxTotal := int32(0)
	for _, hpa := range hpas {
		assert.NotNil(t, hpa)
		assert.Equal(t, fedHpa.Namespace, hpa.Namespace)
		assert.Equal(t, fedHpa.Name, hpa.Name)
		if hpa.Spec.MinReplicas != nil {
			minTotal += *hpa.Spec.MinReplicas
		}
		maxTotal += hpa.Spec.MaxReplicas
	}

	// - Total of max matches the fed max
	assert.Equal(t, fedHpa.Spec.MaxReplicas, maxTotal)
	// - Total of min is not less then fed min
	assert.Condition(t, func() bool {
		if *fedHpa.Spec.MinReplicas <= minTotal {
			return true
		}
		return false
	})
}

func GetHpaFromChan(c chan pkgruntime.Object) *autoscalingv1.HorizontalPodAutoscaler {
	if hpa := GetObjectFromChan(c); hpa == nil {
		return nil
	} else {
		return hpa.(*autoscalingv1.HorizontalPodAutoscaler)
	}
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}

func newHpaWithReplicas(min, targetUtilisation *int32, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myhpa",
			Namespace: apiv1.NamespaceDefault,
			SelfLink:  "/api/mylink",
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: "HorizontalPodAutoscaler",
				Name: "target-",
			},
			MinReplicas:                    min,
			MaxReplicas:                    max,
			TargetCPUUtilizationPercentage: targetUtilisation,
		},
	}
}

func updateHpaStatus(hpa *autoscalingv1.HorizontalPodAutoscaler, currentUtilisation *int32, current, desired int32, scaleable bool) *autoscalingv1.HorizontalPodAutoscaler {
	hpa.Status.CurrentReplicas = current
	hpa.Status.DesiredReplicas = desired
	hpa.Status.CurrentCPUUtilizationPercentage = currentUtilisation
	now := metav1.Now()
	scaledTime := now
	if scaleable {
		// definitely more then 5 minutes ago
		scaledTime = metav1.NewTime(now.Time.Add(-6 * time.Minute))
	}
	hpa.Status.LastScaleTime = &scaledTime
	return hpa
}
