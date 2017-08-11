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

package federatedtypes

import (
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"

	"github.com/stretchr/testify/assert"
)

type replicas struct {
	min int32
	max int32
}

func TestGetHpaScheduleState(t *testing.T) {
	defaultFedHpa := newHpaWithReplicas(NewInt32(1), NewInt32(70), 10)
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
				hpas["c1"] = newHpaWithReplicas(NewInt32(1), NewInt32(70), 10)
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
				hpa1 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 7)
				hpa1 = updateHpaStatus(hpa1, NewInt32(50), 5, 5, true)
				hpa2 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 1)
				hpa2 = updateHpaStatus(hpa2, NewInt32(70), 1, 1, true)
				// include third object to ensure, it does not break the test
				hpa3 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 2)
				hpa3 = updateHpaStatus(hpa3, NewInt32(70), 1, 1, false)
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
				hpa1 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 9)
				hpa1 = updateHpaStatus(hpa1, NewInt32(70), 9, 9, false)
				hpa2 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 1)
				hpa2 = updateHpaStatus(hpa2, NewInt32(70), 1, 1, true)
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
			fedHpa: newHpaWithReplicas(NewInt32(4), NewInt32(70), 10),
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(NewInt32(3), NewInt32(70), 6)
				hpa1 = updateHpaStatus(hpa1, NewInt32(50), 3, 3, true)
				hpa2 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 4)
				hpa2 = updateHpaStatus(hpa2, NewInt32(50), 3, 3, true)
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
			fedHpa: newHpaWithReplicas(NewInt32(4), NewInt32(70), 10),
			localHpas: func() map[string]pkgruntime.Object {
				hpa1 := newHpaWithReplicas(NewInt32(3), NewInt32(70), 6)
				hpa1 = updateHpaStatus(hpa1, NewInt32(50), 4, 4, true)
				hpa2 := newHpaWithReplicas(NewInt32(1), NewInt32(70), 4)
				hpa2 = updateHpaStatus(hpa2, NewInt32(50), 3, 3, true)
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
			// Existing total of local min, max = 1+1, 5+5 decreasing to below
			fedHpa: newHpaWithReplicas(NewInt32(4), NewInt32(70), 14),
			localHpas: func() map[string]pkgruntime.Object {
				// does not matter if scaleability is true
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = newHpaWithReplicas(NewInt32(1), NewInt32(70), 5)
				hpas["c2"] = newHpaWithReplicas(NewInt32(1), NewInt32(70), 5)
				return hpas
			}(),
			// We dont know which cluster gets how many, but the resultant total should match
		},
		"Decreasing replicas on fed object decreases the same on clusters": {
			// Existing total of local min, max = 2+2, 8+8 decreasing to below
			fedHpa: newHpaWithReplicas(NewInt32(3), NewInt32(70), 8),
			localHpas: func() map[string]pkgruntime.Object {
				// does not matter if scaleability is true
				hpas := make(map[string]pkgruntime.Object)
				hpas["c1"] = newHpaWithReplicas(NewInt32(2), NewInt32(70), 8)
				hpas["c2"] = newHpaWithReplicas(NewInt32(2), NewInt32(70), 8)
				return hpas
			}(),
			// We dont know which cluster gets how many, but the resultant total should match
		},
	}

	adapter := &HpaAdapter{
		scaleForbiddenWindow: ScaleForbiddenWindow,
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			if testCase.fedHpa == nil {
				testCase.fedHpa = defaultFedHpa
			}
			scheduledState := adapter.getHpaScheduleState(testCase.fedHpa, testCase.localHpas)
			checkClusterConditions(t, testCase.fedHpa, scheduledState)
			if testCase.expectedReplicas != nil {
				for cluster, replicas := range testCase.expectedReplicas {
					scheduledReplicas := scheduledState[cluster]
					assert.Equal(t, replicas.min, scheduledReplicas.min)
					assert.Equal(t, replicas.max, scheduledReplicas.max)
				}
			}
		})
	}
}

func updateHpaStatus(hpa *autoscalingv1.HorizontalPodAutoscaler, currentUtilisation *int32, current, desired int32, scaleable bool) *autoscalingv1.HorizontalPodAutoscaler {
	hpa.Status.CurrentReplicas = current
	hpa.Status.DesiredReplicas = desired
	hpa.Status.CurrentCPUUtilizationPercentage = currentUtilisation
	now := metav1.Now()
	scaledTime := now
	if scaleable {
		// definitely more then ScaleForbiddenWindow time ago
		scaledTime = metav1.NewTime(now.Time.Add(-2 * ScaleForbiddenWindow))
	}
	hpa.Status.LastScaleTime = &scaledTime
	return hpa
}

func checkClusterConditions(t *testing.T, fedHpa *autoscalingv1.HorizontalPodAutoscaler, scheduled map[string]*replicaNums) {
	minTotal := int32(0)
	maxTotal := int32(0)
	for _, replicas := range scheduled {
		minTotal += replicas.min
		maxTotal += replicas.max
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
