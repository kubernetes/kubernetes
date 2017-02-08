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

package deployment

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller"
)

func TestScaleDownOldReplicaSets(t *testing.T) {
	tests := []struct {
		oldRSSizes []int
		d          *extensions.Deployment
	}{
		{
			oldRSSizes: []int{3},
			d:          newDeployment("foo", 3, nil, nil, nil, map[string]string{"foo": "bar"}),
		},
	}

	for i := range tests {
		t.Logf("running scenario %d", i)
		test := tests[i]

		var oldRSs []*extensions.ReplicaSet
		var expected []runtime.Object

		for n, size := range test.oldRSSizes {
			rs := newReplicaSet(test.d, fmt.Sprintf("%s-%d", test.d.Name, n), size)
			oldRSs = append(oldRSs, rs)

			objCopy, err := api.Scheme.Copy(rs)
			if err != nil {
				t.Errorf("unexpected error while deep-copying: %v", err)
				continue
			}
			rsCopy := objCopy.(*extensions.ReplicaSet)

			zero := int32(0)
			rsCopy.Spec.Replicas = &zero
			expected = append(expected, rsCopy)

			if *(oldRSs[n].Spec.Replicas) == *(expected[n].(*extensions.ReplicaSet).Spec.Replicas) {
				t.Errorf("broken test - original and expected RS have the same size")
			}
		}

		kc := fake.NewSimpleClientset(expected...)
		informers := informers.NewSharedInformerFactory(kc, controller.NoResyncPeriodFunc())
		c := NewDeploymentController(informers.Extensions().V1beta1().Deployments(), informers.Extensions().V1beta1().ReplicaSets(), informers.Core().V1().Pods(), kc)
		c.eventRecorder = &record.FakeRecorder{}

		c.scaleDownOldReplicaSetsForRecreate(oldRSs, test.d)
		for j := range oldRSs {
			rs := oldRSs[j]

			if *rs.Spec.Replicas != 0 {
				t.Errorf("rs %q has non-zero replicas", rs.Name)
			}
		}
	}
}
