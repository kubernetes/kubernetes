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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestScaleDownOldReplicaSets(t *testing.T) {
	tests := []struct {
		name string

		oldRSs []int
		d      *extensions.Deployment
	}{
		{
			oldRSs: []int{3},
			d:      newDeployment("foo", 3, nil, nil, nil, map[string]string{"foo": "bar"}),
		},
	}

	for i := range tests {
		t.Logf("running scenario %d", i)
		test := tests[i]

		var oldRSs []*extensions.ReplicaSet
		var expected []runtime.Object

		for _, size := range test.oldRSs {
			rs := newReplicaSet(test.d, fmt.Sprintf("%s-%d", test.d.Name, i), size)
			oldRSs = append(oldRSs, rs)

			objCopy, _ := api.Scheme.Copy(rs)
			rsCopy := objCopy.(*extensions.ReplicaSet)
			rsCopy.Spec.Replicas = 0
			expected = append(expected, rsCopy)

			if oldRSs[i].Spec.Replicas == expected[i].(*extensions.ReplicaSet).Spec.Replicas {
				t.Errorf("broken test - original and expected RS have the same size")
			}
		}

		kc := fake.NewSimpleClientset(expected...)
		informers := informers.NewSharedInformerFactory(kc, controller.NoResyncPeriodFunc())
		c := NewDeploymentController(informers.Deployments(), informers.ReplicaSets(), informers.Pods(), kc)

		c.scaleDownOldReplicaSetsForRecreate(oldRSs, test.d)
		for j := range oldRSs {
			rs := oldRSs[j]

			if rs.Spec.Replicas != 0 {
				t.Errorf("rs %q has non-zero replicas", rs.Name)
			}
		}
	}
}
