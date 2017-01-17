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

package priorities

import (
	"encoding/json"
	"testing"

	assert "github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/v1"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func TestPriorityMetadata(t *testing.T) {

	nonZeroReqs := &schedulercache.Resource{}

	nonZeroReqs.MilliCPU = priorityutil.DefaultMilliCpuRequest
	nonZeroReqs.Memory = priorityutil.DefaultMemoryRequest

	tolerations := []v1.Toleration{{
		Key:      "foo",
		Operator: v1.TolerationOpEqual,
		Value:    "bar",
		Effect:   v1.TaintEffectPreferNoSchedule,
	}}
	tolerationData, _ := json.Marshal(tolerations)
	podWithTolerations := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Annotations: map[string]string{
				v1.TolerationsAnnotationKey: string(tolerationData),
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
				},
			},
		},
	}

	tests := []struct {
		pod    *v1.Pod
		test   string
		pMdata interface{}
	}{
		{
			pod:    nil,
			pMdata: nil,
		},
		{
			pod: podWithTolerations,
			pMdata: &priorityMetadata{
				nonZeroRequest: nonZeroReqs,
				podTolerations: tolerations,
				affinity:       nil,
			},
		},
	}
	for _, test := range tests {
		ptData := PriorityMetadata(test.pod, nil)
		assert.Equal(t, ptData, test.pMdata)

	}

}
