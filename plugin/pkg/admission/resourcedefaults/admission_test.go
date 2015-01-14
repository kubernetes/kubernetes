/*
Copyright 2014 Google Inc. All rights reserved.

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

package resourcedefaults

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestAdmission(t *testing.T) {
	namespace := "default"

	handler := NewResourceDefaults()
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: "ns"},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}

	err := handler.Admit(admission.NewAttributesRecord(&pod, namespace, "pods", "CREATE"))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler")
	}

	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Memory.String() != "512Mi" {
			t.Errorf("Unexpected memory value %s", pod.Spec.Containers[i].Memory.String())
		}
		if pod.Spec.Containers[i].CPU.String() != "1" {
			t.Errorf("Unexpected cpu value %s", pod.Spec.Containers[i].CPU.String())
		}
	}
}
