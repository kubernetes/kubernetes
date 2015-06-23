/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package scdeny

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// ensures the SecurityContext is denied if it defines anything more than Caps or Privileged
func TestAdmission(t *testing.T) {
	handler := NewSecurityContextDeny(nil)

	var runAsUser int64 = 1
	priv := true
	successCases := map[string]*api.SecurityContext{
		"no sc":    nil,
		"empty sc": {},
		"valid sc": {Privileged: &priv, Capabilities: &api.Capabilities{}},
	}

	pod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{},
			},
		},
	}
	for k, v := range successCases {
		pod.Spec.Containers[0].SecurityContext = v
		err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", "foo", "name", string(api.ResourcePods), "", "ignored", nil))
		if err != nil {
			t.Errorf("Unexpected error returned from admission handler for case %s", k)
		}
	}

	errorCases := map[string]*api.SecurityContext{
		"run as user":     {RunAsUser: &runAsUser},
		"se linux optons": {SELinuxOptions: &api.SELinuxOptions{}},
		"mixed settings":  {Privileged: &priv, RunAsUser: &runAsUser, SELinuxOptions: &api.SELinuxOptions{}},
	}
	for k, v := range errorCases {
		pod.Spec.Containers[0].SecurityContext = v
		err := handler.Admit(admission.NewAttributesRecord(&pod, "Pod", "foo", "name", string(api.ResourcePods), "", "ignored", nil))
		if err == nil {
			t.Errorf("Expected error returned from admission handler for case %s", k)
		}
	}
}

func TestHandles(t *testing.T) {
	handler := NewSecurityContextDeny(nil)
	tests := map[admission.Operation]bool{
		admission.Update:  true,
		admission.Create:  true,
		admission.Delete:  false,
		admission.Connect: false,
	}
	for op, expected := range tests {
		result := handler.Handles(op)
		if result != expected {
			t.Errorf("Unexpected result for operation %s: %v\n", op, result)
		}
	}
}
