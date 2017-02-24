/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
)

// ensures the SecurityContext is denied if it defines anything more than Caps or Privileged
func TestAdmission(t *testing.T) {
	handler := NewSecurityContextDeny()

	var runAsUser int64 = 1
	priv := true

	cases := []struct {
		name        string
		sc          *api.SecurityContext
		podSc       *api.PodSecurityContext
		expectError bool
	}{
		{
			name: "unset",
		},
		{
			name: "empty container.SecurityContext",
			sc:   &api.SecurityContext{},
		},
		{
			name:  "empty pod.Spec.SecurityContext",
			podSc: &api.PodSecurityContext{},
		},
		{
			name: "valid container.SecurityContext",
			sc:   &api.SecurityContext{Privileged: &priv, Capabilities: &api.Capabilities{}},
		},
		{
			name:  "valid pod.Spec.SecurityContext",
			podSc: &api.PodSecurityContext{},
		},
		{
			name:        "container.SecurityContext.RunAsUser",
			sc:          &api.SecurityContext{RunAsUser: &runAsUser},
			expectError: true,
		},
		{
			name:        "container.SecurityContext.SELinuxOptions",
			sc:          &api.SecurityContext{SELinuxOptions: &api.SELinuxOptions{}},
			expectError: true,
		},
		{
			name:        "pod.Spec.SecurityContext.RunAsUser",
			podSc:       &api.PodSecurityContext{RunAsUser: &runAsUser},
			expectError: true,
		},
		{
			name:        "pod.Spec.SecurityContext.SELinuxOptions",
			podSc:       &api.PodSecurityContext{SELinuxOptions: &api.SELinuxOptions{}},
			expectError: true,
		},
	}

	for _, tc := range cases {
		p := pod()
		p.Spec.SecurityContext = tc.podSc
		p.Spec.Containers[0].SecurityContext = tc.sc

		err := handler.Admit(admission.NewAttributesRecord(p, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil))
		if err != nil && !tc.expectError {
			t.Errorf("%v: unexpected error: %v", tc.name, err)
		} else if err == nil && tc.expectError {
			t.Errorf("%v: expected error", tc.name)
		}

		// verify init containers are also checked
		p = pod()
		p.Spec.SecurityContext = tc.podSc
		p.Spec.Containers[0].SecurityContext = tc.sc
		p.Spec.InitContainers = p.Spec.Containers
		p.Spec.Containers = nil

		err = handler.Admit(admission.NewAttributesRecord(p, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil))
		if err != nil && !tc.expectError {
			t.Errorf("%v: unexpected error: %v", tc.name, err)
		} else if err == nil && tc.expectError {
			t.Errorf("%v: expected error", tc.name)
		}
	}
}

func TestPodSecurityContextAdmission(t *testing.T) {
	handler := NewSecurityContextDeny()
	pod := api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{},
			},
		},
	}

	fsGroup := int64(1001)

	tests := []struct {
		securityContext api.PodSecurityContext
		errorExpected   bool
	}{
		{
			securityContext: api.PodSecurityContext{},
			errorExpected:   false,
		},
		{
			securityContext: api.PodSecurityContext{
				SupplementalGroups: []int64{1234},
			},
			errorExpected: true,
		},
		{
			securityContext: api.PodSecurityContext{
				FSGroup: &fsGroup,
			},
			errorExpected: true,
		},
	}
	for _, test := range tests {
		pod.Spec.SecurityContext = &test.securityContext
		err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil))

		if test.errorExpected && err == nil {
			t.Errorf("Expected error for security context %+v but did not get an error", test.securityContext)
		}

		if !test.errorExpected && err != nil {
			t.Errorf("Unexpected error %v for security context %+v", err, test.securityContext)
		}
	}
}

func TestHandles(t *testing.T) {
	handler := NewSecurityContextDeny()
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

func pod() *api.Pod {
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{},
			},
		},
	}
}
