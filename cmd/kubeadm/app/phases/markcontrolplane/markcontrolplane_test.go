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

package markcontrolplane

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestMarkControlPlane(t *testing.T) {
	// Note: this test takes advantage of the deterministic marshalling of
	// JSON provided by strategicpatch so that "expectedPatch" can use a
	// string equality test instead of a logical JSON equality test. That
	// will need to change if strategicpatch's behavior changes in the
	// future.
	tests := []struct {
		name           string
		existingLabel  string
		existingTaints []v1.Taint
		newTaints      []v1.Taint
		expectedPatch  string
	}{
		{
			name:           "control-plane label and taint missing",
			existingLabel:  "",
			existingTaints: nil,
			newTaints:      []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			expectedPatch:  `{"metadata":{"labels":{"node-role.kubernetes.io/master":""}},"spec":{"taints":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master"}]}}`,
		},
		{
			name:           "control-plane label and taint missing but taint not wanted",
			existingLabel:  "",
			existingTaints: nil,
			newTaints:      nil,
			expectedPatch:  `{"metadata":{"labels":{"node-role.kubernetes.io/master":""}}}`,
		},
		{
			name:           "control-plane label missing",
			existingLabel:  "",
			existingTaints: []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			newTaints:      []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			expectedPatch:  `{"metadata":{"labels":{"node-role.kubernetes.io/master":""}}}`,
		},
		{
			name:           "control-plane taint missing",
			existingLabel:  kubeadmconstants.LabelNodeRoleMaster,
			existingTaints: nil,
			newTaints:      []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			expectedPatch:  `{"spec":{"taints":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master"}]}}`,
		},
		{
			name:           "nothing missing",
			existingLabel:  kubeadmconstants.LabelNodeRoleMaster,
			existingTaints: []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			newTaints:      []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			expectedPatch:  `{}`,
		},
		{
			name:          "has taint and no new taints wanted",
			existingLabel: kubeadmconstants.LabelNodeRoleMaster,
			existingTaints: []v1.Taint{
				{
					Key:    "node.cloudprovider.kubernetes.io/uninitialized",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			newTaints:     nil,
			expectedPatch: `{}`,
		},
		{
			name:          "has taint and should merge with wanted taint",
			existingLabel: kubeadmconstants.LabelNodeRoleMaster,
			existingTaints: []v1.Taint{
				{
					Key:    "node.cloudprovider.kubernetes.io/uninitialized",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			newTaints:     []v1.Taint{kubeadmconstants.ControlPlaneTaint},
			expectedPatch: `{"spec":{"taints":[{"effect":"NoSchedule","key":"node-role.kubernetes.io/master"},{"effect":"NoSchedule","key":"node.cloudprovider.kubernetes.io/uninitialized"}]}}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nodename := "node01"
			controlPlaneNode := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodename,
					Labels: map[string]string{
						v1.LabelHostname: nodename,
					},
				},
			}

			if tc.existingLabel != "" {
				controlPlaneNode.ObjectMeta.Labels[tc.existingLabel] = ""
			}

			if tc.existingTaints != nil {
				controlPlaneNode.Spec.Taints = tc.existingTaints
			}

			jsonNode, err := json.Marshal(controlPlaneNode)
			if err != nil {
				t.Fatalf("unexpected encoding error: %v", err)
			}

			var patchRequest string
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.Header().Set("Content-Type", "application/json")

				if req.URL.Path != "/api/v1/nodes/"+nodename {
					t.Errorf("request for unexpected HTTP resource: %v", req.URL.Path)
					http.Error(w, "", http.StatusNotFound)
					return
				}

				switch req.Method {
				case "GET":
				case "PATCH":
					buf := new(bytes.Buffer)
					buf.ReadFrom(req.Body)
					patchRequest = buf.String()
				default:
					t.Errorf("request for unexpected HTTP verb: %v", req.Method)
					http.Error(w, "", http.StatusNotFound)
					return
				}

				w.WriteHeader(http.StatusOK)
				w.Write(jsonNode)
			}))
			defer s.Close()

			cs, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
			if err != nil {
				t.Fatalf("unexpected error building clientset: %v", err)
			}

			if err := MarkControlPlane(cs, nodename, tc.newTaints); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tc.expectedPatch != patchRequest {
				t.Errorf("unexpected error: wanted patch %v, got %v", tc.expectedPatch, patchRequest)
			}
		})
	}
}
