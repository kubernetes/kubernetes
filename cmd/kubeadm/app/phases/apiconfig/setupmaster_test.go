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

package apiconfig

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	apiv1 "k8s.io/client-go/pkg/api/v1"
	restclient "k8s.io/client-go/rest"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/util/node"
)

const masterLabel = kubeadmconstants.LabelNodeRoleMaster

var masterTaint = &apiv1.Taint{Key: kubeadmconstants.LabelNodeRoleMaster, Value: "", Effect: "NoSchedule"}

func TestUpdateMasterRoleLabelsAndTaints(t *testing.T) {
	// Note: this test takes advantage of the deterministic marshalling of
	// JSON provided by strategicpatch so that "expectedPatch" can use a
	// string equality test instead of a logical JSON equality test. That
	// will need to change if strategicpatch's behavior changes in the
	// future.
	tests := []struct {
		name          string
		existingLabel string
		existingTaint *apiv1.Taint
		expectedPatch string
	}{
		{
			"master label and taint missing",
			"",
			nil,
			"{\"metadata\":{\"labels\":{\"node-role.kubernetes.io/master\":\"\"}},\"spec\":{\"taints\":[{\"effect\":\"NoSchedule\",\"key\":\"node-role.kubernetes.io/master\",\"timeAdded\":null}]}}",
		},
		{
			"master label missing",
			"",
			masterTaint,
			"{\"metadata\":{\"labels\":{\"node-role.kubernetes.io/master\":\"\"}}}",
		},
		{
			"master taint missing",
			masterLabel,
			nil,
			"{\"spec\":{\"taints\":[{\"effect\":\"NoSchedule\",\"key\":\"node-role.kubernetes.io/master\",\"timeAdded\":null}]}}",
		},
		{
			"nothing missing",
			masterLabel,
			masterTaint,
			"{}",
		},
	}

	for _, tc := range tests {
		hostname := node.GetHostname("")
		masterNode := &apiv1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: hostname,
				Labels: map[string]string{
					kubeletapis.LabelHostname: hostname,
				},
			},
		}

		if tc.existingLabel != "" {
			masterNode.ObjectMeta.Labels[tc.existingLabel] = ""
		}

		if tc.existingTaint != nil {
			masterNode.Spec.Taints = append(masterNode.Spec.Taints, *tc.existingTaint)
		}

		jsonNode, err := json.Marshal(masterNode)
		if err != nil {
			t.Fatalf("UpdateMasterRoleLabelsAndTaints(%s): unexpected encoding error: %v", tc.name, err)
		}

		var patchRequest string
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Content-Type", "application/json")

			if req.URL.Path != "/api/v1/nodes/"+hostname {
				t.Errorf("UpdateMasterRoleLabelsAndTaints(%s): request for unexpected HTTP resource: %v", tc.name, req.URL.Path)
				w.WriteHeader(http.StatusNotFound)
				return
			}

			switch req.Method {
			case "GET":
			case "PATCH":
				patchRequest = toString(req.Body)
			default:
				t.Errorf("UpdateMasterRoleLabelsAndTaints(%s): request for unexpected HTTP verb: %v", tc.name, req.Method)
				w.WriteHeader(http.StatusNotFound)
				return
			}

			w.WriteHeader(http.StatusOK)
			w.Write(jsonNode)
		}))
		defer s.Close()

		cs, err := clientsetFromTestServer(s)
		if err != nil {
			t.Fatalf("UpdateMasterRoleLabelsAndTaints(%s): unexpected error building clientset: %v", tc.name, err)
		}

		err = UpdateMasterRoleLabelsAndTaints(cs)
		if err != nil {
			t.Errorf("UpdateMasterRoleLabelsAndTaints(%s) returned unexpected error: %v", tc.name, err)
		}

		if tc.expectedPatch != patchRequest {
			t.Errorf("UpdateMasterRoleLabelsAndTaints(%s) wanted patch %v, got %v", tc.name, tc.expectedPatch, patchRequest)
		}
	}
}

func clientsetFromTestServer(s *httptest.Server) (*clientset.Clientset, error) {
	rc := &restclient.Config{Host: s.URL}
	c, err := corev1.NewForConfig(rc)
	if err != nil {
		return nil, err
	}
	return &clientset.Clientset{CoreV1Client: c}, nil
}

func toString(r io.Reader) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	return buf.String()
}
