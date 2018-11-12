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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/util/node"
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
			"control-plane label and taint missing",
			"",
			nil,
			[]v1.Taint{kubeadmconstants.MasterTaint},
			"{\"metadata\":{\"labels\":{\"node-role.kubernetes.io/master\":\"\"}},\"spec\":{\"taints\":[{\"effect\":\"NoSchedule\",\"key\":\"node-role.kubernetes.io/master\"}]}}",
		},
		{
			"control-plane label and taint missing but taint not wanted",
			"",
			nil,
			nil,
			"{\"metadata\":{\"labels\":{\"node-role.kubernetes.io/master\":\"\"}}}",
		},
		{
			"control-plane label missing",
			"",
			[]v1.Taint{kubeadmconstants.MasterTaint},
			[]v1.Taint{kubeadmconstants.MasterTaint},
			"{\"metadata\":{\"labels\":{\"node-role.kubernetes.io/master\":\"\"}}}",
		},
		{
			"control-plane taint missing",
			kubeadmconstants.LabelNodeRoleMaster,
			nil,
			[]v1.Taint{kubeadmconstants.MasterTaint},
			"{\"spec\":{\"taints\":[{\"effect\":\"NoSchedule\",\"key\":\"node-role.kubernetes.io/master\"}]}}",
		},
		{
			"nothing missing",
			kubeadmconstants.LabelNodeRoleMaster,
			[]v1.Taint{kubeadmconstants.MasterTaint},
			[]v1.Taint{kubeadmconstants.MasterTaint},
			"{}",
		},
		{
			"has taint and no new taints wanted",
			kubeadmconstants.LabelNodeRoleMaster,
			[]v1.Taint{
				{
					Key:    "node.cloudprovider.kubernetes.io/uninitialized",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			nil,
			"{}",
		},
		{
			"has taint and should merge with wanted taint",
			kubeadmconstants.LabelNodeRoleMaster,
			[]v1.Taint{
				{
					Key:    "node.cloudprovider.kubernetes.io/uninitialized",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
			[]v1.Taint{kubeadmconstants.MasterTaint},
			"{\"spec\":{\"taints\":[{\"effect\":\"NoSchedule\",\"key\":\"node-role.kubernetes.io/master\"},{\"effect\":\"NoSchedule\",\"key\":\"node.cloudprovider.kubernetes.io/uninitialized\"}]}}",
		},
	}

	for _, tc := range tests {
		hostname, err := node.GetHostname("")
		if err != nil {
			t.Fatalf("MarkControlPlane(%s): unexpected error: %v", tc.name, err)
		}
		controlPlaneNode := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: hostname,
				Labels: map[string]string{
					kubeletapis.LabelHostname: hostname,
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
			t.Fatalf("MarkControlPlane(%s): unexpected encoding error: %v", tc.name, err)
		}

		var patchRequest string
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Content-Type", "application/json")

			if req.URL.Path != "/api/v1/nodes/"+hostname {
				t.Errorf("MarkControlPlane(%s): request for unexpected HTTP resource: %v", tc.name, req.URL.Path)
				http.Error(w, "", http.StatusNotFound)
				return
			}

			switch req.Method {
			case "GET":
			case "PATCH":
				patchRequest = toString(req.Body)
			default:
				t.Errorf("MarkControlPlane(%s): request for unexpected HTTP verb: %v", tc.name, req.Method)
				http.Error(w, "", http.StatusNotFound)
				return
			}

			w.WriteHeader(http.StatusOK)
			w.Write(jsonNode)
		}))
		defer s.Close()

		cs, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
		if err != nil {
			t.Fatalf("MarkControlPlane(%s): unexpected error building clientset: %v", tc.name, err)
		}

		if err := MarkControlPlane(cs, hostname, tc.newTaints); err != nil {
			t.Errorf("MarkControlPlane(%s) returned unexpected error: %v", tc.name, err)
		}

		if tc.expectedPatch != patchRequest {
			t.Errorf("MarkControlPlane(%s) wanted patch %v, got %v", tc.name, tc.expectedPatch, patchRequest)
		}
	}
}

func toString(r io.Reader) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	return buf.String()
}
