/*
Copyright The Kubernetes Authors.

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

package ssh

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

func node(name string, labels map[string]string, addresses ...v1.NodeAddress) v1.Node {
	return v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
		Status:     v1.NodeStatus{Addresses: addresses},
	}
}

func TestInternalHost(t *testing.T) {
	nodes := &v1.NodeList{Items: []v1.Node{
		node("control-plane", nil,
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: "172.20.0.10"},
			v1.NodeAddress{Type: v1.NodeExternalIP, Address: "54.0.0.10"}),
		node("worker", nil,
			v1.NodeAddress{Type: v1.NodeExternalIP, Address: "35.0.0.20"},
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: "172.20.0.20"}),
		node("internal-only", nil,
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: "172.20.0.30"}),
	}}
	for host, want := range map[string]string{
		"35.0.0.20:22":       "172.20.0.20:22",
		"54.0.0.10:22":       "172.20.0.10:22",
		"172.20.0.30:22":     "172.20.0.30:22",
		"203.0.113.1:22":     "203.0.113.1:22",
		"api.example.com:22": "api.example.com:22",
		"35.0.0.20":          "35.0.0.20",
	} {
		if got := internalHost(nodes, host); got != want {
			t.Errorf("internalHost(%q) = %q, want %q", host, got, want)
		}
	}
}

func TestControlPlaneBastion(t *testing.T) {
	controlPlane := map[string]string{framework.ControlPlaneLabel: ""}
	nodes := &v1.NodeList{Items: []v1.Node{
		node("worker", nil,
			v1.NodeAddress{Type: v1.NodeExternalIP, Address: "35.0.0.20"}),
		node("control-plane", controlPlane,
			v1.NodeAddress{Type: v1.NodeInternalIP, Address: "172.20.0.10"},
			v1.NodeAddress{Type: v1.NodeExternalIP, Address: "54.0.0.10"}),
	}}
	got, err := controlPlaneBastion(nodes)
	if err != nil || got != "54.0.0.10:22" {
		t.Errorf("controlPlaneBastion() = %q, %v, want %q", got, err, "54.0.0.10:22")
	}

	nodes.Items[1].Status.Addresses = nodes.Items[1].Status.Addresses[:1]
	if got, err := controlPlaneBastion(nodes); err == nil {
		t.Errorf("controlPlaneBastion() = %q, want an error when no control-plane node has an external IP", got)
	}
}
