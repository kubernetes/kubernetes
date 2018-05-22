/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubectl"
)

func TestProtocolsForObject(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: []api.ContainerPort{
						{
							ContainerPort: 101,
							Protocol:      api.ProtocolTCP,
						},
						{
							ContainerPort: 102,
							Protocol:      api.ProtocolUDP,
						},
					},
				},
			},
		},
	}

	expected := sets.NewString("101/TCP", "102/UDP")
	protocolsMap, err := protocolsForObject(pod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	protocolsString := kubectl.MakeProtocols(protocolsMap)
	protocolsStrings := strings.Split(protocolsString, ",")
	got := sets.NewString(protocolsStrings...)

	if !expected.Equal(got) {
		t.Fatalf("Protocols mismatch! Expected %v, got %v", expected, got)
	}
}
