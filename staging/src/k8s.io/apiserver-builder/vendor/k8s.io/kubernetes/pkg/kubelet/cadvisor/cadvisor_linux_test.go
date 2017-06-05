// +build cgo,linux

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

package cadvisor

import (
	"reflect"
	"testing"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/metrics"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

func TestContainerLabels(t *testing.T) {
	container := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name:    "/docker/f81ad5335d390944e454ea19ab0924037d57337c19731524ad96eb26e74b6c6d",
			Aliases: []string{"k8s_POD.639b2af2_foo-web-315473031-e40e2_foobar_a369ace2-5fa9-11e6-b10f-c81f66e5e84d_851a97fd"},
		},
		Spec: info.ContainerSpec{
			Image: "qux/foo:latest",
			Labels: map[string]string{
				"io.kubernetes.container.hash":                   "639b2af2",
				types.KubernetesContainerNameLabel:               "POD",
				"io.kubernetes.container.restartCount":           "0",
				"io.kubernetes.container.terminationMessagePath": "",
				types.KubernetesPodNameLabel:                     "foo-web-315473031-e40e2",
				types.KubernetesPodNamespaceLabel:                "foobar",
				"io.kubernetes.pod.terminationGracePeriod":       "30",
				types.KubernetesPodUIDLabel:                      "a369ace2-5fa9-11e6-b10f-c81f66e5e84d",
			},
			Envs: map[string]string{
				"foo+env": "prod",
			},
		},
	}
	want := map[string]string{
		metrics.LabelID:    "/docker/f81ad5335d390944e454ea19ab0924037d57337c19731524ad96eb26e74b6c6d",
		metrics.LabelName:  "k8s_POD.639b2af2_foo-web-315473031-e40e2_foobar_a369ace2-5fa9-11e6-b10f-c81f66e5e84d_851a97fd",
		metrics.LabelImage: "qux/foo:latest",
		"namespace":        "foobar",
		"container_name":   "POD",
		"pod_name":         "foo-web-315473031-e40e2",
	}

	if have := containerLabels(container); !reflect.DeepEqual(want, have) {
		t.Errorf("want %v, have %v", want, have)
	}
}
