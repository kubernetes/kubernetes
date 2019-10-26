/*
Copyright 2019 The Kubernetes Authors.

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

package gvisor_reject

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	api "k8s.io/kubernetes/pkg/apis/core"

	"github.com/stretchr/testify/assert"
)

var (
	runtimeHandlerPath     = field.NewPath("metadata", "annotations").Key("runtime-handler.cri.kubernetes.io")
	unstrustedWorkloadPath = field.NewPath("metadata", "annotations").Key("io.kubernetes.cri.untrusted-workload")
	runtimeClassPath       = field.NewPath("spec", "runtimeClassName")
)

func stringPtr(p string) *string {
	return &p
}

func createPod(tolerations []api.Toleration, selectorTerms []api.NodeSelectorTerm) *api.Pod {
	return &api.Pod{
		Spec: api.PodSpec{
			RuntimeClassName: stringPtr(gvisorRuntimeClass),
			Tolerations:      tolerations,
			Affinity: &api.Affinity{
				NodeAffinity: &api.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
						NodeSelectorTerms: selectorTerms,
					},
				},
			},
		},
	}
}

func TestRequestsGvisor(t *testing.T) {
	for name, test := range map[string]struct {
		annotations  map[string]string
		runtimeClass *string
		expected     *field.Path
	}{
		"empty": {
			expected: nil,
		},
		"runtime-handler.cri.kubernetes.io": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "gvisor",
			},
			expected: runtimeHandlerPath,
		},
		"io.kubernetes.cri.untrusted-workload": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
			},
			expected: unstrustedWorkloadPath,
		},
		"both annotations": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
				"runtime-handler.cri.kubernetes.io":    "gvisor",
			},
			expected: runtimeHandlerPath,
		},
		"other runtime-handler.cri.kubernetes.io": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "foo",
			},
			expected: nil,
		},
		"other io.kubernetes.cri.untrusted-workload": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "false",
			},
			expected: nil,
		},
		"other for both annotations": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "false",
				"runtime-handler.cri.kubernetes.io":    "other",
			},
			expected: nil,
		},
		"RuntimeClass": {
			runtimeClass: stringPtr("gvisor"),
			expected:     runtimeClassPath,
		},
		"RuntimeClass empty": {
			runtimeClass: stringPtr(""),
			expected:     nil,
		},
		"RuntimeClass with annotation": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "gvisor",
			},
			runtimeClass: stringPtr("gvisor"),
			expected:     runtimeClassPath,
		},
	} {
		t.Run(name, func(t *testing.T) {
			pod := &api.Pod{}
			pod.Annotations = test.annotations
			pod.Spec.RuntimeClassName = test.runtimeClass

			actual := requestsGvisor(pod)
			if test.expected == nil {
				assert.Nil(t, actual)
			} else {
				assert.Equal(t, test.expected.String(), actual.String())
			}
		})
	}
}

func TestRejectGvisor(t *testing.T) {
	for name, test := range map[string]struct {
		annotations  map[string]string
		runtimeClass *string
		want         *field.Path
	}{
		"no annotation": {
			want: nil,
		},
		"runtime-handler.cri.kubernetes.io": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "foo",
			},
			want: runtimeHandlerPath,
		},
		"io.kubernetes.cri.untrusted-workload": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "false",
			},
			want: unstrustedWorkloadPath,
		},
		"both annotations": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "false",
				"runtime-handler.cri.kubernetes.io":    "foo",
			},
			want: runtimeHandlerPath,
		},
		"gvisor runtime-handler.cri.kubernetes.io": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "gvisor",
			},
			want: nil,
		},
		"gvisor io.kubernetes.cri.untrusted-workload": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
			},
			want: nil,
		},
		"gvisor both annotation": {
			annotations: map[string]string{
				"io.kubernetes.cri.untrusted-workload": "true",
				"runtime-handler.cri.kubernetes.io":    "gvisor",
			},
			want: nil,
		},
		"RuntimeClass": {
			runtimeClass: stringPtr("foo"),
			want:         runtimeClassPath,
		},
		"RuntimeClass empty": {
			runtimeClass: stringPtr(""),
			want:         nil,
		},
		"RuntimeClass with annotation": {
			annotations: map[string]string{
				"runtime-handler.cri.kubernetes.io": "foo",
			},
			runtimeClass: stringPtr("foo"),
			want:         runtimeClassPath,
		},
	} {
		t.Run(name, func(t *testing.T) {
			pod := &api.Pod{}
			pod.Annotations = test.annotations
			pod.Spec.RuntimeClassName = test.runtimeClass

			got := rejectsGvisor(pod)
			if test.want == nil {
				assert.Nil(t, got)
			} else {
				assert.Equal(t, test.want.String(), got.String())
			}
		})
	}
}

func TestValidateGvisorPod(t *testing.T) {
	otherToleration := api.Toleration{
		Key:      "foo",
		Operator: api.TolerationOpEqual,
		Value:    gvisorNodeValue,
		Effect:   api.TaintEffectNoSchedule,
	}
	anotherToleration := api.Toleration{
		Key:      gvisorNodeKey,
		Operator: api.TolerationOpEqual,
		Value:    "foo",
		Effect:   api.TaintEffectNoSchedule,
	}
	otherNodeSelector := api.NodeSelectorRequirement{
		Key:      "foo",
		Operator: api.NodeSelectorOpIn,
		Values:   []string{gvisorNodeValue},
	}
	anotherNodeSelector := api.NodeSelectorRequirement{
		Key:      gvisorNodeKey,
		Operator: api.NodeSelectorOpIn,
		Values:   []string{"foo"},
	}

	for name, test := range map[string]struct {
		pod  *api.Pod
		want string
	}{
		"simple": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "",
		},
		"tolerations": {
			pod: createPod(
				[]api.Toleration{otherToleration, gvisorToleration, anotherToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "",
		},
		"terms": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "",
		},
		"selectors": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector, gvisorNodeSelector, anotherNodeSelector},
					},
				}),
			want: "",
		},
		"terms+selectors": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector, gvisorNodeSelector},
					},
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector, anotherNodeSelector},
					},
				}),
			want: "",
		},
		"no tolerations": {
			pod: createPod(
				nil,
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "gVisor pod has invalid scheduling options",
		},
		"no terms": {
			pod:  createPod([]api.Toleration{gvisorToleration}, nil),
			want: "gVisor pod has invalid scheduling options",
		},
		"no selectors": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{api.NodeSelectorTerm{}}),
			want: "gVisor pod has invalid scheduling options",
		},
		"wrong toleration": {
			pod: createPod(
				[]api.Toleration{otherToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "gVisor pod has invalid scheduling options",
		},
		"wrong term": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector},
					},
				}),
			want: "gVisor pod has invalid scheduling options",
		},
		"wrong terms simple": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector},
					},
				}),
			want: "gVisor pod has invalid scheduling options",
		},
		"wrong terms+selectors": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector, anotherNodeSelector, gvisorNodeSelector},
					},
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
					{
						MatchExpressions: []api.NodeSelectorRequirement{otherNodeSelector, anotherNodeSelector},
					},
				}),
			want: "gVisor pod has invalid scheduling options",
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateGvisorPod(test.pod)
			if len(test.want) == 0 {
				assert.Nil(t, err)
			} else {
				if assert.Error(t, err) {
					assert.Contains(t, err.Error(), test.want)
				}
			}
		})
	}
}

func TestEndToEnd(t *testing.T) {
	for name, test := range map[string]struct {
		pod  *api.Pod
		want string
	}{
		"no gvisor": {
			pod:  &api.Pod{},
			want: "",
		},
		"gvisor": {
			pod: createPod(
				[]api.Toleration{gvisorToleration},
				[]api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{gvisorNodeSelector},
					},
				}),
			want: "",
		},
		"fail": {
			pod:  createPod([]api.Toleration{gvisorToleration}, nil),
			want: "gVisor pod has invalid scheduling options",
		},
		"conflict bad annotation": {
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"runtime-handler.cri.kubernetes.io": "foo",
					},
				},
				Spec: api.PodSpec{
					RuntimeClassName: stringPtr(gvisorRuntimeClass),
				},
			},
			want: "conflict",
		},
		"conflict bad runtime": {
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"runtime-handler.cri.kubernetes.io": "gvisor",
					},
				},
				Spec: api.PodSpec{
					RuntimeClassName: stringPtr("foo"),
				},
			},
			want: "conflict",
		},
	} {
		t.Run(name, func(t *testing.T) {
			attr := admission.NewAttributesRecord(
				test.pod,
				nil,
				api.Kind("Pod").WithVersion("version"),
				"namespace", // namespace
				"",          // name
				api.Resource("pods").WithVersion("version"),
				"", // subresource
				admission.Create,
				&metav1.CreateOptions{},
				false, // dryRun
				&user.DefaultInfo{})

			gv := gvisorReject{}
			err := gv.Validate(context.Background(), attr, nil)
			if len(test.want) == 0 {
				assert.Nil(t, err)
			} else {
				if assert.Error(t, err) {
					assert.Contains(t, err.Error(), test.want)
				}
			}
		})
	}
}
