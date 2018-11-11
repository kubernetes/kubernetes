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

package defaulttolerationseconds

import (
	"testing"

	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

func TestForgivenessAdmission(t *testing.T) {
	var defaultTolerationSeconds int64 = 300

	genTolerationSeconds := func(s int64) *int64 {
		return &s
	}

	handler := NewDefaultTolerationSeconds()
	// NOTE: for anyone who want to modify this test, the order of tolerations matters!
	tests := []struct {
		description  string
		requestedPod api.Pod
		expectedPod  api.Pod
	}{
		{
			description: "pod has no tolerations, expect add tolerations for `not-ready:NoExecute` and `unreachable:NoExecute`",
			requestedPod: api.Pod{
				Spec: api.PodSpec{},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod has alpha tolerations, expect add tolerations for `not-ready:NoExecute` and `unreachable:NoExecute`" +
				", the alpha tolerations will not be touched",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.DeprecatedTaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.DeprecatedTaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod has alpha not-ready toleration, expect add tolerations for `not-ready:NoExecute` and `unreachable:NoExecute`" +
				", the alpha tolerations will not be touched",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod has alpha unreachable toleration, expect add tolerations for `not-ready:NoExecute` and `unreachable:NoExecute`" +
				", the alpha tolerations will not be touched",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.DeprecatedTaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod has tolerations, but none is for taint `not-ready:NoExecute` or `unreachable:NoExecute`, expect add tolerations for `not-ready:NoExecute` and `unreachable:NoExecute`",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               "foo",
							Operator:          api.TolerationOpEqual,
							Value:             "bar",
							Effect:            api.TaintEffectNoSchedule,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               "foo",
							Operator:          api.TolerationOpEqual,
							Value:             "bar",
							Effect:            api.TaintEffectNoSchedule,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod specified a toleration for taint `not-ready:NoExecute`, expect add toleration for `unreachable:NoExecute`",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod specified a toleration for taint `unreachable:NoExecute`, expect add toleration for `not-ready:NoExecute`",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
					},
				},
			},
		},
		{
			description: "pod specified tolerations for both `not-ready:NoExecute` and `unreachable:NoExecute`, expect no change",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
		},
		{
			description: "pod specified toleration for taint `unreachable`, expect add toleration for `not-ready:NoExecute`",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Key:               schedulerapi.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               schedulerapi.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(300),
						},
					},
				},
			},
		},
		{
			description: "pod has wildcard toleration for all kind of taints, expect no change",
			requestedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{Operator: api.TolerationOpExists, TolerationSeconds: genTolerationSeconds(700)},
					},
				},
			},
			expectedPod: api.Pod{
				Spec: api.PodSpec{
					Tolerations: []api.Toleration{
						{
							Operator:          api.TolerationOpExists,
							TolerationSeconds: genTolerationSeconds(700),
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		err := handler.Admit(admission.NewAttributesRecord(&test.requestedPod, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", false, nil))
		if err != nil {
			t.Errorf("[%s]: unexpected error %v for pod %+v", test.description, err, test.requestedPod)
		}

		if !helper.Semantic.DeepEqual(test.expectedPod.Spec.Tolerations, test.requestedPod.Spec.Tolerations) {
			t.Errorf("[%s]: expected %#v got %#v", test.description, test.expectedPod.Spec.Tolerations, test.requestedPod.Spec.Tolerations)
		}
	}
}

func TestHandles(t *testing.T) {
	handler := NewDefaultTolerationSeconds()
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
