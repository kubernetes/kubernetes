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
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

func TestForgivenessAdmission(t *testing.T) {
	var defaultTolerationSeconds int64 = 300

	genTolerationSeconds := func(s int64) *int64 {
		return &s
	}

	plugin, err := newHandlerForTest()
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	handler := admissiontesting.WithReinvocationTesting(t, plugin)
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
							Key:               v1.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: &defaultTolerationSeconds,
						},
						{
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeNotReady,
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
							Key:               v1.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               v1.TaintNodeNotReady,
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
							Key:               v1.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeNotReady,
							Operator:          api.TolerationOpExists,
							Effect:            api.TaintEffectNoExecute,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeUnreachable,
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
							Key:               v1.TaintNodeUnreachable,
							Operator:          api.TolerationOpExists,
							TolerationSeconds: genTolerationSeconds(700),
						},
						{
							Key:               v1.TaintNodeNotReady,
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
		err := handler.Admit(context.TODO(), admission.NewAttributesRecord(&test.requestedPod, nil, api.Kind("Pod").WithVersion("version"), "foo", "name", api.Resource("pods").WithVersion("version"), "", "ignored", nil, false, nil), nil)
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

// newHandlerForTest returns a handler configured for testing.
func newHandlerForTest() (*Plugin, error) {
	handler := NewDefaultTolerationSeconds()
	pluginInitializer := initializer.New(nil, nil, nil, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	return handler, admission.ValidateInitialization(handler)
}
