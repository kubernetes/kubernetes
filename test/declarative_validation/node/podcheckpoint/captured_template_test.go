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

package podcheckpoint

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/node/podcheckpoint"
	podcases "k8s.io/kubernetes/test/declarative_validation/core/pod"
)

func TestDeclarativeCapturedTemplatePodSpec(t *testing.T) {
	path := field.NewPath("status", "checkpointedPodTemplate", "spec")
	responders := make([]core.EvictionResponder, 11)
	for i := range responders {
		responders[i] = core.EvictionResponder{Name: fmt.Sprintf("example.com/r%d", i), Priority: new(int32(1000))}
	}
	withResponders := func(values ...core.EvictionResponder) []podtest.Tweak {
		return []podtest.Tweak{func(p *core.Pod) { p.Spec.EvictionResponders = values }}
	}
	tests := append(podcases.InvalidPodSpecTestcases(path), []podcases.PodSpecTestcase{
		{ID: "responder count", Tweaks: withResponders(responders...), ExpectedErrs: field.ErrorList{field.TooMany(path.Child("evictionResponders"), 11, 10).WithOrigin("maxItems")}},
		{ID: "responder name required", Tweaks: withResponders(core.EvictionResponder{Priority: new(int32(1000))}), ExpectedErrs: field.ErrorList{field.Required(path.Child("evictionResponders").Index(0).Child("name"), "")}},
		{ID: "responder name format", Tweaks: withResponders(core.EvictionResponder{Name: "invalid", Priority: new(int32(1000))}), ExpectedErrs: field.ErrorList{field.Invalid(path.Child("evictionResponders").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-prefixed-label-key")}},
		{ID: "responder reserved name", Tweaks: withResponders(core.EvictionResponder{Name: "example.k8s.io/worker", Priority: new(int32(1000))}), ExpectedErrs: field.ErrorList{field.Invalid(path.Child("evictionResponders").Index(0).Child("name"), nil, "")}},
		{ID: "responder priority required", Tweaks: withResponders(core.EvictionResponder{Name: "example.com/worker"}), ExpectedErrs: field.ErrorList{field.Required(path.Child("evictionResponders").Index(0).Child("priority"), "")}},
		{ID: "responder priority minimum", Tweaks: withResponders(core.EvictionResponder{Name: "example.com/worker", Priority: new(int32(-1))}), ExpectedErrs: field.ErrorList{field.Invalid(path.Child("evictionResponders").Index(0).Child("priority"), nil, "").WithOrigin("minimum")}},
		{ID: "responder priority maximum", Tweaks: withResponders(core.EvictionResponder{Name: "example.com/worker", Priority: new(int32(100001))}), ExpectedErrs: field.ErrorList{field.Invalid(path.Child("evictionResponders").Index(0).Child("priority"), nil, "").WithOrigin("maximum")}},
		{ID: "responder reserved priority", Tweaks: withResponders(core.EvictionResponder{Name: "example.com/worker", Priority: new(int32(1))}), ExpectedErrs: field.ErrorList{field.Invalid(path.Child("evictionResponders").Index(0).Child("priority"), nil, "")}},
		{ID: "responder duplicate", Tweaks: withResponders(responders[0], responders[0]), ExpectedErrs: field.ErrorList{field.Duplicate(path.Child("evictionResponders").Index(1), nil)}},
		{ID: "invalid toleration key", Tweaks: []podtest.Tweak{podtest.SetTolerations(core.Toleration{Key: "invalid key", Operator: core.TolerationOpExists})}, ExpectedErrs: field.ErrorList{field.Invalid(path.Child("tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkBeta()}},
		{ID: "deadline minimum", Tweaks: []podtest.Tweak{func(p *core.Pod) { p.Spec.ActiveDeadlineSeconds = new(int64(0)) }}, ExpectedErrs: field.ErrorList{field.Invalid(path.Child("activeDeadlineSeconds"), nil, "").WithOrigin("minimum").MarkAlpha()}},
		{ID: "deadline maximum", Tweaks: []podtest.Tweak{func(p *core.Pod) { p.Spec.ActiveDeadlineSeconds = new(int64(1 << 31)) }}, ExpectedErrs: field.ErrorList{field.Invalid(path.Child("activeDeadlineSeconds"), nil, "").WithOrigin("maximum").MarkAlpha()}},
		{ID: "responders with scheduling group", Tweaks: []podtest.Tweak{func(p *core.Pod) {
			p.Spec.EvictionResponders = responders[:1]
			p.Spec.SchedulingGroup = &core.PodSchedulingGroup{PodGroupName: new("group")}
		}}, ExpectedErrs: field.ErrorList{field.Forbidden(path.Child("schedulingGroup"), "").WithOrigin("dependentForbidden").MarkBeta()}},
	}...)
	for _, apiVersion := range apiVersions {
		ctx := requestContext(apiVersion, "update", "status")
		for _, tc := range tests {
			t.Run(apiVersion+"/"+tc.ID, func(t *testing.T) {
				old := mkPodCheckpoint()
				old.ResourceVersion = "1"
				updated := old.DeepCopy()
				updated.Status.CheckpointedPodTemplate = &core.PodTemplateSpec{Spec: podtest.MakePodSpec(tc.Tweaks...)}
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updated, old, registry.StatusStrategy, tc.ExpectedErrs, apitesting.WithSubResources("status"))
			})
		}
	}
}
