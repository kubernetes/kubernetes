/*
Copyright 2025 The Kubernetes Authors.

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

package devicetaintrule

import (
	"context"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

// deviceTaintRuleStrategy implements behavior for DeviceTaintRule objects
type deviceTaintRuleStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

var (
	Strategy       = &deviceTaintRuleStrategy{rest.DeclarativeValidation{Scheme: legacyscheme.Scheme}, names.SimpleNameGenerator}
	StatusStrategy = &deviceTaintRuleStatusStrategy{deviceTaintRuleStrategy: Strategy}
)

func (deviceTaintRuleStrategy) NamespaceScoped() bool {
	return false
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a new DeviceTaintRule that is the
// status.
func (*deviceTaintRuleStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"resource.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

func (*deviceTaintRuleStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	rule := obj.(*resource.DeviceTaintRule)
	// Status must not be set by user on create.
	rule.Status = resource.DeviceTaintRuleStatus{}
	rule.Generation = 1
}

func (*deviceTaintRuleStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	rule := obj.(*resource.DeviceTaintRule)
	return validation.ValidateDeviceTaintRule(rule)
}

func (*deviceTaintRuleStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*deviceTaintRuleStrategy) Canonicalize(obj runtime.Object) {
}

func (*deviceTaintRuleStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*deviceTaintRuleStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	rule := obj.(*resource.DeviceTaintRule)
	oldRule := old.(*resource.DeviceTaintRule)
	rule.Status = oldRule.Status

	// Automatically bump the TimeAdded when the effect changes and the
	// client hasn't already changed it. This makes the TimeAdded track
	// "when was this *effect* added" instead of "when was this
	// *DeviceTaintRule* added". This is relevant for computing the
	// toleration duration of affected claims.
	//
	// The downside is that clients cannot keep the old time even if that
	// is what they want - this seems extremely unlikely compared to "I
	// want TimeAdded to work automatically".
	if rule.Spec.Taint.Effect != oldRule.Spec.Taint.Effect &&
		rule.Spec.Taint.TimeAdded.Equal(oldRule.Spec.Taint.TimeAdded) {
		// Encoding only has seconds as resolution, so don't even
		// generate something which does not survive encode/decode (can cause
		// random unit test failures).
		rule.Spec.Taint.TimeAdded = &metav1.Time{Time: time.Now().Truncate(time.Second)}
	}

	// Any changes to the spec increment the generation number.
	if !apiequality.Semantic.DeepEqual(oldRule.Spec, rule.Spec) {
		rule.Generation = oldRule.Generation + 1
	}
}

func (*deviceTaintRuleStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateDeviceTaintRuleUpdate(obj.(*resource.DeviceTaintRule), old.(*resource.DeviceTaintRule))
}

func (*deviceTaintRuleStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*deviceTaintRuleStrategy) AllowUnconditionalUpdate() bool {
	return true
}

type deviceTaintRuleStatusStrategy struct {
	*deviceTaintRuleStrategy
}

// GetResetFields returns the set of fields that get reset by the strategy and
// should not be modified by the user. For a status update that is the spec.
func (*deviceTaintRuleStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"resource.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
		"resource.k8s.io/v1beta2": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
}

func (*deviceTaintRuleStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newRule := obj.(*resource.DeviceTaintRule)
	oldRule := old.(*resource.DeviceTaintRule)
	newRule.Spec = oldRule.Spec
	metav1.ResetObjectMetaForStatus(&newRule.ObjectMeta, &oldRule.ObjectMeta)
}

func (r *deviceTaintRuleStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newRule := obj.(*resource.DeviceTaintRule)
	oldRule := old.(*resource.DeviceTaintRule)
	return validation.ValidateDeviceTaintRuleStatusUpdate(newRule, oldRule)
}

// WarningsOnUpdate returns warnings for the given update.
func (*deviceTaintRuleStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
