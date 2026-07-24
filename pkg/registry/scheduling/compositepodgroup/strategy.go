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

package compositepodgroup

import (
	"context"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/apis/scheduling/validation"
	"k8s.io/kubernetes/pkg/features"
)

type compositePodGroupStrategy struct {
	rest.DeclarativeValidation
	names.NameGenerator
}

func NewStrategy() *compositePodGroupStrategy {
	return &compositePodGroupStrategy{
		rest.DeclarativeValidation{Scheme: legacyscheme.Scheme},
		names.SimpleNameGenerator,
	}
}

func (*compositePodGroupStrategy) NamespaceScoped() bool {
	return true
}

func (*compositePodGroupStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

func (*compositePodGroupStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	cpg := obj.(*scheduling.CompositePodGroup)
	// Status must not be set by user on create.
	cpg.Status = scheduling.CompositePodGroupStatus{}
	dropDisabledCompositePodGroupFields(cpg, nil)
}

func (*compositePodGroupStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	cpg := obj.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroup(cpg)
}

// DeclarativeValidationConfig declares the options referenced by this type's tags,
// mapped to whether each is enabled.
func (*compositePodGroupStrategy) DeclarativeValidationConfig(ctx context.Context, obj, oldObj runtime.Object) rest.DeclarativeValidationConfig {
	return rest.DeclarativeValidationConfig{Options: map[string]bool{
		string(features.CompositePodGroup):        utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup),
		string(features.PodGroupPreemptionPolicy): utilfeature.DefaultFeatureGate.Enabled(features.PodGroupPreemptionPolicy),
	}}
}

func (*compositePodGroupStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) Canonicalize(obj runtime.Object) {}

func (*compositePodGroupStrategy) AllowCreateOnUpdate(ctx context.Context) bool {
	return false
}

func (*compositePodGroupStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	newCPG.Status = oldCPG.Status
	dropDisabledCompositePodGroupFields(newCPG, oldCPG)
}

func (*compositePodGroupStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroupUpdate(newCPG, oldCPG)
}

func (*compositePodGroupStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*compositePodGroupStrategy) AllowUnconditionalUpdate(ctx context.Context) bool {
	return false
}

type compositePodGroupStatusStrategy struct {
	*compositePodGroupStrategy
}

func NewStatusStrategy(strategy *compositePodGroupStrategy) *compositePodGroupStatusStrategy {
	return &compositePodGroupStatusStrategy{strategy}
}

func (*compositePodGroupStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"scheduling.k8s.io/v1alpha3": fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("spec"),
		),
	}
	return fields
}

func (*compositePodGroupStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	newCPG.Spec = oldCPG.Spec
	metav1.ResetObjectMetaForStatus(&newCPG.ObjectMeta, &oldCPG.ObjectMeta)
}

func (r *compositePodGroupStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCPG := obj.(*scheduling.CompositePodGroup)
	oldCPG := old.(*scheduling.CompositePodGroup)
	return validation.ValidateCompositePodGroupStatusUpdate(newCPG, oldCPG)
}

func (*compositePodGroupStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

// dropDisabledCompositePodGroupFields removes fields which are covered by a feature gate.
func dropDisabledCompositePodGroupFields(newCPG, oldCPG *scheduling.CompositePodGroup) {
	var newCPGSpec, oldCPGSpec *scheduling.CompositePodGroupSpec
	if newCPG != nil {
		newCPGSpec = &newCPG.Spec
	}
	if oldCPG != nil {
		oldCPGSpec = &oldCPG.Spec
	}
	dropDisabledCompositePodGroupSpecFields(newCPGSpec, oldCPGSpec)
}

func dropDisabledCompositePodGroupSpecFields(newCPGSpec, oldCPGSpec *scheduling.CompositePodGroupSpec) {
	dropDisabledPreemptionPolicyField(newCPGSpec, oldCPGSpec)
}

// dropDisabledPreemptionPolicyField removes the PreemptionPolicy field unless it is
// already used in the old CompositePodGroup spec.
func dropDisabledPreemptionPolicyField(newCPGSpec, oldCPGSpec *scheduling.CompositePodGroupSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodGroupPreemptionPolicy) || preemptionPolicyInUse(oldCPGSpec) {
		// No need to drop anything.
		return
	}
	newCPGSpec.PreemptionPolicy = nil
}

func preemptionPolicyInUse(cpgSpec *scheduling.CompositePodGroupSpec) bool {
	return cpgSpec != nil && cpgSpec.PreemptionPolicy != nil
}
