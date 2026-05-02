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

package validators

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	immutableTagName = "k8s:immutable"
)

func init() {
	RegisterTagValidator(immutableTagValidator{})
}

type immutableTagValidator struct{}

func (immutableTagValidator) Init(_ Config) {}

func (immutableTagValidator) TagName() string {
	return immutableTagName
}

var immutableTagValidScopes = sets.New(ScopeField, ScopeType, ScopeMapVal, ScopeListVal)

func (immutableTagValidator) ValidScopes() sets.Set[Scope] {
	return immutableTagValidScopes
}

var (
	immutableValidator = types.Name{Package: libValidationPkg, Name: "Immutable"}
)

// UpdateCohort is the name of the cohort used for update-related validations
// (immutable, update constraints). This cohort runs independently from the
// default cohort, ensuring that update-related errors are always reported even
// when the default cohort short-circuits due to other failures (e.g. required
// or maxItems).
//
// NOTE: There should be exactly two cohorts: the default cohort (for structural
// validity checks like required/maxItems) and this update cohort (for
// update-constraint checks like immutable). These two represent orthogonal
// concerns: "is this value structurally valid?" vs "was this value illegally
// modified?". Adding more cohorts would make the short-circuit semantics harder
// to reason about and should be avoided.
const UpdateCohort = "update"

func (immutableTagValidator) GetValidations(context Context, _ codetags.Tag) (Validations, error) {
	var result Validations

	// Use ShortCircuit flag and put in the "update" cohort so that immutable
	// checks run independently from the default cohort. This ensures that if a
	// field has both +k8s:immutable and +k8s:required/+k8s:maxItems, the
	// immutable error is always reported on updates even when the default cohort
	// short-circuits.
	fn := Function(immutableTagName, ShortCircuit, immutableValidator)
	fn.Cohort = UpdateCohort
	result.AddFunction(fn)
	return result, nil
}

func (itv immutableTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            itv.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(itv.ValidScopes()),
		Description:    "Indicates that a field may not be updated.",
	}
}
