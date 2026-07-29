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
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

func init() {
	RegisterTagValidator(&monotonicTagValidator{})
}

const (
	monotonicTagName = "k8s:monotonic"
)

var (
	monotonicValidator = types.Name{Package: libValidationPkg, Name: "Monotonic"}
)

type monotonicTagValidator struct{}

func (v *monotonicTagValidator) Init(_ Config) {}

func (v *monotonicTagValidator) TagName() string {
	return monotonicTagName
}

func (v *monotonicTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeType, ScopeField)
}

func (v *monotonicTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	t := util.NonPointer(util.NativeType(context.Type))
	if !types.IsInteger(t) {
		return Validations{}, fmt.Errorf("must be an integer type (got %s)", rootTypeString(context.Type, t))
	}

	result := Validations{}
	result.AddFunction(Function(monotonicTagName, DefaultFlags, monotonicValidator))
	return result, nil
}

func (v *monotonicTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            monotonicTagName,
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(v.ValidScopes()),
		Description:    "ensures that a field's value never decreases on update",
	}
}
