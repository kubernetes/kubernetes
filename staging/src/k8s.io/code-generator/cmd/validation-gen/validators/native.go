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
)

// declarativeValidationNative implements the TagValidator interface for the
// +k8s:declarativeValidationNative tag.
type declarativeValidationNative struct{}

func init() {
	RegisterTagValidator(&declarativeValidationNative{})
}

func (d *declarativeValidationNative) Init(cfg Config) {}

func (d *declarativeValidationNative) TagName() string {
	return "k8s:declarativeValidationNative"
}

func (d *declarativeValidationNative) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField, ScopeListVal)
}

func (d *declarativeValidationNative) LateTagValidator() {}

func (d *declarativeValidationNative) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// Mark union members as declarative if this tag is present.
	// This requires union processing to have run first, so we implement LateTagValidator.
	MarkUnionDeclarative(context.ParentPath.String(), context.Member)
	MarkZeroOrOneOfDeclarative(context.ParentPath.String(), context.Member)
	// This tag is a marker and does not generate any validations itself.
	return Validations{}, nil
}

func (d *declarativeValidationNative) Docs() TagDoc {
	return TagDoc{
		Tag:            d.TagName(),
		Description:    "Indicates that all validations for the field, including any on the field's type, are declarative and do not have a corresponding handwritten equivalent. This is only allowed for validations that are 'Stable'. When used, validation errors will be marked to show they originated from a declarative-only validation.",
		Scopes:         d.ValidScopes().UnsortedList(),
		StabilityLevel: TagStabilityLevelStable,
	}
}
