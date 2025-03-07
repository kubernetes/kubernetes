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

import "k8s.io/apimachinery/pkg/util/sets"

const (
	opaqueTypeTagName = "k8s:opaqueType"
)

type opaqueTypeTagValidator struct{}

func init() {
	RegisterTagValidator(opaqueTypeTagValidator{})
}

func (opaqueTypeTagValidator) Init(Config) {}

func (opaqueTypeTagValidator) TagName() string {
	return opaqueTypeTagName
}

func (opaqueTypeTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeAny)
}

func (opaqueTypeTagValidator) GetValidations(context Context, _ []string, _ string) (Validations, error) {
	return Validations{OpaqueType: true}, nil
}

func (opaqueTypeTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:    opaqueTypeTagName,
		Scopes: []Scope{ScopeField},
		Description: "Indicates that any validations declared on the referenced type will be ignored. " +
			"If a referenced type's package is not included in the generator's current " +
			"flags, this tag must be set, or code generation will fail (preventing silent " +
			"mistakes). If the validations should not be ignored, add the type's package " +
			"to the generator using the --extra-pkg flag.",
	}
	return doc
}
