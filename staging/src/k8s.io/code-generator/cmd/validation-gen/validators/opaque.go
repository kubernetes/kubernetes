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
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
)

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
	return sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)
}

func (opaqueTypeTagValidator) GetValidations(context Context, _ SchemaMetadata, _ codetags.Tag) (EmittedGroup, error) {
	return EmittedGroup{Validations: Validations{OpaqueType: true}}, nil
}

// Other validators (currently subfield) consult opaque tags to
// decide whether to inherit short-circuit validations from the field.
// isFieldOpaque reports whether the field or its type was tagged +k8s:opaqueType.
func isFieldOpaque(context Context) bool {
	if context.Member != nil && len(context.Member.CommentLines) > 0 {
		if hasOpaqueTag(context.Member.CommentLines) {
			return true
		}
	}
	if context.Type != nil {
		if len(context.Type.CommentLines) > 0 && hasOpaqueTag(context.Type.CommentLines) {
			return true
		}
		t := util.NonPointer(util.NativeType(context.Type))
		if t != nil && len(t.CommentLines) > 0 && hasOpaqueTag(t.CommentLines) {
			return true
		}
	}
	return false
}

func hasOpaqueTag(commentLines []string) bool {
	extracted := codetags.Extract("+", commentLines)
	for _, lines := range extracted {
		parsedTags, err := codetags.ParseAll(lines)
		if err != nil {
			continue
		}
		for _, tagItem := range parsedTags {
			for t := &tagItem; t != nil; t = t.ValueTag {
				if t.Name == opaqueTypeTagName {
					return true
				}
			}
		}
	}
	return false
}

func (v opaqueTypeTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            opaqueTypeTagName,
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(v.ValidScopes()),
		Description: "Indicates that any validations declared on the referenced type will be ignored. " +
			"If a referenced type's package is not included in the generator's current " +
			"flags, this tag must be set, or code generation will fail (preventing silent " +
			"mistakes). If the validations should not be ignored, add the type's package " +
			"to the generator using the --readonly-pkg flag.",
	}
	return doc
}
