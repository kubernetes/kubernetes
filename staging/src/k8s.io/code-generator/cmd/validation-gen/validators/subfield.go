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

const (
	subfieldTagName = "k8s:subfield"
)

func init() {
	RegisterTagValidator(&subfieldTagValidator{})
}

type subfieldTagValidator struct {
	validator              TagValidationExtractor
	processedShortCircuits map[string]bool
}

func (stv *subfieldTagValidator) Init(cfg Config) {
	stv.validator = cfg.TagValidator
	stv.processedShortCircuits = make(map[string]bool)
}

func (*subfieldTagValidator) TagName() string {
	return subfieldTagName
}

var subfieldTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (*subfieldTagValidator) ValidScopes() sets.Set[Scope] {
	return subfieldTagValidScopes
}

var (
	validateSubfield = types.Name{Package: libValidationPkg, Name: "Subfield"}
)

func (stv *subfieldTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	args := tag.Args
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	t := context.Type
	nt := util.NonPointer(util.NativeType(t))
	if nt.Kind != types.Struct {
		return Validations{}, fmt.Errorf("can only be used on struct types: %v", nt.Kind)
	}
	subname := args[0].Value
	submemb := util.GetMemberByJSON(nt, subname)
	if submemb == nil {
		return Validations{}, fmt.Errorf("no field for json name %q", subname)
	}

	subContext := Context{
		Scope:          ScopeField,
		Type:           submemb.Type,
		Path:           context.Path.Child(subname),
		Member:         submemb,
		ParentPath:     context.Path,
		ParentType:     context.Type,
		StabilityLevel: context.StabilityLevel,
	}

	nilableStructType := context.Type
	if !util.IsNilableType(nilableStructType) {
		nilableStructType = types.PointerTo(nilableStructType)
	}

	nilableFieldType := submemb.Type
	fieldExprPrefix := ""
	if !util.IsNilableType(nilableFieldType) {
		nilableFieldType = types.PointerTo(nilableFieldType)
		fieldExprPrefix = "&"
	}

	// getFn is the function that retrieves the subfield value from the
	// struct.
	getFn := FunctionLiteral{
		Parameters: []ParamResult{{"o", nilableStructType}},
		Results:    []ParamResult{{"", nilableFieldType}},
	}
	getFn.Body = fmt.Sprintf("return %so.%s", fieldExprPrefix, submemb.Name)

	// equivArg compares the correlated elements in the old and new lists, for
	// ratcheting. directComparable selects "==" vs semantic DeepEqual.
	var equivArg any = Identifier(validateSemanticDeepEqual)
	if util.IsDirectComparable(util.NonPointer(util.NativeType(submemb.Type))) {
		// It must be a pointer, since other nilable types are not directly
		// comparable.
		equivArg = Identifier(validateDirectEqualPtr)
	}

	tagValidations, err := stv.validator.ExtractTagValidations(subContext, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	// Only the parent-member short-circuit inheritance depends on opaque-tag
	// globals (see globalOpaqueTypes / globalOpaqueMembers in opaque.go), so
	// defer just that decision until all opaque tags have been registered. The
	// rest of the wrapping is independent and runs eagerly.
	result := Validations{}
	result.AddDeferred(Deferred(ThisContext, func() (Validations, error) {
		// Subfield's own validations (which contain its short-circuits) are added first,
		// and inherited parent member short-circuits are added second. The cohort-based
		// partitioner (sortIntoCohorts) will then sort them so that all short-circuits
		// run first, preserving the relative addition order (subfield short-circuits
		// first, then inherited short-circuits). Finally, non-short-circuits will run.
		var combined Validations
		combined.Add(tagValidations)
		if !isFieldOpaque(context) {
			fieldValidations, err := stv.extractMemberShortCircuits(subContext)
			if err != nil {
				return Validations{}, err
			}
			combined.Add(fieldValidations)
		}

		mapped := WrapFunctions(combined, func(fn FunctionGen, scope DeferredScope) FunctionGen {
			// ParentContext functions (e.g. Union) emit without a cohort.
			if scope == ParentContext {
				return fn
			}
			f := Function(subfieldTagName, fn.Flags, validateSubfield, subname, getFn, equivArg,
				WrapperFunction{Function: fn, ObjType: submemb.Type, PathFragment: "." + subname})
			f.Cohort = subname
			return f
		})

		for i := range mapped.Deferred {
			// The validations belong to the subfield, so re-scope ParentContext
			// (which would target the enclosing struct) to ThisContext.
			if mapped.Deferred[i].Scope == ParentContext {
				mapped.Deferred[i].Scope = ThisContext
			}
		}

		return mapped, nil
	}))
	return result, nil
}

// extractMemberShortCircuits extracts short-circuit validations (e.g. required,
// immutable) from the child member's own comment lines so that the subfield
// validator can inherit them. Each subfield path is processed at most once;
// without this dedup, multiple subfield tags for the same child field would emit
// duplicate short-circuit validations.
func (stv *subfieldTagValidator) extractMemberShortCircuits(subContext Context) (Validations, error) {
	key := subContext.Path.String()
	if stv.processedShortCircuits[key] {
		return Validations{}, nil
	}
	stv.processedShortCircuits[key] = true

	ve, ok := stv.validator.(ValidationExtractor)
	if !ok {
		return Validations{}, fmt.Errorf("validator does not implement ValidationExtractor")
	}
	tags, err := ve.ExtractTags(subContext, subContext.Member.CommentLines)
	if err != nil {
		return Validations{}, err
	}
	memberValidations, err := ve.ExtractValidations(subContext, tags...)
	if err != nil {
		return Validations{}, err
	}
	return copyShortCircuitsAsNonError(memberValidations), nil
}

func (stv *subfieldTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            stv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(stv.ValidScopes()),
		Description:    "Declares a validation for a subfield of a struct.",
		Args: []TagArgDoc{{
			Description: "<field-json-name>",
			Type:        codetags.ArgTypeString,
			Required:    true,
		}},
		Docs: "The named subfield must be a direct field of the struct, or of an embedded struct.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for the subfield.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
}

// copyShortCircuitsAsNonError recursively traverses the validations (and deferred ones)
// and returns a new Validations object containing only the functions that have
// ShortCircuit flag set. The returned functions are also marked as NonError.
func copyShortCircuitsAsNonError(v Validations) Validations {
	res := Validations{}
	for _, fn := range v.Functions {
		if fn.Flags.IsSet(ShortCircuit) {
			fn.Flags |= NonError
			res.AddFunction(fn)
		}
	}
	for _, d := range v.Deferred {
		res.AddDeferred(Deferred(d.Scope, func() (Validations, error) {
			inner, err := d.Callback()
			if err != nil {
				return Validations{}, err
			}
			return copyShortCircuitsAsNonError(inner), nil
		}))
	}
	return res
}
