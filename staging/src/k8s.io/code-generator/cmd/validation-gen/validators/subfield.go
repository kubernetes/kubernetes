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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	subfieldTagName = "k8s:subfield"
)

func init() {
	RegisterTagValidator(&subfieldTagValidator{})
	RegisterEmissionFinalizer(FinalizeSubfieldOrder, subfieldFinalizer{})
}

// subfieldFinalizer re-homes an EmittedGroup's validations at its TargetPath
// by nesting subfield wrappers. Subfield traversal is owned here, next to the
// tag which declares subfields.
type subfieldFinalizer struct{}

func (subfieldFinalizer) Name() string { return "subfield" }

func (subfieldFinalizer) Finalize(context Context, group EmittedGroup) (EmittedGroup, error) {
	if group.TargetPath == nil {
		return group, nil
	}
	wrapped, err := wrapWithSubfieldPath(group.Validations, group.TargetPath, context)
	if err != nil {
		return EmittedGroup{}, err
	}
	group.Validations = wrapped
	group.TargetPath = nil
	return group, nil
}

type subfieldTagValidator struct {
	extractor              Extractor
	processedShortCircuits map[string]bool
}

func (stv *subfieldTagValidator) Init(cfg Config) {
	stv.extractor = cfg.Extractor
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

func (stv *subfieldTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (Validations, error) {
	// TODO: Support nested subfields once the generated validation code can
	// prevent nil pointer dereferences by checking optionality or requiredness
	// for intermediate fields.
	for t := tag.ValueTag; t != nil; t = t.ValueTag {
		if t.Name == subfieldTagName {
			return Validations{}, fmt.Errorf("nested +%s tags are unsupported; apply validations directly to the nested type instead", subfieldTagName)
		}
	}

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
		ParentMember:   context.Member,
		StabilityLevel: context.StabilityLevel,
	}

	tagValidations, err := stv.extractor.ExtractTagValidations(subContext, metadata, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

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

	mapped := wrapWithSubfield(combined, subname, context.Type, submemb)
	return mapped, nil
}

// wrapWithSubfield wraps validations in a subfield validation step for a single member,
// ensuring similarity between how type-level and field-level validations process subfields.
func wrapWithSubfield(validations Validations, subname string, parentType *types.Type, submemb *types.Member) Validations {
	nilableStructType := parentType
	if !util.IsNilableType(nilableStructType) {
		nilableStructType = types.PointerTo(nilableStructType)
	}
	nilableFieldType := submemb.Type
	fieldExprPrefix := ""
	if !util.IsNilableType(nilableFieldType) {
		nilableFieldType = types.PointerTo(nilableFieldType)
		fieldExprPrefix = "&"
	}

	getFn := FunctionLiteral{
		Parameters: []ParamResult{{"o", nilableStructType}},
		Results:    []ParamResult{{"", nilableFieldType}},
		Body:       fmt.Sprintf("return %so.%s", fieldExprPrefix, submemb.Name),
	}

	// equivArg compares the correlated elements in the old and new lists, for
	// ratcheting. directComparable selects "==" vs semantic DeepEqual.
	// It must be a pointer, since other nilable types are not directly
	// comparable.
	var equivArg any = Identifier(validateSemanticDeepEqual)
	if util.IsDirectComparable(util.NonPointer(util.NativeType(submemb.Type))) {
		equivArg = Identifier(validateDirectEqual)
	}

	for i, fn := range validations.Functions {
		f := Function(subfieldTagName, fn.Flags, validateSubfield, subname, getFn, equivArg,
			WrapperFunction{Function: fn, ObjType: submemb.Type, PathFragment: "." + subname})
		f.Cohort = subname
		validations.Functions[i] = f
	}
	for i, fn := range validations.TypeFunctions {
		f := Function(subfieldTagName, fn.Flags, validateSubfield, subname, getFn, equivArg,
			WrapperFunction{Function: fn, ObjType: submemb.Type, PathFragment: "." + subname})
		f.Cohort = subname
		validations.TypeFunctions[i] = f
	}
	return validations
}

// wrapWithSubfieldPath traces a dotted path and repeatedly wraps the provided validations
// for each segment using wrapWithSubfield. A nil target or a target equal to
// the context path means the validations apply at the current context and are
// returned unchanged; a target that cannot be resolved from the context is an
// error.
func wrapWithSubfieldPath(validations Validations, targetPath *field.Path, context Context) (Validations, error) {
	if targetPath == nil || context.Path == nil {
		return validations, nil
	}
	targetStr := targetPath.String()
	ctxStr := context.Path.String()
	if targetStr == ctxStr {
		return validations, nil
	}

	relStr := strings.TrimPrefix(targetStr, ctxStr+".")
	if relStr == targetStr { // Did not have the prefix
		return Validations{}, fmt.Errorf("target path %q is not a descendant of context path %q", targetStr, ctxStr)
	}

	subSegments := strings.Split(relStr, ".")
	currType := context.Type

	type segment struct {
		subname string
		parent  *types.Type
		submemb *types.Member
	}
	var segments []segment

	for _, seg := range subSegments {
		st := util.NonPointer(util.NativeType(currType))
		if st.Kind != types.Struct {
			return Validations{}, fmt.Errorf("cannot resolve %q in target path %q: %v is not a struct", seg, targetStr, st.Kind)
		}
		sm := util.GetMemberByJSON(st, seg)
		if sm == nil {
			return Validations{}, fmt.Errorf("cannot resolve %q in target path %q: no field for json name %q in %v", seg, targetStr, seg, st.Name)
		}
		segments = append(segments, segment{
			subname: seg,
			parent:  currType,
			submemb: sm,
		})
		currType = sm.Type
	}

	for i := len(segments) - 1; i >= 0; i-- {
		seg := segments[i]
		validations = wrapWithSubfield(validations, seg.subname, seg.parent, seg.submemb)
	}

	return validations, nil
}

// resolveSubfieldContext traverses a dotted subfield path from the given context,
// updating Scope, Type, Path, Member, ParentPath, and ParentType so that
// downstream validators operate directly on the target nested field without
// managing their own path-resolution loops.
func resolveSubfieldContext(context Context, subPath string) Context {
	if subPath == "" {
		return context
	}
	subSegments := strings.Split(subPath, ".")
	currCtxType := context.Type
	if context.Member != nil {
		currCtxType = context.Member.Type
	}
	currPath := context.Path
	var lastMemb *types.Member
	var lastParentType *types.Type
	var lastParentPath *field.Path
	lastParentMember := context.Member
	valid := true
	for i, seg := range subSegments {
		st := util.NonPointer(util.NativeType(currCtxType))
		if st.Kind != types.Struct && context.ParentType != nil && lastMemb == nil {
			st = util.NonPointer(util.NativeType(context.ParentType))
			currPath = context.ParentPath
		}
		if st.Kind != types.Struct {
			valid = false
			break
		}
		sm := util.GetMemberByJSON(st, seg)
		if sm == nil {
			valid = false
			break
		}
		if i > 0 {
			lastParentMember = lastMemb
		}
		lastMemb = sm
		lastParentType = currCtxType
		lastParentPath = currPath
		currCtxType = sm.Type
		if currPath != nil {
			currPath = currPath.Child(seg)
		}
	}
	if valid && lastMemb != nil {
		context = Context{
			Scope:          ScopeField,
			Type:           lastMemb.Type,
			Path:           currPath,
			Member:         lastMemb,
			ListSelector:   context.ListSelector,
			ParentPath:     lastParentPath,
			ParentType:     lastParentType,
			ParentMember:   lastParentMember,
			Conditions:     context.Conditions,
			StabilityLevel: context.StabilityLevel,
		}
	}
	return context
}

func (stv *subfieldTagValidator) Unwrap(context Context, tag codetags.Tag) (Context, error) {
	if arg, ok := tag.PositionalArg(); ok {
		context = resolveSubfieldContext(context, arg.Value)
	}
	return context, nil
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

	tags, err := stv.extractor.ExtractTags(subContext, subContext.Member.CommentLines)
	if err != nil {
		return Validations{}, err
	}
	memberMeta, err := stv.extractor.ExtractMetadata(subContext, tags...)
	if err != nil {
		return Validations{}, err
	}
	memberValidations, err := stv.extractor.ExtractValidations(subContext, memberMeta, tags...)
	if err != nil {
		return Validations{}, err
	}
	return copyShortCircuitsAsNonError(memberValidations), nil
}

func (stv *subfieldTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	if tag.ValueTag == nil {
		return SchemaMetadata{}, nil
	}
	subContext, err := stv.Unwrap(context, tag)
	if err != nil {
		return SchemaMetadata{}, err
	}
	return stv.extractor.ExtractMetadata(subContext, *tag.ValueTag)
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

// copyShortCircuitsAsNonError traverses the validations and returns a new
// Validations object containing only the functions that have ShortCircuit
// flag set. The returned functions are also marked as NonError.
func copyShortCircuitsAsNonError(v Validations) Validations {
	res := Validations{}
	for _, fn := range v.Functions {
		if fn.Flags.IsSet(ShortCircuit) {
			fn.Flags |= NonError
			res.AddFunction(fn)
		}
	}
	return res
}
