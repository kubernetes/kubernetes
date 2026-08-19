/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

func TestGetStability(t *testing.T) {
	tests := []struct {
		tagName     string
		expected    TagStabilityLevel
		expectError bool
	}{
		{
			tagName:  "k8s:validateTrueAlpha",
			expected: TagStabilityLevelAlpha,
		},
		{
			tagName:  "k8s:validateTrueBeta",
			expected: TagStabilityLevelBeta,
		},
		{
			tagName:  "k8s:required",
			expected: TagStabilityLevelStable,
		},
		{
			tagName:     "k8s:unknownTag",
			expected:    "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.tagName, func(t *testing.T) {
			got, err := GetStability(tt.tagName)
			if err != nil && !tt.expectError {
				t.Errorf("Unexpected error: %v", err)
			}
			if got != tt.expected {
				t.Errorf("GetStability(%q) = %v, want %v", tt.tagName, got, tt.expected)
			}
		})
	}
}

func TestIsKnownTag(t *testing.T) {
	tests := []struct {
		tagName  string
		expected bool
	}{
		{
			tagName:  "k8s:validateTrueAlpha",
			expected: true,
		},
		{
			tagName:  "k8s:validateTrueBeta",
			expected: true,
		},
		{
			tagName:  "k8s:required",
			expected: true,
		},
		{
			tagName:  "k8s:unknownTag",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.tagName, func(t *testing.T) {
			got := IsKnownTag(tt.tagName)
			if got != tt.expected {
				t.Errorf("IsKnownTag(%q) = %v, want %v", tt.tagName, got, tt.expected)
			}
		})
	}
}

func TestCollectMetadata_Unions(t *testing.T) {
	structPath := field.NewPath("MyStruct")
	stringType := types.String

	ctx := Context{
		Scope: ScopeType,
		Path:  structPath,
		Type: &types.Type{
			Kind: types.Struct,
			Members: []types.Member{
				{
					Name:         "Type",
					Type:         stringType,
					CommentLines: []string{"+k8s:unionDiscriminator"},
					Tags:         `json:"type"`,
				},
				{
					Name:         "FieldA",
					Type:         stringType,
					CommentLines: []string{"+k8s:unionMember"},
					Tags:         `json:"fieldA,omitempty"`,
				},
			},
		},
	}

	meta, err := globalRegistry.CollectMetadata(ctx)
	if err != nil {
		t.Fatalf("unexpected error from CollectMetadata: %v", err)
	}

	var u *union
	for _, node := range meta.SortedNodes() {
		if len(node.Unions) > 0 {
			u = node.Unions[0].Payload
			break
		}
	}
	if u == nil {
		t.Fatalf("expected union for default name '', got %v", meta.Nodes)
	}

	if u.discriminator == nil || *u.discriminator != "type" {
		t.Errorf("expected discriminator 'type', got %v", u.discriminator)
	}

	if len(u.members) != 1 || u.members[0].fieldName != "fieldA" {
		t.Errorf("expected 1 member with fieldName 'fieldA', got %v", u.members)
	}
}

type testCollectorValidator struct {
	collected bool
	validated bool
	callOrder []string
}

func (tcv *testCollectorValidator) Init(_ Config) {}

func (tcv *testCollectorValidator) TagName() string {
	return "k8s:testCollector"
}

func (tcv *testCollectorValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField, ScopeType)
}

func (tcv *testCollectorValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	tcv.collected = true
	tcv.callOrder = append(tcv.callOrder, "CollectMetadata")
	return SchemaMetadata{}, nil
}

func (tcv *testCollectorValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (EmittedGroup, error) {
	tcv.validated = true
	tcv.callOrder = append(tcv.callOrder, "GetValidations")
	return EmittedGroup{}, nil
}

func (tcv *testCollectorValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            tcv.TagName(),
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(tcv.ValidScopes()),
	}
}

func TestExtractValidations_RunsCollectorFirst(t *testing.T) {
	mockVal := &testCollectorValidator{}
	reg := &registry{
		tagValidators: map[string]TagValidator{
			mockVal.TagName(): mockVal,
		},
		tagIndex: []string{mockVal.TagName()},
	}
	reg.initialized.Store(true)

	ctx := Context{
		Scope: ScopeType,
		Path:  field.NewPath("MyType"),
		Type: &types.Type{
			Kind:         types.Struct,
			CommentLines: []string{"+" + mockVal.TagName()},
		},
	}
	tag := codetags.Tag{
		Name: mockVal.TagName(),
	}

	meta, err := reg.CollectMetadata(ctx)
	if err != nil {
		t.Fatalf("unexpected error collecting metadata: %v", err)
	}
	_, err = reg.ExtractValidations(ctx, meta, tag)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !mockVal.collected || !mockVal.validated {
		t.Errorf("expected both collect and validate to run, got collected=%v, validated=%v", mockVal.collected, mockVal.validated)
	}

	if len(mockVal.callOrder) < 2 || mockVal.callOrder[0] != "CollectMetadata" || mockVal.callOrder[1] != "GetValidations" {
		t.Errorf("expected call order [CollectMetadata, GetValidations], got %v", mockVal.callOrder)
	}
}

func TestConditionsCompare(t *testing.T) {
	c1 := Conditions{}.WithOptionEnabled("Alpha")
	c2 := Conditions{}.WithOptionEnabled("Beta")
	c3 := Conditions{}.WithOptionEnabled("Alpha").WithOptionDisabled("DisabledA")
	c4 := Conditions{}.WithOptionEnabled("Alpha").WithOptionDisabled("DisabledB")

	if c1.Compare(c2) >= 0 {
		t.Errorf("expected c1 < c2")
	}
	if c2.Compare(c1) <= 0 {
		t.Errorf("expected c2 > c1")
	}
	if c3.Compare(c4) >= 0 {
		t.Errorf("expected c3 < c4")
	}
}

func TestCollectMetadata_Dependencies(t *testing.T) {
	structPath := field.NewPath("MyStruct")
	stringPtrType := types.PointerTo(types.String)

	ctx := Context{
		Scope: ScopeType,
		Path:  structPath,
		Type: &types.Type{
			Kind: types.Struct,
			Members: []types.Member{
				{
					Name:         "Foo",
					Type:         stringPtrType,
					CommentLines: []string{`+k8s:dependentRequired("bar")`},
					Tags:         `json:"foo,omitempty"`,
				},
				{
					Name: "Bar",
					Type: stringPtrType,
					Tags: `json:"bar,omitempty"`,
				},
			},
		},
	}

	meta, err := globalRegistry.CollectMetadata(ctx)
	if err != nil {
		t.Fatalf("unexpected error from CollectMetadata: %v", err)
	}

	deps := meta.SortedDependencies()
	if len(deps) != 1 {
		t.Fatalf("expected 1 dependency, got %v", deps)
	}

	if deps[0].triggerJSONName != "foo" || deps[0].dependentJSONName != "bar" {
		t.Errorf("expected trigger 'foo' and dependent 'bar', got %q -> %q", deps[0].triggerJSONName, deps[0].dependentJSONName)
	}

	fieldCtx := Context{Scope: ScopeField, ParentType: ctx.Type}
	val, err := dependencyTagValidator{dependencyRequired}.GetValidations(fieldCtx, meta, codetags.Tag{})
	if err != nil {
		t.Fatalf("unexpected error from GetValidations: %v", err)
	}
	if len(val.Validations.Functions)+len(val.Validations.TypeFunctions) != 1 {
		t.Errorf("expected 1 generated validation function, got %d (Functions: %d, TypeFunctions: %d)",
			len(val.Validations.Functions)+len(val.Validations.TypeFunctions),
			len(val.Validations.Functions), len(val.Validations.TypeFunctions))
	}
}

func TestCollectMetadata_UpdateConstraints(t *testing.T) {
	stringPtrType := types.PointerTo(types.String)
	ctx := Context{
		Scope: ScopeField,
		Path:  field.NewPath("foo"),
		Type:  stringPtrType,
		Member: &types.Member{
			Name:         "Foo",
			Type:         stringPtrType,
			CommentLines: []string{`+k8s:update=NoSet`, `+k8s:update=NoModify`},
			Tags:         `json:"foo,omitempty"`,
		},
	}

	meta, err := globalRegistry.CollectMetadata(ctx)
	if err != nil {
		t.Fatalf("unexpected error from CollectMetadata: %v", err)
	}

	ucs := meta.SortedUpdateConstraints()
	if len(ucs) == 0 || ucs[0].Payload == nil {
		t.Fatalf("expected updateMetadata, got %v", ucs)
	}
	um := ucs[0].Payload

	if um.constraints.Len() != 2 {
		t.Errorf("expected 2 merged constraints (NoSet, NoModify), got %d: %v", um.constraints.Len(), um.constraints.UnsortedList())
	}

	evalCtx := ctx
	evalCtx.Member = nil
	val, err := GenerateUpdateValidations(evalCtx, meta)
	if err != nil {
		t.Fatalf("unexpected error from GenerateUpdateValidations: %v", err)
	}
	if len(val.Functions) != 1 {
		t.Errorf("expected 1 generated validation function, got %d", len(val.Functions))
	}
}

func TestSchemaMetadata_Merge(t *testing.T) {
	sm1 := SchemaMetadata{}
	n1 := sm1.GetOrCreateNode("spec")
	n1.Opaque = true
	n1.Dependencies = []Conditional[*dependencyMetadata]{
		{Conditions: Conditions{}.WithOptionEnabled("Alpha"), Payload: &dependencyMetadata{}},
	}

	sm2 := SchemaMetadata{}
	n2 := sm2.GetOrCreateNode("spec.template")
	n2.Items = []Conditional[*itemMetadata]{
		{Conditions: Conditions{}, Payload: &itemMetadata{itemKey: "name"}},
	}
	n2_spec := sm2.GetOrCreateNode("spec")
	n2_spec.Dependencies = []Conditional[*dependencyMetadata]{
		{Conditions: Conditions{}.WithOptionEnabled("Beta"), Payload: &dependencyMetadata{}},
	}

	merged := SchemaMetadata{}
	if err := merged.Merge(sm1); err != nil {
		t.Fatalf("unexpected error merging sm1: %v", err)
	}
	if err := merged.Merge(sm2); err != nil {
		t.Fatalf("unexpected error merging sm2: %v", err)
	}

	sorted := merged.SortedNodes()
	if len(sorted) != 2 {
		t.Fatalf("expected 2 sorted nodes, got %d", len(sorted))
	}
	if sorted[0].Path != "spec" || sorted[1].Path != "spec.template" {
		t.Errorf("expected sorted paths ['spec', 'spec.template'], got [%q, %q]", sorted[0].Path, sorted[1].Path)
	}

	specNode := merged.Nodes["spec"]
	if !specNode.Opaque {
		t.Errorf("expected specNode.Opaque=true after merge")
	}
	if len(specNode.Dependencies) != 2 {
		t.Errorf("expected 2 Dependencies conditional items on specNode, got %d", len(specNode.Dependencies))
	}

	tplNode := merged.Nodes["spec.template"]
	if len(tplNode.Items) != 1 || tplNode.Items[0].Payload.itemKey != "name" {
		t.Errorf("expected Items with itemKey 'name' on tplNode, got %v", tplNode.Items)
	}
}

func TestCollectMetadata_Tree(t *testing.T) {
	structPath := field.NewPath("MyStruct")
	stringType := types.String

	ctx := Context{
		Scope: ScopeType,
		Path:  structPath,
		Type: &types.Type{
			Kind: types.Struct,
			Members: []types.Member{
				{
					Name:         "Type",
					Type:         stringType,
					CommentLines: []string{"+k8s:unionDiscriminator"},
					Tags:         `json:"type"`,
				},
				{
					Name:         "FieldA",
					Type:         stringType,
					CommentLines: []string{"+k8s:unionMember", "+k8s:update=NoSet"},
					Tags:         `json:"fieldA,omitempty"`,
				},
				{
					Name: "FieldB",
					Type: types.PointerTo(&types.Type{
						Name: types.Name{Name: "FieldBType"},
						Kind: types.Struct,
						Members: []types.Member{
							{Name: "Foo", Type: stringType, Tags: `json:"foo"`},
						},
					}),
					CommentLines: []string{"+k8s:unionMember", `+k8s:subfield(foo)=+k8s:update=NoSet`},
					Tags:         `json:"fieldB,omitempty"`,
				},
			},
		},
	}

	meta, err := globalRegistry.CollectMetadata(ctx)
	if err != nil {
		t.Fatalf("unexpected error from CollectMetadata: %v", err)
	}

	sorted := meta.SortedNodes()
	if len(sorted) != 3 {
		var paths []string
		for _, node := range sorted {
			paths = append(paths, fmt.Sprintf("%q", node.Path))
		}
		t.Fatalf("expected 3 sorted nodes, got %d: %v", len(sorted), paths)
	}
	if sorted[0].Path != "MyStruct" || sorted[1].Path != "MyStruct.fieldA" || sorted[2].Path != "MyStruct.fieldB.foo" {
		t.Errorf("expected sorted paths ['MyStruct', 'MyStruct.fieldA', 'MyStruct.fieldB.foo'], got [%q, %q, %q]", sorted[0].Path, sorted[1].Path, sorted[2].Path)
	}

	if len(sorted[0].Unions) != 1 {
		t.Errorf("expected 1 union item on root node, got %d", len(sorted[0].Unions))
	}

	if len(sorted[1].UpdateConstraints) != 1 {
		t.Errorf("expected 1 update constraint on fieldA node, got %v", sorted[1].UpdateConstraints)
	}

	if len(sorted[2].UpdateConstraints) != 1 {
		t.Errorf("expected 1 update constraint on fieldB.spec.foo node, got %v", sorted[2].UpdateConstraints)
	}
}
