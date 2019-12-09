package typed

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/schema"
	"sigs.k8s.io/structured-merge-diff/value"
)

func TestRemoveShallowRemovesParents(t *testing.T) {

	test, err := value.FromJSON([]byte(`
	{
		"removeMap": {
			"child": "value"
		},
		"removeList": [
			{"child": "value"}
		],
		"keep": "value",
		"keepMap": {
			"child": "value"
		},
		"keepList": [
			{"child": "value"}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}

	remove := fieldpath.NewSet(
		fieldpath.MakePathOrDie("removeMap"),
		fieldpath.MakePathOrDie("removeList"),
	)

	str := schema.String

	atom := schema.Atom{
		Map: &schema.Map{
			Fields: []schema.StructField{
				{
					Name: "removeMap",
					Type: schema.TypeRef{
						Inlined: schema.Atom{Map: &schema.Map{
							Fields: []schema.StructField{
								{Name: "child", Type: schema.TypeRef{Inlined: schema.Atom{Scalar: &str}}}}},
						},
					},
				},
				{
					Name: "removeList",
					Type: schema.TypeRef{
						Inlined: schema.Atom{
							List: &schema.List{
								ElementRelationship: schema.Separable,
								ElementType: schema.TypeRef{
									Inlined: schema.Atom{Map: &schema.Map{
										Fields: []schema.StructField{
											{Name: "child", Type: schema.TypeRef{Inlined: schema.Atom{Scalar: &str}}}}},
									},
								},
							},
						},
					},
				},
				{
					Name: "keep",
					Type: schema.TypeRef{
						Inlined: schema.Atom{
							Scalar: &str,
						},
					},
				},
				{
					Name: "keepMap",
					Type: schema.TypeRef{
						Inlined: schema.Atom{Map: &schema.Map{
							Fields: []schema.StructField{
								{Name: "child", Type: schema.TypeRef{Inlined: schema.Atom{Scalar: &str}}}}},
						},
					},
				},
				{
					Name: "keepList",
					Type: schema.TypeRef{
						Inlined: schema.Atom{
							List: &schema.List{
								ElementRelationship: schema.Separable,
								ElementType: schema.TypeRef{
									Inlined: schema.Atom{Map: &schema.Map{
										Fields: []schema.StructField{
											{Name: "child", Type: schema.TypeRef{Inlined: schema.Atom{Scalar: &str}}}}},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	typeDef := schema.TypeDef{
		Name: "test",
		Atom: atom,
	}

	typeRef := schema.TypeRef{
		Inlined: atom,
	}

	schema := &schema.Schema{
		Types: []schema.TypeDef{
			typeDef,
		},
	}

	removeItemsWithSchema(&test, remove, schema, typeRef, false)

	expect, err := value.FromJSON([]byte(`{"keep": "value", "keepMap": {"child": "value"}, "keepList": [{"child": "value"}], "removeMap": {"child":"value"}, "removeList": [{"child": "value"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !test.Equals(expect) {
		t.Fatalf("unexpected result after remove:\ngot: %v\nexp: %v", test.String(), expect.String())
	}

	removeItemsWithSchema(&test, remove, schema, typeRef, true)

	expect, err = value.FromJSON([]byte(`{"keep": "value", "keepMap": {"child": "value"}, "keepList": [{"child": "value"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !test.Equals(expect) {
		t.Fatalf("unexpected result after remove:\ngot: %v\nexp: %v", test.String(), expect.String())
	}
}
