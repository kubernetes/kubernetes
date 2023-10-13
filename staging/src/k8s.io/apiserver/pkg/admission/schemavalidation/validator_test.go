package schemavalidation_test

import (
	"context"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission/schemavalidation"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// Show that the factory is capable of giving validators for schemas it knows
// about, and that it returns errors for schemas it doesn't know about.
func TestFactory(t *testing.T) {
	fooV1Bar := schema.GroupVersionKind{Group: "foo", Version: "v1", Kind: "Bar"}
	reg := &fakeSchemaRegistry{
		Scheme:   runtime.NewScheme(),
		registry: map[schema.GroupVersionKind]*spec.Schema{},
		typeMap:  map[string]schema.GroupVersionKind{},
	}
	reg.AddKnownType(fooV1Bar, &runtime.Unknown{}, &spec.Schema{})

	resolve := resolver.NewDefinitionsSchemaResolverFromNamer(reg.GetDefinitionName, reg.GetOpenAPIDefinitions)
	factory, err := schemavalidation.NewFactory(resolve)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Show that the factory is capable of giving validators for schemas it knows
	// about
	validator, err := factory.ForGroupVersionKind(fooV1Bar, reg.Scheme)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if validator == nil {
		t.Fatalf("unexpected nil validator")
	}

	// Show that the factory returns errors for schemas it doesn't know about
	_, err = factory.ForGroupVersionKind(schema.GroupVersionKind{Group: "foo", Version: "v1", Kind: "Baz"}, reg.Scheme)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}

type baseObject struct {
	metav1.ObjectMeta
	metav1.TypeMeta
}

func (b *baseObject) GroupVersionKind() schema.GroupVersionKind {
	gv, err := schema.ParseGroupVersion(b.TypeMeta.APIVersion)
	if err != nil {
		panic(err)
	}
	return schema.GroupVersionKind{
		Group:   gv.Group,
		Version: gv.Version,
		Kind:    b.TypeMeta.Kind,
	}
}

func (b *baseObject) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	b.TypeMeta.APIVersion = gvk.GroupVersion().String()
	b.TypeMeta.Kind = gvk.Kind
}

func (b *baseObject) GetObjectKind() schema.ObjectKind {
	return b
}

// Shouldn't be called for these tests
func (b baseObject) DeepCopyObject() runtime.Object {
	panic("unimplemented")
}

// Show that the validator properly converts its input to the versioned
// schema, and validates the converted input against the schema
func TestValidatorConversion(t *testing.T) {
	type FooBar struct {
		baseObject
		MyField string `json:"myField,omitempty"`
	}
	type FooV1Bar struct {
		baseObject
		MyField string `json:"myField,omitempty"`
	}

	type FooV2Bar struct {
		baseObject
		MyField string `json:"myField,omitempty"`
	}

	fooInternal := schema.GroupVersionKind{Group: "foo", Version: runtime.APIVersionInternal, Kind: "Bar"}
	fooV2Bar := schema.GroupVersionKind{Group: "foo", Version: "v2", Kind: "Bar"}
	fooV1Bar := schema.GroupVersionKind{Group: "foo", Version: "v1", Kind: "Bar"}

	reg := &fakeSchemaRegistry{
		Scheme:   runtime.NewScheme(),
		registry: map[schema.GroupVersionKind]*spec.Schema{},
		typeMap:  map[string]schema.GroupVersionKind{},
	}
	reg.AddKnownType(fooInternal, &FooBar{}, nil)
	reg.AddKnownType(fooV1Bar, &FooV1Bar{}, &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type: spec.StringOrArray{"object"},
			Properties: map[string]spec.Schema{
				"myField": {
					SchemaProps: spec.SchemaProps{
						Type: spec.StringOrArray{"string"},
						Enum: []interface{}{"v1"},
					},
				},
			},
		},
	})
	reg.AddKnownType(fooV2Bar, &FooV2Bar{}, &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type: spec.StringOrArray{"object"},
			Properties: map[string]spec.Schema{
				"myField": {
					SchemaProps: spec.SchemaProps{
						Type: spec.StringOrArray{"string"},
						Enum: []interface{}{"v2"},
					},
				},
			},
		},
	})

	reg.AddConversionFunc((*FooBar)(nil), (*FooV1Bar)(nil), func(a, b interface{}, scope conversion.Scope) error {
		b.(*FooV1Bar).MyField = "notv1"
		return nil
	})

	reg.AddConversionFunc((*FooBar)(nil), (*FooV2Bar)(nil), func(a, b interface{}, scope conversion.Scope) error {
		b.(*FooV2Bar).MyField = "notv2"
		return nil
	})

	resolve := resolver.NewDefinitionsSchemaResolverFromNamer(reg.GetDefinitionName, reg.GetOpenAPIDefinitions)
	factory, err := schemavalidation.NewFactory(resolve)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fooV1Validator, err := factory.ForGroupVersionKind(fooV1Bar, reg.Scheme)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fooV2Validator, err := factory.ForGroupVersionKind(fooV2Bar, reg.Scheme)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Show that the validator properly converts its input to the versioned
	// schema, and validates the converted input against the schema
	fooBar := &FooBar{}
	fooBar.SetGroupVersionKind(fooInternal)
	errs := fooV1Validator.Validate(context.TODO(), fooBar)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %v", errs)
	}

	expectation := `myField: Unsupported value: "notv1": supported values: "v1"`
	if errs[0].Error() != expectation {
		t.Fatalf("expected error %q, got %q", expectation, errs[0].Error())
	}

	expectation = `myField: Unsupported value: "notv2": supported values: "v2"`
	errs = fooV2Validator.Validate(context.TODO(), fooBar)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %v", errs)
	}
}

type fakeSchemaRegistry struct {
	*runtime.Scheme
	registry map[schema.GroupVersionKind]*spec.Schema
	typeMap  map[string]schema.GroupVersionKind
}

func (f *fakeSchemaRegistry) AddKnownType(gvk schema.GroupVersionKind, obj runtime.Object, sch *spec.Schema) {

	f.Scheme.AddKnownTypeWithName(gvk, obj)

	if gvk.Version == runtime.APIVersionInternal {
		// We don't store schemas of internal versions
		return
	}

	typ := reflect.Indirect(reflect.ValueOf(obj)).Type()
	f.typeMap[typ.String()] = gvk
	f.registry[gvk] = sch
}

func (f *fakeSchemaRegistry) GetOpenAPIDefinitions(ref common.ReferenceCallback) map[string]common.OpenAPIDefinition {
	res := map[string]common.OpenAPIDefinition{}
	for t, gvk := range f.typeMap {
		sch := f.registry[gvk]
		res[t] = common.OpenAPIDefinition{
			Schema: *sch,
		}
	}
	return res
}

func (f *fakeSchemaRegistry) GetDefinitionName(s string) (string, spec.Extensions) {
	gvk, ok := f.typeMap[s]
	if !ok {
		return s, nil
	}

	cs := struct {
		GVKs []metav1.GroupVersionKind `json:"x-kubernetes-group-version-kind"`
	}{
		GVKs: []metav1.GroupVersionKind{metav1.GroupVersionKind(gvk)},
	}
	v, e := runtime.DefaultUnstructuredConverter.ToUnstructured(&cs)
	if e != nil {
		panic(e)
	}
	return s, spec.Extensions(v)
}
