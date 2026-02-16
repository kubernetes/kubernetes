package pod

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func RunDeclarativeValidateEvictionRespondersTestCases[T runtime.Object](t *testing.T, ctx context.Context, strategy rest.RESTCreateStrategy, specPath *field.Path, baseObj T, setEvictionResponders func(baseObj T, responders []api.EvictionResponder, schedulingGroup *api.PodSchedulingGroup)) {
	var maxLimitResponders []api.EvictionResponder
	for i := range 11 {
		maxLimitResponders = append(maxLimitResponders, api.EvictionResponder{
			Name:     fmt.Sprintf("%d-foo.example.com", i),
			Priority: new(int32(1000 + i)),
		})
	}

	testCases := map[string]struct {
		input           []api.EvictionResponder
		schedulingGroup *api.PodSchedulingGroup
		expectedErrs    field.ErrorList
	}{
		"none": {
			input: []api.EvictionResponder{},
		},
		"correct": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/bar",
					Priority: new(int32(1000)),
				},
			},
		},
		"long": {
			input: []api.EvictionResponder{
				{
					Name:     "a.b.c.d.e.f.g.h.i.foo.example.com/bar",
					Priority: new(int32(10000)),
				},
			},
		},
		"long subdomains": {
			input: []api.EvictionResponder{
				{
					Name:     strings.Repeat("a", 63) + "." + strings.Repeat("b", 63) + "." + strings.Repeat("c", 63) + "." + strings.Repeat("d", 49) + ".example.com/bar",
					Priority: new(int32(7000)),
				},
			},
		},
		"numbers": {
			input: []api.EvictionResponder{
				{
					Name:     "5.3.5.9/bar",
					Priority: new(int32(1000)),
				},
			},
		},
		"multiple": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/test",
					Priority: new(int32(7500)),
				}, {
					Name:     "bar.example.com/baz",
					Priority: new(int32(7501)),
				},
			},
		},
		"multiple with the same priority": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/test",
					Priority: new(int32(7500)),
				}, {
					Name:     "bar.example.com/baz",
					Priority: new(int32(7500)),
				},
			},
		},
		"schedulingGroup not supported": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/test",
					Priority: new(int32(10000)),
				},
			},
			schedulingGroup: &api.PodSchedulingGroup{
				PodGroupName: new("blue"),
			},
			expectedErrs: field.ErrorList{
				field.Forbidden(specPath.Child("schedulingGroup"), "").WithOrigin("dependentForbidden").MarkAlpha(),
			},
		},
		"max items": {
			input: maxLimitResponders,
			expectedErrs: field.ErrorList{
				field.TooMany(specPath.Child("evictionResponders"), 0, 0).WithOrigin("maxItems"),
			},
		},
		"duplicate": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/test",
					Priority: new(int32(10000)),
				}, {
					Name:     "bar.example.com/test",
					Priority: new(int32(9000)),
				}, {
					Name:     "foo.example.com/test",
					Priority: new(int32(8000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Duplicate(specPath.Child("evictionResponders").Index(2), ""),
			},
		},
		// name
		"empty name": {
			input: []api.EvictionResponder{
				{
					Name:     "",
					Priority: new(int32(5000)),
				}, {
					Name:     "foo.example.com/test",
					Priority: new(int32(10000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("evictionResponders").Index(0).Child("name"), ""),
			},
		},
		"one segment": {
			input: []api.EvictionResponder{
				{
					Name:     "invalid",
					Priority: new(int32(10000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("name"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"underscores": {
			input: []api.EvictionResponder{
				{
					Name:     "underscores_are_bad.example.com/bar",
					Priority: new(int32(10000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("name"), "", "").WithOrigin("format=k8s-prefixed-label-key"),
			},
		},
		"reserved k8s.io domain": {
			input: []api.EvictionResponder{
				{
					Name:     "k8s.io/key",
					Priority: new(int32(10000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("name"), "", "domain names *.k8s.io, *.kubernetes.io are reserved"),
			},
		},
		"reserved kubernetes.io domain": {
			input: []api.EvictionResponder{
				{
					Name:     "dev.kubernetes.io/key",
					Priority: new(int32(10000)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("name"), "", "domain names *.k8s.io, *.kubernetes.io are reserved"),
			},
		},
		// priority
		"priority required": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/bar",
					Priority: nil,
				},
			},
			expectedErrs: field.ErrorList{
				field.Required(specPath.Child("evictionResponders").Index(0).Child("priority"), ""),
			},
		},
		"negative priority": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/bar",
					Priority: new(int32(-1)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("priority"), "", "").WithOrigin("minimum"),
			},
		},
		"priority over maximum": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/bar",
					Priority: new(int32(100001)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("priority"), "", "").WithOrigin("maximum"),
			},
		},
		"reserved priority - end": {
			input: []api.EvictionResponder{
				{
					Name:     "foo.example.com/bar",
					Priority: new(int32(999)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("priority"), "", "priorities 0-999 are reserved for responders with *.k8s.io suffix"),
			},
		},
		// Support for new k8s responders and a new priority slot must be explicitly added to the API and validation.
		"reserved priority triggers unknown k8s responders": {
			input: []api.EvictionResponder{
				{
					Name:     "dev.kubernetes.io/key",
					Priority: new(int32(0)),
				},
			},
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("name"), "", "domain names *.k8s.io, *.kubernetes.io are reserved"),
				field.Invalid(specPath.Child("evictionResponders").Index(0).Child("priority"), "", "priorities 0-999 are reserved for responders with *.k8s.io suffix"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run("evictionResponders: "+k, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setEvictionResponders(obj, tc.input, tc.schedulingGroup)
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.expectedErrs)
		})
	}
}
