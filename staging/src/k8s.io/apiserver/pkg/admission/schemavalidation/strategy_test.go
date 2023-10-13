package schemavalidation_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission/schemavalidation"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

type validateFunc func(ctx context.Context, obj runtime.Object) field.ErrorList
type validateUpdateFunc func(ctx context.Context, newObj, oldObj runtime.Object) field.ErrorList

type fakeStrategy struct {
	validateFunc
	validateUpdateFunc
}

var _ rest.RESTCreateStrategy = &fakeStrategy{}
var _ rest.RESTUpdateStrategy = &fakeStrategy{}

type fakeValidator struct {
	validateFunc
	validateUpdateFunc
}

var _ schemavalidation.Validator = &fakeValidator{}

func TestStrategyWrapEnablement(t *testing.T) {
	// Show that wrapping a strategy with the feature disabled returns the strategy
	// unmodified, and that wrapping a strategy with the feature enabled returns
	// the strategy wrapped.
	myStrategy := &fakeStrategy{}
	for _, enabled := range []bool{false, true} {
		t.Run("enabled="+fmt.Sprint(enabled), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, enabled)()
			create := schemavalidation.WrapCreateStrategyIfEnabled(myStrategy)
			update := schemavalidation.WrapUpdateStrategyIfEnabled(myStrategy)

			// If feature is enabled, must conform to the interface DeclarativeValidationStrategy
			// If feature is not enabled, must be equal to the input strategy
			if _, ok := create.(schemavalidation.DeclarativeValidationStrategy); enabled != ok {
				t.Errorf("WrapCreateStrategyIfEnabled() is DeclarativeValidationStrategy: %v, want %v", ok, enabled)
			}
			if _, ok := update.(schemavalidation.DeclarativeValidationStrategy); enabled != ok {
				t.Errorf("WrapUpdateStrategyIfEnabled() is DeclarativeValidationStrategy: %v, want %v", ok, enabled)
			}

			if enabled == (create == myStrategy) {
				t.Errorf("WrapCreateStrategyIfEnabled() is no-op: %v, want %v", create == myStrategy, enabled)
			}

			if enabled == (update == myStrategy) {
				t.Errorf("WrapUpdateStrategyIfEnabled() is no-op: %v, want %v", update == myStrategy, enabled)
			}
		})
	}
}

func TestWarnings(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeclarativeValidation, true)()
	strategyErrors := &field.ErrorList{}
	validatorErrors := &field.ErrorList{}

	myStrategy := &fakeStrategy{
		validateFunc: func(ctx context.Context, obj runtime.Object) field.ErrorList {
			return *strategyErrors
		},
		validateUpdateFunc: func(ctx context.Context, newObj, oldObj runtime.Object) field.ErrorList {
			return *strategyErrors
		},
	}
	myValidator := &fakeValidator{
		validateFunc: func(ctx context.Context, obj runtime.Object) field.ErrorList {
			return *validatorErrors
		},
		validateUpdateFunc: func(ctx context.Context, newObj, oldObj runtime.Object) field.ErrorList {
			return *validatorErrors
		},
	}

	create := schemavalidation.WrapCreateStrategyIfEnabled(myStrategy)
	update := schemavalidation.WrapUpdateStrategyIfEnabled(myStrategy)

	create.(schemavalidation.DeclarativeValidationStrategy).SetValidator(myValidator)
	update.(schemavalidation.DeclarativeValidationStrategy).SetValidator(myValidator)

	warningRecorder := newWarningRecorder()
	ctx := context.TODO()
	ctx = warning.WithWarningRecorder(ctx, warningRecorder)

	type testCase struct {
		name             string
		strategyErrors   field.ErrorList
		validatorErrors  field.ErrorList
		expectedErrors   field.ErrorList
		expectedWarnings []string
	}

	cases := []testCase{
		{
			name:             "errors raised by the validator but not the strategy are thrown as warnings",
			strategyErrors:   nil,
			validatorErrors:  field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			expectedErrors:   nil,
			expectedWarnings: []string{"Added Error: foo: Invalid value: \"bar\": baz"},
		},
		// 2. Test that errors raised by the stratgy but not the validator are
		//		thrown as warnings with prefix `Deleted Error: `
		{
			name:             "errors raised by the strategy but not the validator are thrown as errors. but with a warning about the deletion",
			strategyErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			validatorErrors:  nil,
			expectedErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			expectedWarnings: []string{"Deleted Error: foo: Invalid value: \"bar\": baz"},
		},
		// 3. Test that errors thrown by both the strategy and validator are
		//		not thrown as warnings.
		{
			name:             "errors raised by both the strategy and validator are not thrown as warnings",
			strategyErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			validatorErrors:  field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			expectedErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			expectedWarnings: nil,
		},

		// 4. Test that errors thrown by both the strategy and validator are
		//		not thrown as warnings, but if the validator has extra errors,
		//		only those are thrown as warnings.
		{
			name:             "errors raised by both the strategy and validator are not thrown as warnings, but if the validator has extra errors, only those are thrown as warnings",
			strategyErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			validatorErrors:  field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz"), field.Invalid(field.NewPath("foo"), "bar", "Extra error")},
			expectedErrors:   field.ErrorList{field.Invalid(field.NewPath("foo"), "bar", "baz")},
			expectedWarnings: []string{"Added Error: foo: Invalid value: \"bar\": Extra error"},
		},
	}

	checkCase := func(t *testing.T, c *testCase, errs field.ErrorList, warnings []string) {
		if len(errs) != len(c.expectedErrors) {
			t.Errorf("expected %d errors, got %d", len(c.expectedErrors), len(errs))
		}
		for i := range errs {
			if errs[i].Error() != c.expectedErrors[i].Error() {
				t.Errorf("expected error %q, got %q", c.expectedErrors[i].Error(), errs[i].Error())
			}
		}

		if len(warnings) != len(c.expectedWarnings) {
			t.Errorf("expected %d warnings, got %d", len(c.expectedWarnings), len(warnings))
		}

		for i := range warnings {
			if warnings[i] != c.expectedWarnings[i] {
				t.Errorf("expected warning %q, got %q", c.expectedWarnings[i], warnings[i])
			}
		}
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			*strategyErrors = c.strategyErrors
			*validatorErrors = c.validatorErrors

			t.Run("create", func(t *testing.T) {
				warningRecorder.Reset()
				errs := create.Validate(ctx, nil)
				checkCase(t, &c, errs, warningRecorder.Warnings())
			})

			t.Run("update", func(t *testing.T) {
				warningRecorder.Reset()
				errs := update.ValidateUpdate(ctx, nil, nil)
				checkCase(t, &c, errs, warningRecorder.Warnings())
			})
		})
	}
}

func (f *fakeStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	if f.validateFunc == nil {
		return nil
	}
	return f.validateFunc(ctx, obj)
}
func (f *fakeStrategy) ValidateUpdate(ctx context.Context, newObj, oldObj runtime.Object) field.ErrorList {
	if f.validateUpdateFunc == nil {
		return nil
	}

	return f.validateUpdateFunc(ctx, newObj, oldObj)
}

// Validate implements schemavalidation.Validator.
func (v *fakeValidator) Validate(ctx context.Context, new runtime.Object) field.ErrorList {
	if v.validateFunc == nil {
		return nil
	}

	return v.validateFunc(ctx, new)
}

// ValidateUpdate implements schemavalidation.Validator.
func (v *fakeValidator) ValidateUpdate(ctx context.Context, new runtime.Object, old runtime.Object) field.ErrorList {
	if v.validateUpdateFunc == nil {
		return nil
	}

	return v.validateUpdateFunc(ctx, new, old)
}

func (fakeStrategy) Recognizes(gvk schema.GroupVersionKind) bool { return false }
func (fakeStrategy) ObjectKinds(runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	return nil, false, nil
}
func (fakeStrategy) GenerateName(base string) string                                     { return "" }
func (fakeStrategy) NamespaceScoped() bool                                               { return true }
func (fakeStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object)            {}
func (fakeStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string   { return nil }
func (fakeStrategy) Canonicalize(obj runtime.Object)                                     {}
func (fakeStrategy) AllowCreateOnUpdate() bool                                           { return false }
func (fakeStrategy) AllowUnconditionalUpdate() bool                                      { return true }
func (fakeStrategy) PrepareForUpdate(ctx context.Context, newObj, oldObj runtime.Object) {}
func (fakeStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

type warningRecorder struct {
	sync.Mutex
	warnings []string
}

func newWarningRecorder() *warningRecorder {
	return &warningRecorder{}
}

func (r *warningRecorder) AddWarning(_, text string) {
	r.Lock()
	defer r.Unlock()
	r.warnings = append(r.warnings, text)
}

func (r *warningRecorder) Warnings() []string {
	r.Lock()
	defer r.Unlock()
	res := make([]string, len(r.warnings))
	copy(res, r.warnings)
	return res
}

func (r *warningRecorder) Reset() {
	r.Lock()
	defer r.Unlock()
	r.warnings = nil
}
