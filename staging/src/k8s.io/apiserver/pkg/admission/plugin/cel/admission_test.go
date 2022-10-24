/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/api/admissionregistration/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/cel/internal/generic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

var (
	scheme *runtime.Scheme = func() *runtime.Scheme {
		res := runtime.NewScheme()
		res.AddKnownTypeWithName(paramsGVK, &unstructured.Unstructured{})
		res.AddKnownTypeWithName(schema.GroupVersionKind{
			Group:   paramsGVK.Group,
			Version: paramsGVK.Version,
			Kind:    paramsGVK.Kind + "List",
		}, &unstructured.UnstructuredList{})

		if err := v1alpha1.AddToScheme(res); err != nil {
			panic(err)
		}
		return res
	}()
	codecs    serializer.CodecFactory = serializer.NewCodecFactory(scheme)
	paramsGVK schema.GroupVersionKind = schema.GroupVersionKind{
		Group:   "example.com",
		Version: "v1",
		Kind:    "ParamsConfig",
	}

	fakeRestMapper *meta.DefaultRESTMapper = func() *meta.DefaultRESTMapper {
		res := meta.NewDefaultRESTMapper([]schema.GroupVersion{
			{
				Group:   "",
				Version: "v1",
			},
		})

		res.Add(paramsGVK, meta.RESTScopeNamespace)
		res.Add(definitionGVK, meta.RESTScopeRoot)
		res.Add(bindingGVK, meta.RESTScopeRoot)
		return res
	}()

	definitionGVK schema.GroupVersionKind = must3(scheme.ObjectKinds(&v1alpha1.ValidatingAdmissionPolicy{}))[0]
	bindingGVK    schema.GroupVersionKind = must3(scheme.ObjectKinds(&v1alpha1.ValidatingAdmissionPolicyBinding{}))[0]

	definitionsGVR schema.GroupVersionResource = must(fakeRestMapper.RESTMapping(definitionGVK.GroupKind(), definitionGVK.Version)).Resource
	bindingsGVR    schema.GroupVersionResource = must(fakeRestMapper.RESTMapping(bindingGVK.GroupKind(), bindingGVK.Version)).Resource

	// Common objects
	denyPolicy *v1alpha1.ValidatingAdmissionPolicy = &v1alpha1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denypolicy.example.com",
			ResourceVersion: "1",
		},
		Spec: v1alpha1.ValidatingAdmissionPolicySpec{
			ParamKind: &v1alpha1.ParamKind{
				APIVersion: paramsGVK.GroupVersion().String(),
				Kind:       paramsGVK.Kind,
			},
			FailurePolicy: ptrTo(v1alpha1.Fail),
		},
	}

	fakeParams *unstructured.Unstructured = &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            "replicas-test.example.com",
				"resourceVersion": "1",
			},
			"maxReplicas": int64(3),
		},
	}

	denyBinding *v1alpha1.ValidatingAdmissionPolicyBinding = &v1alpha1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1alpha1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: denyPolicy.Name,
			ParamRef: &v1alpha1.ParamRef{
				Name:      fakeParams.GetName(),
				Namespace: fakeParams.GetNamespace(),
			},
		},
	}
)

// Interface which has fake compile and match functionality for use in tests
// So that we can test the controller without pulling in any CEL functionality
type fakeCompiler struct {
	DefaultMatch         bool
	CompileFuncs         map[string]func(*v1alpha1.ValidatingAdmissionPolicy) (Validator, error)
	DefinitionMatchFuncs map[string]func(*v1alpha1.ValidatingAdmissionPolicy, admission.Attributes) bool
	BindingMatchFuncs    map[string]func(*v1alpha1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool
}

var _ ValidatorCompiler = &fakeCompiler{}

// Matches says whether this policy definition matches the provided admission
// resource request
func (f *fakeCompiler) DefinitionMatches(definition *v1alpha1.ValidatingAdmissionPolicy, a admission.Attributes) bool {
	namespace, name := definition.Namespace, definition.Name
	key := namespace + "/" + name
	if fun, ok := f.DefinitionMatchFuncs[key]; ok {
		return fun(definition, a)
	}

	// Default is match everything
	return f.DefaultMatch
}

// Matches says whether this policy definition matches the provided admission
// resource request
func (f *fakeCompiler) BindingMatches(binding *v1alpha1.ValidatingAdmissionPolicyBinding, a admission.Attributes) bool {
	namespace, name := binding.Namespace, binding.Name
	key := namespace + "/" + name
	if fun, ok := f.BindingMatchFuncs[key]; ok {
		return fun(binding, a)
	}

	// Default is match everything
	return f.DefaultMatch
}

func (f *fakeCompiler) Compile(
	definition *v1alpha1.ValidatingAdmissionPolicy,
	// Injected RESTMapper to assist with compilation
	mapper meta.RESTMapper,
) (Validator, error) {
	namespace, name := definition.Namespace, definition.Name

	key := namespace + "/" + name
	if fun, ok := f.CompileFuncs[key]; ok {
		return fun(definition)
	}

	return nil, fmt.Errorf("no compile func found for %s", key)
}

func (f *fakeCompiler) RegisterDefinition(definition *v1alpha1.ValidatingAdmissionPolicy, compileFunc func(*v1alpha1.ValidatingAdmissionPolicy) (Validator, error), matchFunc func(*v1alpha1.ValidatingAdmissionPolicy, admission.Attributes) bool) {
	namespace, name := definition.Namespace, definition.Name
	key := namespace + "/" + name
	if compileFunc != nil {

		if f.CompileFuncs == nil {
			f.CompileFuncs = make(map[string]func(*v1alpha1.ValidatingAdmissionPolicy) (Validator, error))
		}
		f.CompileFuncs[key] = compileFunc
	}

	if matchFunc != nil {
		if f.DefinitionMatchFuncs == nil {
			f.DefinitionMatchFuncs = make(map[string]func(*v1alpha1.ValidatingAdmissionPolicy, admission.Attributes) bool)
		}
		f.DefinitionMatchFuncs[key] = matchFunc
	}
}

func (f *fakeCompiler) RegisterBinding(binding *v1alpha1.ValidatingAdmissionPolicyBinding, matchFunc func(*v1alpha1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool) {
	namespace, name := binding.Namespace, binding.Name
	key := namespace + "/" + name

	if matchFunc != nil {
		if f.BindingMatchFuncs == nil {
			f.BindingMatchFuncs = make(map[string]func(*v1alpha1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool)
		}
		f.BindingMatchFuncs[key] = matchFunc
	}
}

func setupFakeTest(t *testing.T, comp *fakeCompiler) (plugin admission.ValidationInterface, paramTracker, policyTracker clienttesting.ObjectTracker, controller *celAdmissionController) {
	return setupTestCommon(t, comp)
}

// Starts CEL admission controller and sets up a plugin configured with it as well
// as object trackers for manipulating the objects available to the system
//
// ParamTracker only knows the gvk `paramGVK`. If in the future we need to
// support multiple types of params this function needs to be augmented
//
// PolicyTracker expects FakePolicyDefinition and FakePolicyBinding types
func setupTestCommon(t *testing.T, compiler ValidatorCompiler) (plugin admission.ValidationInterface, paramTracker, policyTracker clienttesting.ObjectTracker, controller *celAdmissionController) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	t.Cleanup(testContextCancel)

	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)
	tracker := clienttesting.NewObjectTracker(scheme, codecs.UniversalDecoder())

	// Set up fake informers that return instances of mock Policy definitoins
	// and mock policy bindings
	definitionInformer := cache.NewSharedIndexInformer(&cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return tracker.List(definitionsGVR, definitionGVK, "")
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return tracker.Watch(definitionsGVR, "")
		},
	}, &v1alpha1.ValidatingAdmissionPolicy{}, 30*time.Second, nil)

	bindingInformer := cache.NewSharedIndexInformer(&cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			return tracker.List(bindingsGVR, bindingGVK, "")
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			return tracker.Watch(bindingsGVR, "")
		},
	}, &v1alpha1.ValidatingAdmissionPolicyBinding{}, 30*time.Second, nil)

	go definitionInformer.Run(testContext.Done())
	go bindingInformer.Run(testContext.Done())

	admissionController := NewAdmissionController(
		definitionInformer,
		bindingInformer,
		compiler,
		fakeRestMapper,
		dynamicClient,
	).(*celAdmissionController)

	handler, err := NewPlugin()
	require.NoError(t, err)

	pluginInitializer := NewPluginInitializer(admissionController)
	pluginInitializer.Initialize(handler)
	err = admission.ValidateInitialization(handler)
	require.NoError(t, err)

	go admissionController.Run(testContext.Done())
	return handler, dynamicClient.Tracker(), tracker, admissionController
}

// Gets the last reconciled value in the controller of an object with the same
// gvk and name as the given object
//
// If the object is not found both the error and object will be nil.
func (c *celAdmissionController) getCurrentObject(obj runtime.Object) (runtime.Object, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}

	c.mutex.RLock()
	defer c.mutex.RUnlock()

	switch obj.(type) {
	case *unstructured.Unstructured:
		paramSourceGVK := obj.GetObjectKind().GroupVersionKind()
		paramKind := v1alpha1.ParamKind{
			APIVersion: paramSourceGVK.GroupVersion().String(),
			Kind:       paramSourceGVK.Kind,
		}
		var paramInformer generic.Informer[*unstructured.Unstructured]
		if paramInfo, ok := c.paramsCRDControllers[paramKind]; ok {
			paramInformer = paramInfo.controller.Informer()
		} else {
			// Treat unknown CRD the same as not found
			return nil, nil
		}

		// Param type. Just check informer for its GVK
		item, err := paramInformer.Get(accessor.GetName())
		if err != nil {
			if k8serrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}

		return item, nil
	case *v1alpha1.ValidatingAdmissionPolicyBinding:
		namespacedName := accessor.GetNamespace() + "/" + accessor.GetName()
		info, ok := c.bindingInfos[namespacedName]
		if !ok {
			return nil, nil
		}

		return info.lastReconciledValue, nil
	case *v1alpha1.ValidatingAdmissionPolicy:
		namespacedName := accessor.GetNamespace() + "/" + accessor.GetName()
		info, ok := c.definitionInfo[namespacedName]
		if !ok {
			return nil, nil
		}

		return info.lastReconciledValue, nil
	default:
		panic(fmt.Errorf("unhandled object type: %T", obj))
	}
}

// Waits for the given objects to have been the latest reconciled values of
// their gvk/name in the controller
func waitForReconcile(ctx context.Context, controller *celAdmissionController, objects ...runtime.Object) error {
	return wait.PollWithContext(ctx, 200*time.Millisecond, 5*time.Second, func(ctx context.Context) (done bool, err error) {
		for _, obj := range objects {
			currentValue, err := controller.getCurrentObject(obj)
			if err != nil {
				return false, fmt.Errorf("error getting current object: %w", err)
			} else if currentValue == nil {
				// Object not found, but not an error. Keep waiting.
				return false, nil
			}

			objMeta, err := meta.Accessor(obj)
			if err != nil {
				return false, fmt.Errorf("error getting meta accessor for original %T object (%v): %w", obj, obj, err)
			}
			valueMeta, err := meta.Accessor(currentValue)
			if err != nil {
				return false, fmt.Errorf("error getting meta accessor for current %T object (%v): %w", currentValue, currentValue, err)
			}

			if len(objMeta.GetResourceVersion()) == 0 {
				return false, fmt.Errorf("%s named %s has no resource version. please ensure your test objects have an RV",
					obj.GetObjectKind().GroupVersionKind().String(), objMeta.GetName())
			} else if len(valueMeta.GetResourceVersion()) == 0 {
				return false, fmt.Errorf("%s named %s has no resource version. please ensure your test objects have an RV",
					currentValue.GetObjectKind().GroupVersionKind().String(), valueMeta.GetName())
			} else if objMeta.GetResourceVersion() != valueMeta.GetResourceVersion() {
				return false, nil
			}
		}

		return true, nil
	})
}

// Waits for the admissoin controller to have no knowledge of the objects
// with the given GVKs and namespace/names
func waitForReconcileDeletion(ctx context.Context, controller *celAdmissionController, objects ...runtime.Object) error {
	return wait.PollWithContext(ctx, 200*time.Millisecond, 3*time.Hour, func(ctx context.Context) (done bool, err error) {
		for _, obj := range objects {
			currentValue, err := controller.getCurrentObject(obj)
			if err != nil {
				return false, err
			}

			if currentValue != nil {
				return false, nil
			}
		}

		return true, nil
	})
}

func attributeRecord(
	old, new runtime.Object,
	operation admission.Operation,
) admission.Attributes {
	if old == nil && new == nil {
		panic("both `old` and `new` may not be nil")
	}

	accessor, err := meta.Accessor(new)
	if err != nil {
		panic(err)
	}

	// one of old/new may be nil, but not both
	example := new
	if example == nil {
		example = old
	}

	gvk := example.GetObjectKind().GroupVersionKind()
	if gvk.Empty() {
		// If gvk is not populated, try to fetch it from the scheme
		gvk = must3(scheme.ObjectKinds(example))[0]
	}
	mapping, err := fakeRestMapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		panic(err)
	}

	return admission.NewAttributesRecord(
		new,
		old,
		gvk,
		accessor.GetNamespace(),
		accessor.GetName(),
		mapping.Resource,
		"",
		operation,
		nil,
		false,
		nil,
	)
}

func ptrTo[T any](obj T) *T {
	return &obj
}

func must[T any](val T, err error) T {
	if err != nil {
		panic(err)
	}
	return val
}

func must3[T any, I any](val T, _ I, err error) T {
	if err != nil {
		panic(err)
	}
	return val
}

////////////////////////////////////////////////////////////////////////////////
// Functionality Tests
////////////////////////////////////////////////////////////////////////////////

func TestBasicPolicyDefinitionFailure(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	compiler.RegisterDefinition(denyPolicy, func(policy *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
			datalock.Lock()
			passedParams = append(passedParams, params)
			datalock.Unlock()

			// Policy always denies
			return []PolicyDecision{
				{
					Kind:    Deny,
					Message: "Denied",
				},
			}, nil
		}), nil
	}, nil)

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler)

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, denyBinding, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `{"kind":"Deny","message":"Denied"}`)

	require.Equal(t, []*unstructured.Unstructured{fakeParams}, passedParams)
}

// Shows that if a definition does not match the input, it will not be used.
// But with a different input it will be used.
func TestDefinitionDoesntMatch(t *testing.T) {
	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler)
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy,
		func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
			datalock.Lock()
			numCompiles += 1
			datalock.Unlock()

			return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
				datalock.Lock()
				passedParams = append(passedParams, params)
				datalock.Unlock()

				// Policy always denies
				return []PolicyDecision{
					{
						Kind:    Deny,
						Message: "Denied",
					},
				}, nil
			}), nil

		}, func(vap *v1alpha1.ValidatingAdmissionPolicy, a admission.Attributes) bool {
			// Match names with even-numbered length
			obj := a.GetObject()

			accessor, err := meta.Accessor(obj)
			if err != nil {
				t.Fatal(err)
				return false
			}

			return len(accessor.GetName())%2 == 0
		})

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	// Validate a non-matching input.
	// Should pass validation with no error.

	nonMatchingParams := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            "oddlength",
				"resourceVersion": "1",
			},
		},
	}
	require.NoError(t,
		handler.Validate(testContext,
			attributeRecord(
				nil, nonMatchingParams,
				admission.Create), &admission.RuntimeObjectInterfaces{}))
	require.Zero(t, numCompiles)
	require.Empty(t, passedParams)

	// Validate a matching input.
	// Should match and be denied.
	matchingParams := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            "evenlength",
				"resourceVersion": "1",
			},
		},
	}
	require.ErrorContains(t,
		handler.Validate(testContext,
			attributeRecord(
				nil, matchingParams,
				admission.Create), &admission.RuntimeObjectInterfaces{}),
		`{"kind":"Deny","message":"Denied"}`)
	require.Equal(t, numCompiles, 1)
	require.Equal(t, passedParams, []*unstructured.Unstructured{fakeParams})
}

func TestReconfigureBinding(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	fakeParams2 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            "replicas-test2.example.com",
				"resourceVersion": "2",
			},
			"maxReplicas": int64(35),
		},
	}

	compiler.RegisterDefinition(denyPolicy,
		func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
			datalock.Lock()
			numCompiles += 1
			datalock.Unlock()

			return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
				datalock.Lock()
				passedParams = append(passedParams, params)
				datalock.Unlock()

				// Policy always denies
				return []PolicyDecision{
					{
						Kind:    Deny,
						Message: "Denied",
					},
				}, nil
			}), nil

		}, nil)

	denyBinding2 := &v1alpha1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "2",
		},
		Spec: v1alpha1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: denyPolicy.Name,
			ParamRef: &v1alpha1.ParamRef{
				Name:      fakeParams2.GetName(),
				Namespace: fakeParams2.GetNamespace(),
			},
		},
	}

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	ar := attributeRecord(nil, denyBinding, admission.Create)

	err := handler.Validate(
		testContext,
		attributeRecord(nil, denyBinding, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// Expect validation to fail for first time due to binding unconditionally
	// failing
	require.ErrorContains(t, err, `{"kind":"Deny","message":"Denied"}`, "expect policy validation error")

	// Expect `Compile` only called once
	require.Equal(t, 1, numCompiles, "expect `Compile` to be called only once")

	// Show Evaluator was called
	require.Len(t, passedParams, 1, "expect evaluator is called due to proper configuration")

	// Update the tracker to point at different params
	require.NoError(t, tracker.Update(bindingsGVR, denyBinding2, ""))

	// Wait for update to propagate
	// Wait for controller to reconcile given objects
	require.NoError(t, waitForReconcile(testContext, controller, denyBinding2))

	err = handler.Validate(
		testContext,
		ar,
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `{"decision":{"kind":"Deny","message":"configuration error: replicas-test2.example.com not found"}`)
	require.Equal(t, 1, numCompiles, "expect compile is not called when there is configuration error")
	require.Len(t, passedParams, 1, "expect evaluator was not called when there is configuration error")

	// Add the missing params
	require.NoError(t, paramTracker.Add(fakeParams2))

	// Wait for update to propagate
	require.NoError(t, waitForReconcile(testContext, controller, fakeParams2))

	// Expect validation to now fail again.
	err = handler.Validate(
		testContext,
		ar,
		&admission.RuntimeObjectInterfaces{},
	)

	// Expect validation to fail the third time due to validation failure
	require.ErrorContains(t, err, `{"kind":"Deny","message":"Denied"}`, "expected a true policy failure, not a configuration error")
	require.Equal(t, []*unstructured.Unstructured{fakeParams, fakeParams2}, passedParams, "expected call to `Validate` to cause call to evaluator")
	require.Equal(t, 2, numCompiles, "expect changing binding causes a recompile")
}

// Shows that a policy which is in effect will stop being in effect when removed
func TestRemoveDefinition(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
			datalock.Lock()
			passedParams = append(passedParams, params)
			datalock.Unlock()

			// Policy always denies
			return []PolicyDecision{
				{
					Kind:    Deny,
					Message: "Denied",
				},
			}, nil
		}), nil
	}, nil)

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	record := attributeRecord(nil, denyBinding, admission.Create)
	require.ErrorContains(t,
		handler.Validate(
			testContext,
			record,
			&admission.RuntimeObjectInterfaces{},
		),
		`{"kind":"Deny","message":"Denied"}`)

	require.Equal(t, []*unstructured.Unstructured{fakeParams}, passedParams)
	require.NoError(t, tracker.Delete(definitionsGVR, denyPolicy.Namespace, denyPolicy.Name))
	require.NoError(t, waitForReconcileDeletion(testContext, controller, denyPolicy))

	require.NoError(t, handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		record,
		&admission.RuntimeObjectInterfaces{},
	))
}

// Shows that a binding which is in effect will stop being in effect when removed
func TestRemoveBinding(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()
	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
			datalock.Lock()
			passedParams = append(passedParams, params)
			datalock.Unlock()

			// Policy always denies
			return []PolicyDecision{
				{
					Kind:    Deny,
					Message: "Denied",
				},
			}, nil
		}), nil
	}, nil)

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	record := attributeRecord(nil, denyBinding, admission.Create)

	require.ErrorContains(t,
		handler.Validate(
			testContext,
			record,
			&admission.RuntimeObjectInterfaces{},
		),
		`{"kind":"Deny","message":"Denied"}`)

	require.Equal(t, []*unstructured.Unstructured{fakeParams}, passedParams)
	require.NoError(t, tracker.Delete(bindingsGVR, denyBinding.Namespace, denyBinding.Name))
	require.NoError(t, waitForReconcileDeletion(testContext, controller, denyBinding))

	require.ErrorContains(t, handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		record,
		&admission.RuntimeObjectInterfaces{},
	), `{"decision":{"kind":"Deny","message":"configuration error: no bindings found"}`)
}

// Shows that an error is surfaced if a paramSource specified in a binding does
// not actually exist
func TestInvalidParamSourceGVK(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()
	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler)
	passedParams := make(chan *unstructured.Unstructured)

	badPolicy := *denyPolicy
	badPolicy.Spec.ParamKind = &v1alpha1.ParamKind{
		APIVersion: paramsGVK.GroupVersion().String(),
		Kind:       "BadParamKind",
	}

	require.NoError(t, tracker.Create(definitionsGVR, &badPolicy, badPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, &badPolicy))

	err := handler.Validate(
		testContext,
		attributeRecord(nil, denyBinding, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// expect the specific error to be that the param was not found, not that CRD
	// is not existing
	require.ErrorContains(t, err,
		`{"decision":{"kind":"Deny","message":"configuration error: failed to find resource mapping for param source: 'example.com/v1, Kind=BadParamKind'"}`)

	close(passedParams)
	require.Len(t, passedParams, 0)
}

// Shows that an error is surfaced if a param specified in a binding does not
// actually exist
func TestInvalidParamSourceInstanceName(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()
	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
			datalock.Lock()
			passedParams = append(passedParams, params)
			datalock.Unlock()

			// Policy always denies
			return []PolicyDecision{
				{
					Kind:    Deny,
					Message: "Denied",
				},
			}, nil
		}), nil
	}, nil)

	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyPolicy))

	err := handler.Validate(
		testContext,
		attributeRecord(nil, denyBinding, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// expect the specific error to be that the param was not found, not that CRD
	// is not existing
	require.ErrorContains(t, err,
		`{"decision":{"kind":"Deny","message":"configuration error: replicas-test.example.com not found"}`)
	require.Len(t, passedParams, 0)
}

// Shows that a definition with no param source works just fine, and has
// nil params passed to its evaluator.
//
// Also shows that if binding has specified params in this instance then they
// are silently ignored.
func TestEmptyParamSource(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()
	compiler := &fakeCompiler{
		// Match everything by default
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	// Push some fake
	noParamSourcePolicy := *denyPolicy
	noParamSourcePolicy.Spec.ParamKind = nil

	compiler.RegisterDefinition(&noParamSourcePolicy, func(vap *v1alpha1.ValidatingAdmissionPolicy) (Validator, error) {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return ValidatorFunc(func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
			datalock.Lock()
			passedParams = append(passedParams, params)
			datalock.Unlock()

			// Policy always denies
			return []PolicyDecision{
				{
					Kind:    Deny,
					Message: "Denied",
				},
			}, nil
		}), nil
	}, nil)

	require.NoError(t, tracker.Create(definitionsGVR, &noParamSourcePolicy, noParamSourcePolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyPolicy))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, denyBinding, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `{"kind":"Deny","message":"Denied"}`)
	require.Equal(t, 1, numCompiles)
	require.Equal(t, []*unstructured.Unstructured{nil}, passedParams)
}
