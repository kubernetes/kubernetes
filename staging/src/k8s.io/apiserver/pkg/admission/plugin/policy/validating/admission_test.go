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

package validating

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	celgo "github.com/google/cel-go/cel"
	"github.com/stretchr/testify/require"

	admissionv1 "k8s.io/api/admission/v1"
	admissionRegistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/api/admissionregistration/v1beta1"
	v1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/policy/internal/generic"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/warning"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
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

		res.AddKnownTypeWithName(clusterScopedParamsGVK, &unstructured.Unstructured{})
		res.AddKnownTypeWithName(schema.GroupVersionKind{
			Group:   clusterScopedParamsGVK.Group,
			Version: clusterScopedParamsGVK.Version,
			Kind:    clusterScopedParamsGVK.Kind + "List",
		}, &unstructured.UnstructuredList{})

		if err := v1beta1.AddToScheme(res); err != nil {
			panic(err)
		}

		if err := fake.AddToScheme(res); err != nil {
			panic(err)
		}

		return res
	}()

	clusterScopedParamsGVK schema.GroupVersionKind = schema.GroupVersionKind{
		Group:   "example.com",
		Version: "v1",
		Kind:    "ClusterScopedParamsConfig",
	}

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
		res.Add(clusterScopedParamsGVK, meta.RESTScopeRoot)
		res.Add(definitionGVK, meta.RESTScopeRoot)
		res.Add(bindingGVK, meta.RESTScopeRoot)
		res.Add(v1.SchemeGroupVersion.WithKind("ConfigMap"), meta.RESTScopeNamespace)
		return res
	}()

	definitionGVK schema.GroupVersionKind = must3(scheme.ObjectKinds(&v1beta1.ValidatingAdmissionPolicy{}))[0]
	bindingGVK    schema.GroupVersionKind = must3(scheme.ObjectKinds(&v1beta1.ValidatingAdmissionPolicyBinding{}))[0]

	definitionsGVR schema.GroupVersionResource = must(fakeRestMapper.RESTMapping(definitionGVK.GroupKind(), definitionGVK.Version)).Resource
	bindingsGVR    schema.GroupVersionResource = must(fakeRestMapper.RESTMapping(bindingGVK.GroupKind(), bindingGVK.Version)).Resource

	// Common objects
	denyPolicy *v1beta1.ValidatingAdmissionPolicy = &v1beta1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denypolicy.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicySpec{
			ParamKind: &v1beta1.ParamKind{
				APIVersion: paramsGVK.GroupVersion().String(),
				Kind:       paramsGVK.Kind,
			},
			FailurePolicy: ptrTo(v1beta1.Fail),
			Validations: []v1beta1.Validation{
				{
					Expression: "messageId for deny policy",
				},
			},
		},
	}

	fakeParams *unstructured.Unstructured = &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            "replicas-test.example.com",
				"namespace":       "default",
				"resourceVersion": "1",
			},
			"maxReplicas": int64(3),
		},
	}

	denyBinding *v1beta1.ValidatingAdmissionPolicyBinding = &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: denyPolicy.Name,
			ParamRef: &v1beta1.ParamRef{
				Name:      fakeParams.GetName(),
				Namespace: fakeParams.GetNamespace(),
				// fake object tracker does not populate defaults
				ParameterNotFoundAction: ptrTo(v1beta1.DenyAction),
			},
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Deny},
		},
	}
	denyBindingWithNoParamRef *v1beta1.ValidatingAdmissionPolicyBinding = &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        denyPolicy.Name,
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Deny},
		},
	}

	denyBindingWithAudit = &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        denyPolicy.Name,
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Audit},
		},
	}
	denyBindingWithWarn = &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        denyPolicy.Name,
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Warn},
		},
	}
	denyBindingWithAll = &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "1",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        denyPolicy.Name,
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Deny, v1beta1.Warn, v1beta1.Audit},
		},
	}
)

func newParam(name, namespace string, labels map[string]string) *unstructured.Unstructured {
	if len(namespace) == 0 {
		namespace = metav1.NamespaceDefault
	}
	res := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            name,
				"namespace":       namespace,
				"resourceVersion": "1",
			},
		},
	}
	res.SetLabels(labels)
	return res
}

func newClusterScopedParam(name string, labels map[string]string) *unstructured.Unstructured {
	res := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": clusterScopedParamsGVK.GroupVersion().String(),
			"kind":       clusterScopedParamsGVK.Kind,
			"metadata": map[string]interface{}{
				"name":            name,
				"resourceVersion": "1",
			},
		},
	}
	res.SetLabels(labels)
	return res
}

// Interface which has fake compile functionality for use in tests
// So that we can test the controller without pulling in any CEL functionality
type fakeCompiler struct {
	CompileFuncs map[string]func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter
}

var _ cel.FilterCompiler = &fakeCompiler{}

func (f *fakeCompiler) HasSynced() bool {
	return true
}

func (f *fakeCompiler) Compile(
	expressions []cel.ExpressionAccessor,
	options cel.OptionalVariableDeclarations,
	envType environment.Type,
) cel.Filter {
	if len(expressions) > 0 && expressions[0] != nil {
		key := expressions[0].GetExpression()
		if fun, ok := f.CompileFuncs[key]; ok {
			return fun(expressions, options)
		}
	}
	return &fakeFilter{}
}

func (f *fakeCompiler) RegisterDefinition(definition *v1beta1.ValidatingAdmissionPolicy, compileFunc func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter) {
	//Key must be something that we can decipher from the inputs to Validate so using expression which will be passed to validate on the filter
	key := definition.Spec.Validations[0].Expression
	if compileFunc != nil {
		if f.CompileFuncs == nil {
			f.CompileFuncs = make(map[string]func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter)
		}
		f.CompileFuncs[key] = compileFunc
	}
}

var _ cel.ExpressionAccessor = &fakeEvalRequest{}

type fakeEvalRequest struct {
	Key string
}

func (f *fakeEvalRequest) GetExpression() string {
	return ""
}

func (f *fakeEvalRequest) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}

var _ cel.Filter = &fakeFilter{}

type fakeFilter struct {
	keyId string
}

func (f *fakeFilter) ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *admissionv1.AdmissionRequest, inputs cel.OptionalVariableBindings, namespace *v1.Namespace, runtimeCELCostBudget int64) ([]cel.EvaluationResult, int64, error) {
	return []cel.EvaluationResult{}, 0, nil
}

func (f *fakeFilter) CompilationErrors() []error {
	return []error{}
}

var _ Validator = &fakeValidator{}

type fakeValidator struct {
	validationFilter, auditAnnotationFilter, messageFilter *fakeFilter
	ValidateFunc                                           func(ctx context.Context, matchResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult
}

func (f *fakeValidator) RegisterDefinition(definition *v1beta1.ValidatingAdmissionPolicy, validateFunc func(ctx context.Context, matchResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult) {
	//Key must be something that we can decipher from the inputs to Validate so using message which will be on the validationCondition object of evalResult
	var key string
	if len(definition.Spec.Validations) > 0 {
		key = definition.Spec.Validations[0].Expression
	} else {
		key = definition.Spec.AuditAnnotations[0].Key
	}

	if validatorMap == nil {
		validatorMap = make(map[string]*fakeValidator)
	}

	f.ValidateFunc = validateFunc
	validatorMap[key] = f
}

func (f *fakeValidator) Validate(ctx context.Context, matchResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
	return f.ValidateFunc(ctx, matchResource, versionedAttr, versionedParams, namespace, runtimeCELCostBudget, authz)
}

var _ Matcher = &fakeMatcher{}

func (f *fakeMatcher) ValidateInitialization() error {
	return nil
}

func (f *fakeMatcher) GetNamespace(name string) (*v1.Namespace, error) {
	return nil, nil
}

type fakeMatcher struct {
	DefaultMatch         bool
	DefinitionMatchFuncs map[namespacedName]func(*v1beta1.ValidatingAdmissionPolicy, admission.Attributes) bool
	BindingMatchFuncs    map[namespacedName]func(*v1beta1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool
}

func (f *fakeMatcher) RegisterDefinition(definition *v1beta1.ValidatingAdmissionPolicy, matchFunc func(*v1beta1.ValidatingAdmissionPolicy, admission.Attributes) bool) {
	namespace, name := definition.Namespace, definition.Name
	key := namespacedName{
		name:      name,
		namespace: namespace,
	}

	if matchFunc != nil {
		if f.DefinitionMatchFuncs == nil {
			f.DefinitionMatchFuncs = make(map[namespacedName]func(*v1beta1.ValidatingAdmissionPolicy, admission.Attributes) bool)
		}
		f.DefinitionMatchFuncs[key] = matchFunc
	}
}

func (f *fakeMatcher) RegisterBinding(binding *v1beta1.ValidatingAdmissionPolicyBinding, matchFunc func(*v1beta1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool) {
	namespace, name := binding.Namespace, binding.Name
	key := namespacedName{
		name:      name,
		namespace: namespace,
	}

	if matchFunc != nil {
		if f.BindingMatchFuncs == nil {
			f.BindingMatchFuncs = make(map[namespacedName]func(*v1beta1.ValidatingAdmissionPolicyBinding, admission.Attributes) bool)
		}
		f.BindingMatchFuncs[key] = matchFunc
	}
}

// Matches says whether this policy definition matches the provided admission
// resource request
func (f *fakeMatcher) DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1beta1.ValidatingAdmissionPolicy) (bool, schema.GroupVersionResource, schema.GroupVersionKind, error) {
	namespace, name := definition.Namespace, definition.Name
	key := namespacedName{
		name:      name,
		namespace: namespace,
	}
	if fun, ok := f.DefinitionMatchFuncs[key]; ok {
		return fun(definition, a), a.GetResource(), a.GetKind(), nil
	}

	// Default is match everything
	return f.DefaultMatch, a.GetResource(), a.GetKind(), nil
}

// Matches says whether this policy definition matches the provided admission
// resource request
func (f *fakeMatcher) BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, binding *v1beta1.ValidatingAdmissionPolicyBinding) (bool, error) {
	namespace, name := binding.Namespace, binding.Name
	key := namespacedName{
		name:      name,
		namespace: namespace,
	}
	if fun, ok := f.BindingMatchFuncs[key]; ok {
		return fun(binding, a), nil
	}

	// Default is match everything
	return f.DefaultMatch, nil
}

var validatorMap map[string]*fakeValidator

func reset() {
	validatorMap = make(map[string]*fakeValidator)
}

func setupFakeTest(t *testing.T, comp *fakeCompiler, match *fakeMatcher) (plugin admission.ValidationInterface, paramTracker, policyTracker clienttesting.ObjectTracker, controller *celAdmissionController) {
	return setupTestCommon(t, comp, match, true)
}

// Starts CEL admission controller and sets up a plugin configured with it as well
// as object trackers for manipulating the objects available to the system
//
// ParamTracker only knows the gvk `paramGVK`. If in the future we need to
// support multiple types of params this function needs to be augmented
//
// PolicyTracker expects FakePolicyDefinition and FakePolicyBinding types
// !TODO: refactor this test/framework to remove startInformers argument and
// clean up the return args, and in general make it more accessible.
func setupTestCommon(t *testing.T, compiler cel.FilterCompiler, matcher Matcher, shouldStartInformers bool) (plugin admission.ValidationInterface, paramTracker, policyTracker clienttesting.ObjectTracker, controller *celAdmissionController) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	t.Cleanup(testContextCancel)

	fakeAuthorizer := fakeAuthorizer{}
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)

	fakeClient := fake.NewSimpleClientset()
	fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, time.Second)
	featureGate := featuregate.NewFeatureGate()
	err := featureGate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		features.ValidatingAdmissionPolicy: {
			Default: true, PreRelease: featuregate.Alpha}})
	if err != nil {
		t.Fatalf("Unable to add feature gate: %v", err)
	}
	err = featureGate.SetFromMap(map[string]bool{string(features.ValidatingAdmissionPolicy): true})
	if err != nil {
		t.Fatalf("Unable to store flag gate: %v", err)
	}

	plug, err := NewPlugin()
	require.NoError(t, err)

	handler := plug.(*celAdmissionPlugin)
	handler.enabled = true

	genericInitializer := initializer.New(fakeClient, dynamicClient, fakeInformerFactory, fakeAuthorizer, featureGate, testContext.Done())
	genericInitializer.Initialize(handler)
	handler.SetRESTMapper(fakeRestMapper)
	err = admission.ValidateInitialization(handler)
	require.NoError(t, err)
	require.True(t, handler.enabled)

	// Override compiler used by controller for tests
	controller = handler.evaluator.(*celAdmissionController)
	controller.policyController.filterCompiler = compiler
	controller.policyController.newValidator = func(validationFilter cel.Filter, celMatcher matchconditions.Matcher, auditAnnotationFilter, messageFilter cel.Filter, fail *admissionRegistrationv1.FailurePolicyType) Validator {
		f := validationFilter.(*fakeFilter)
		v := validatorMap[f.keyId]
		v.validationFilter = f
		v.messageFilter = f
		v.auditAnnotationFilter = auditAnnotationFilter.(*fakeFilter)
		return v
	}
	controller.policyController.matcher = matcher

	t.Cleanup(func() {
		testContextCancel()
		// wait for informer factory to shutdown
		fakeInformerFactory.Shutdown()
	})

	if !shouldStartInformers {
		return handler, dynamicClient.Tracker(), fakeClient.Tracker(), controller
	}

	// Make sure to start the fake informers
	fakeInformerFactory.Start(testContext.Done())

	// Wait for admission controller to begin its object watches
	// This is because there is a very rare (0.05% on my machine) race doing the
	// initial List+Watch if an object is added after the list, but before the
	// watch it could be missed.
	//
	// This is only due to the fact that NewSimpleClientset above ignores
	// LastSyncResourceVersion on watch calls, so do it does not provide "catch up"
	// which may have been added since the call to list.
	if !cache.WaitForNamedCacheSync("initial sync", testContext.Done(), handler.evaluator.HasSynced) {
		t.Fatal("failed to do perform initial cache sync")
	}

	// WaitForCacheSync only tells us the list was performed.
	// Keep changing an object until it is observable, then remove it

	i := 0

	dummyPolicy := &v1beta1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "dummypolicy.example.com",
			Annotations: map[string]string{
				"myValue": fmt.Sprint(i),
			},
		},
	}

	dummyBinding := &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "dummybinding.example.com",
			Annotations: map[string]string{
				"myValue": fmt.Sprint(i),
			},
		},
	}

	require.NoError(t, fakeClient.Tracker().Create(definitionsGVR, dummyPolicy, dummyPolicy.Namespace))
	require.NoError(t, fakeClient.Tracker().Create(bindingsGVR, dummyBinding, dummyBinding.Namespace))

	wait.PollWithContext(testContext, 100*time.Millisecond, 300*time.Millisecond, func(ctx context.Context) (done bool, err error) {
		defer func() {
			i += 1
		}()

		dummyPolicy.Annotations = map[string]string{
			"myValue": fmt.Sprint(i),
		}
		dummyBinding.Annotations = dummyPolicy.Annotations

		require.NoError(t, fakeClient.Tracker().Update(definitionsGVR, dummyPolicy, dummyPolicy.Namespace))
		require.NoError(t, fakeClient.Tracker().Update(bindingsGVR, dummyBinding, dummyBinding.Namespace))

		if obj, err := controller.getCurrentObject(dummyPolicy); obj == nil || err != nil {
			return false, nil
		}

		if obj, err := controller.getCurrentObject(dummyBinding); obj == nil || err != nil {
			return false, nil
		}

		return true, nil
	})

	require.NoError(t, fakeClient.Tracker().Delete(definitionsGVR, dummyPolicy.Namespace, dummyPolicy.Name))
	require.NoError(t, fakeClient.Tracker().Delete(bindingsGVR, dummyBinding.Namespace, dummyBinding.Name))

	return handler, dynamicClient.Tracker(), fakeClient.Tracker(), controller
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

	c.policyController.mutex.RLock()
	defer c.policyController.mutex.RUnlock()

	switch obj.(type) {
	case *v1beta1.ValidatingAdmissionPolicyBinding:
		nn := getNamespaceName(accessor.GetNamespace(), accessor.GetName())
		info, ok := c.policyController.bindingInfos[nn]
		if !ok {
			return nil, nil
		}

		return info.lastReconciledValue, nil
	case *v1beta1.ValidatingAdmissionPolicy:
		nn := getNamespaceName(accessor.GetNamespace(), accessor.GetName())
		info, ok := c.policyController.definitionInfo[nn]
		if !ok {
			return nil, nil
		}

		return info.lastReconciledValue, nil
	default:
		// If test isn't trying to fetch a policy or binding, assume it is
		// fetching a param
		paramSourceGVK := obj.GetObjectKind().GroupVersionKind()
		paramKind := v1beta1.ParamKind{
			APIVersion: paramSourceGVK.GroupVersion().String(),
			Kind:       paramSourceGVK.Kind,
		}

		var paramInformer generic.Informer[runtime.Object]
		if paramInfo, ok := c.policyController.paramsCRDControllers[paramKind]; ok {
			paramInformer = paramInfo.controller.Informer()
		} else {
			// Treat unknown CRD the same as not found
			return nil, nil
		}

		// Param type. Just check informer for its GVK
		var item runtime.Object
		var err error
		if namespace := accessor.GetNamespace(); len(namespace) > 0 {
			item, err = paramInformer.Namespaced(namespace).Get(accessor.GetName())
		} else {
			item, err = paramInformer.Get(accessor.GetName())
		}

		if err != nil {
			if k8serrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}

		return item, nil
	}
}

// Waits for the given objects to have been the latest reconciled values of
// their gvk/name in the controller
func waitForReconcile(ctx context.Context, controller *celAdmissionController, objects ...runtime.Object) error {
	return wait.PollWithContext(ctx, 100*time.Millisecond, 1*time.Second, func(ctx context.Context) (done bool, err error) {
		defer func() {
			if done {
				// force admission controller to refresh the information it
				// uses for validation now that it is done in the background
				controller.refreshPolicies()
			}
		}()
		for _, obj := range objects {

			objMeta, err := meta.Accessor(obj)
			if err != nil {
				return false, fmt.Errorf("error getting meta accessor for original %T object (%v): %w", obj, obj, err)
			}

			currentValue, err := controller.getCurrentObject(obj)
			if err != nil {
				return false, fmt.Errorf("error getting current object: %w", err)
			} else if currentValue == nil {
				// Object not found, but not an error. Keep waiting.
				klog.Infof("%v not found. keep waiting", objMeta.GetName())
				return false, nil
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
				klog.Infof("%v has RV %v. want RV %v", objMeta.GetName(), objMeta.GetResourceVersion(), objMeta.GetResourceVersion())
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
		defer func() {
			if done {
				// force admission controller to refresh the information it
				// uses for validation now that it is done in the background
				controller.refreshPolicies()
			}
		}()

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
) *FakeAttributes {
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

	return &FakeAttributes{
		Attributes: admission.NewAttributesRecord(
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
		),
	}
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

// //////////////////////////////////////////////////////////////////////////////
// Functionality Tests
// //////////////////////////////////////////////////////////////////////////////

func TestPluginNotReady(t *testing.T) {
	reset()
	compiler := &fakeCompiler{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	// Show that an unstarted informer (or one that has failed its listwatch)
	// will show proper error from plugin
	handler, _, _, _ := setupTestCommon(t, compiler, matcher, false)
	err := handler.Validate(
		context.Background(),
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, "not yet ready to handle request")

	// Show that by now starting the informer, the error is dissipated
	handler, _, _, _ = setupTestCommon(t, compiler, matcher, true)
	err = handler.Validate(
		context.Background(),
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.NoError(t, err)
}

func TestBasicPolicyDefinitionFailure(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	datalock := sync.Mutex{}
	numCompiles := 0

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	warningRecorder := newWarningRecorder()
	warnCtx := warning.WithWarningRecorder(testContext, warningRecorder)
	attr := attributeRecord(nil, fakeParams, admission.Create)
	err := handler.Validate(
		warnCtx,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attr,
		&admission.RuntimeObjectInterfaces{},
	)

	require.Equal(t, 0, warningRecorder.len())

	annotations := attr.GetAnnotations(auditinternal.LevelMetadata)
	require.Equal(t, 0, len(annotations))

	require.ErrorContains(t, err, `Denied`)
}

// Shows that if a definition does not match the input, it will not be used.
// But with a different input it will be used.
func TestDefinitionDoesntMatch(t *testing.T) {
	reset()
	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	matcher.RegisterDefinition(denyPolicy, func(vap *v1beta1.ValidatingAdmissionPolicy, a admission.Attributes) bool {
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
		`Denied`)
	require.Equal(t, numCompiles, 1)
}

func TestReconfigureBinding(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	numCompiles := 0

	fakeParams2 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": paramsGVK.GroupVersion().String(),
			"kind":       paramsGVK.Kind,
			"metadata": map[string]interface{}{
				"name": "replicas-test2.example.com",
				// fake object tracker does not populate missing namespace
				"namespace":       "default",
				"resourceVersion": "2",
			},
			"maxReplicas": int64(35),
		},
	}

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	denyBinding2 := &v1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "denybinding.example.com",
			ResourceVersion: "2",
		},
		Spec: v1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: denyPolicy.Name,
			ParamRef: &v1beta1.ParamRef{
				Name:                    fakeParams2.GetName(),
				Namespace:               fakeParams2.GetNamespace(),
				ParameterNotFoundAction: ptrTo(v1beta1.DenyAction),
			},
			ValidationActions: []v1beta1.ValidationAction{v1beta1.Deny},
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

	err := handler.Validate(
		testContext,
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// Expect validation to fail for first time due to binding unconditionally
	// failing
	require.ErrorContains(t, err, `Denied`, "expect policy validation error")

	// Expect `Compile` only called once
	require.Equal(t, 1, numCompiles, "expect `Compile` to be called only once")

	// Update the tracker to point at different params
	require.NoError(t, tracker.Update(bindingsGVR, denyBinding2, ""))

	// Wait for update to propagate
	// Wait for controller to reconcile given objects
	require.NoError(t, waitForReconcile(testContext, controller, denyBinding2))

	err = handler.Validate(
		testContext,
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, "no params found for policy binding with `Deny` parameterNotFoundAction")

	// Add the missing params
	require.NoError(t, paramTracker.Add(fakeParams2))

	// Wait for update to propagate
	require.NoError(t, waitForReconcile(testContext, controller, fakeParams2))

	// Expect validation to now fail again.
	err = handler.Validate(
		testContext,
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// Expect validation to fail the third time due to validation failure
	require.ErrorContains(t, err, `Denied`, "expected a true policy failure, not a configuration error")
	// require.Equal(t, []*unstructured.Unstructured{fakeParams, fakeParams2}, passedParams, "expected call to `Validate` to cause call to evaluator")
	require.Equal(t, 2, numCompiles, "expect changing binding causes a recompile")
}

// Shows that a policy which is in effect will stop being in effect when removed
func TestRemoveDefinition(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	record := attributeRecord(nil, fakeParams, admission.Create)
	require.ErrorContains(t,
		handler.Validate(
			testContext,
			record,
			&admission.RuntimeObjectInterfaces{},
		),
		`Denied`)

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
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	require.NoError(t, paramTracker.Add(fakeParams))
	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			fakeParams, denyBinding, denyPolicy))

	record := attributeRecord(nil, fakeParams, admission.Create)

	require.ErrorContains(t,
		handler.Validate(
			testContext,
			record,
			&admission.RuntimeObjectInterfaces{},
		),
		`Denied`)

	// require.Equal(t, []*unstructured.Unstructured{fakeParams}, passedParams)
	require.NoError(t, tracker.Delete(bindingsGVR, denyBinding.Namespace, denyBinding.Name))
	require.NoError(t, waitForReconcileDeletion(testContext, controller, denyBinding))
}

// Shows that an error is surfaced if a paramSource specified in a binding does
// not actually exist
func TestInvalidParamSourceGVK(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)
	passedParams := make(chan *unstructured.Unstructured)

	badPolicy := *denyPolicy
	badPolicy.Spec.ParamKind = &v1beta1.ParamKind{
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
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// expect the specific error to be that the param was not found, not that CRD
	// is not existing
	require.ErrorContains(t, err,
		`failed to configure policy: failed to find resource referenced by paramKind: 'example.com/v1, Kind=BadParamKind'`)

	close(passedParams)
	require.Len(t, passedParams, 0)
}

// Shows that an error is surfaced if a param specified in a binding does not
// actually exist
func TestInvalidParamSourceInstanceName(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	passedParams := []*unstructured.Unstructured{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyPolicy))

	err := handler.Validate(
		testContext,
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	// expect the specific error to be that the param was not found, not that CRD
	// is not existing
	require.ErrorContains(t, err,
		"no params found for policy binding with `Deny` parameterNotFoundAction")
	require.Len(t, passedParams, 0)
}

// Show that policy still gets evaluated with `nil` param if paramRef & namespaceParamRef
// are both unset
func TestEmptyParamRef(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	numCompiles := 0

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		// Versioned params must be nil to pass the test
		if versionedParams != nil {
			return ValidateResult{
				Decisions: []PolicyDecision{
					{
						Action: ActionAdmit,
					},
				},
			}
		}
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, denyPolicy, denyPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBindingWithNoParamRef, denyBindingWithNoParamRef.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBindingWithNoParamRef, denyPolicy))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `Denied`)
	require.Equal(t, 1, numCompiles)
}

// Shows that a definition with no param source works just fine, and has
// nil params passed to its evaluator.
//
// Also shows that if binding has specified params in this instance then they
// are silently ignored.
func TestEmptyParamSource(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	datalock := sync.Mutex{}
	numCompiles := 0

	// Push some fake
	noParamSourcePolicy := *denyPolicy
	noParamSourcePolicy.Spec.ParamKind = nil

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		datalock.Lock()
		numCompiles += 1
		datalock.Unlock()

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &noParamSourcePolicy, noParamSourcePolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBindingWithNoParamRef, denyBindingWithNoParamRef.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyPolicy))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns a denial
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `Denied`)
	require.Equal(t, 1, numCompiles)
}

// Shows what happens when multiple policies share one param type, then
// one policy stops using the param. The expectation is the second policy
// keeps behaving normally
func TestMultiplePoliciesSharedParamType(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator1 := &fakeValidator{}
	validator2 := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Use ConfigMap native-typed param
	policy1 := *denyPolicy
	policy1.Name = "denypolicy1.example.com"
	policy1.Spec = v1beta1.ValidatingAdmissionPolicySpec{
		ParamKind: &v1beta1.ParamKind{
			APIVersion: paramsGVK.GroupVersion().String(),
			Kind:       paramsGVK.Kind,
		},
		FailurePolicy: ptrTo(v1beta1.Fail),
		Validations: []v1beta1.Validation{
			{
				Expression: "policy1",
			},
		},
	}

	policy2 := *denyPolicy
	policy2.Name = "denypolicy2.example.com"
	policy2.Spec = v1beta1.ValidatingAdmissionPolicySpec{
		ParamKind: &v1beta1.ParamKind{
			APIVersion: paramsGVK.GroupVersion().String(),
			Kind:       paramsGVK.Kind,
		},
		FailurePolicy: ptrTo(v1beta1.Fail),
		Validations: []v1beta1.Validation{
			{
				Expression: "policy2",
			},
		},
	}

	binding1 := *denyBinding
	binding2 := *denyBinding

	binding1.Name = "denybinding1.example.com"
	binding1.Spec.PolicyName = policy1.Name
	binding2.Name = "denybinding2.example.com"
	binding2.Spec.PolicyName = policy2.Name

	compiles1 := atomic.Int64{}
	evaluations1 := atomic.Int64{}

	compiles2 := atomic.Int64{}
	evaluations2 := atomic.Int64{}

	compiler.RegisterDefinition(&policy1, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		compiles1.Add(1)

		return &fakeFilter{
			keyId: policy1.Spec.Validations[0].Expression,
		}
	})

	validator1.RegisterDefinition(&policy1, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		evaluations1.Add(1)
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action: ActionAdmit,
				},
			},
		}
	})

	compiler.RegisterDefinition(&policy2, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		compiles2.Add(1)

		return &fakeFilter{
			keyId: policy2.Spec.Validations[0].Expression,
		}
	})

	validator2.RegisterDefinition(&policy2, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		evaluations2.Add(1)
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Policy2Denied",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &policy1, policy1.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, &binding1, binding1.Namespace))
	require.NoError(t, paramTracker.Add(fakeParams))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			&binding1, &policy1, fakeParams))

	// Make sure policy 1 is created and bound to the params type first
	require.NoError(t, tracker.Create(definitionsGVR, &policy2, policy2.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, &binding2, binding2.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			&binding1, &binding2, &policy1, &policy2, fakeParams))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns admit meaning the params
		// passed was a configmap
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `Denied`)
	require.EqualValues(t, 1, compiles1.Load())
	require.EqualValues(t, 1, evaluations1.Load())
	require.EqualValues(t, 1, compiles2.Load())
	require.EqualValues(t, 1, evaluations2.Load())

	// Remove param type from policy1
	// Show that policy2 evaluator is still being passed the configmaps
	policy1.Spec.ParamKind = nil
	policy1.ResourceVersion = "2"

	binding1.Spec.ParamRef = nil
	binding1.ResourceVersion = "2"

	require.NoError(t, tracker.Update(definitionsGVR, &policy1, policy1.Namespace))
	require.NoError(t, tracker.Update(bindingsGVR, &binding1, binding1.Namespace))
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			&binding1, &policy1))

	err = handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns admit meaning the params
		// passed was a configmap
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, `Policy2Denied`)
	require.EqualValues(t, 2, compiles1.Load())
	require.EqualValues(t, 2, evaluations1.Load())
	require.EqualValues(t, 1, compiles2.Load())
	require.EqualValues(t, 2, evaluations2.Load())
}

// Shows that we can refer to native-typed params just fine
// (as opposed to CRD params)
func TestNativeTypeParam(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	compiles := atomic.Int64{}
	evaluations := atomic.Int64{}

	// Use ConfigMap native-typed param
	nativeTypeParamPolicy := *denyPolicy
	nativeTypeParamPolicy.Spec.ParamKind = &v1beta1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}

	compiler.RegisterDefinition(&nativeTypeParamPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		compiles.Add(1)

		return &fakeFilter{
			keyId: nativeTypeParamPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(&nativeTypeParamPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		evaluations.Add(1)
		if _, ok := versionedParams.(*v1.ConfigMap); ok {
			return ValidateResult{
				Decisions: []PolicyDecision{
					{
						Action:  ActionDeny,
						Message: "correct type",
					},
				},
			}
		}
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Incorrect param type",
				},
			},
		}
	})

	configMapParam := &v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "replicas-test.example.com",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Data: map[string]string{
			"coolkey": "coolvalue",
		},
	}
	require.NoError(t, tracker.Create(definitionsGVR, &nativeTypeParamPolicy, nativeTypeParamPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))
	require.NoError(t, tracker.Add(configMapParam))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyPolicy, configMapParam))

	err := handler.Validate(
		testContext,
		// Object is irrelevant/unchecked for this test. Just test that
		// the evaluator is executed, and returns admit meaning the params
		// passed was a configmap
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, "correct type")
	require.EqualValues(t, 1, compiles.Load())
	require.EqualValues(t, 1, evaluations.Load())
}

func TestAuditValidationAction(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Push some fake
	noParamSourcePolicy := *denyPolicy
	noParamSourcePolicy.Spec.ParamKind = nil

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "I'm sorry Dave",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &noParamSourcePolicy, noParamSourcePolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBindingWithAudit, denyBindingWithAudit.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBindingWithAudit, &noParamSourcePolicy))
	attr := attributeRecord(nil, fakeParams, admission.Create)
	warningRecorder := newWarningRecorder()
	warnCtx := warning.WithWarningRecorder(testContext, warningRecorder)
	err := handler.Validate(
		warnCtx,
		attr,
		&admission.RuntimeObjectInterfaces{},
	)

	require.Equal(t, 0, warningRecorder.len())

	annotations := attr.GetAnnotations(auditinternal.LevelMetadata)
	require.Equal(t, 1, len(annotations))
	valueJson, ok := annotations["validation.policy.admission.k8s.io/validation_failure"]
	require.True(t, ok)
	var value []validationFailureValue
	jsonErr := utiljson.Unmarshal([]byte(valueJson), &value)
	require.NoError(t, jsonErr)
	expected := []validationFailureValue{{
		ExpressionIndex:   0,
		Message:           "I'm sorry Dave",
		ValidationActions: []v1beta1.ValidationAction{v1beta1.Audit},
		Binding:           "denybinding.example.com",
		Policy:            noParamSourcePolicy.Name,
	}}
	require.Equal(t, expected, value)

	require.NoError(t, err)
}

func TestWarnValidationAction(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Push some fake
	noParamSourcePolicy := *denyPolicy
	noParamSourcePolicy.Spec.ParamKind = nil

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "I'm sorry Dave",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &noParamSourcePolicy, noParamSourcePolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBindingWithWarn, denyBindingWithWarn.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBindingWithWarn, &noParamSourcePolicy))
	attr := attributeRecord(nil, fakeParams, admission.Create)
	warningRecorder := newWarningRecorder()
	warnCtx := warning.WithWarningRecorder(testContext, warningRecorder)
	err := handler.Validate(
		warnCtx,
		attr,
		&admission.RuntimeObjectInterfaces{},
	)

	require.Equal(t, 1, warningRecorder.len())
	require.True(t, warningRecorder.hasWarning("Validation failed for ValidatingAdmissionPolicy 'denypolicy.example.com' with binding 'denybinding.example.com': I'm sorry Dave"))

	annotations := attr.GetAnnotations(auditinternal.LevelMetadata)
	require.Equal(t, 0, len(annotations))

	require.NoError(t, err)
}

func TestAllValidationActions(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Push some fake
	noParamSourcePolicy := *denyPolicy
	noParamSourcePolicy.Spec.ParamKind = nil

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "I'm sorry Dave",
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &noParamSourcePolicy, noParamSourcePolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBindingWithAll, denyBindingWithAll.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBindingWithAll, &noParamSourcePolicy))
	attr := attributeRecord(nil, fakeParams, admission.Create)
	warningRecorder := newWarningRecorder()
	warnCtx := warning.WithWarningRecorder(testContext, warningRecorder)
	err := handler.Validate(
		warnCtx,
		attr,
		&admission.RuntimeObjectInterfaces{},
	)

	require.Equal(t, 1, warningRecorder.len())
	require.True(t, warningRecorder.hasWarning("Validation failed for ValidatingAdmissionPolicy 'denypolicy.example.com' with binding 'denybinding.example.com': I'm sorry Dave"))

	annotations := attr.GetAnnotations(auditinternal.LevelMetadata)
	require.Equal(t, 1, len(annotations))
	valueJson, ok := annotations["validation.policy.admission.k8s.io/validation_failure"]
	require.True(t, ok)
	var value []validationFailureValue
	jsonErr := utiljson.Unmarshal([]byte(valueJson), &value)
	require.NoError(t, jsonErr)
	expected := []validationFailureValue{{
		ExpressionIndex:   0,
		Message:           "I'm sorry Dave",
		ValidationActions: []v1beta1.ValidationAction{v1beta1.Deny, v1beta1.Warn, v1beta1.Audit},
		Binding:           "denybinding.example.com",
		Policy:            noParamSourcePolicy.Name,
	}}
	require.Equal(t, expected, value)

	require.ErrorContains(t, err, "I'm sorry Dave")
}

func TestNamespaceParamRefName(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	compiles := atomic.Int64{}
	evaluations := atomic.Int64{}

	// Use ConfigMap native-typed param
	nativeTypeParamPolicy := *denyPolicy
	nativeTypeParamPolicy.Spec.ParamKind = &v1beta1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}

	namespaceParamBinding := *denyBinding
	namespaceParamBinding.Spec.ParamRef = &v1beta1.ParamRef{
		Name: "replicas-test.example.com",
	}

	compiler.RegisterDefinition(&nativeTypeParamPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		compiles.Add(1)

		return &fakeFilter{
			keyId: nativeTypeParamPolicy.Spec.Validations[0].Expression,
		}
	})

	lock := sync.Mutex{}
	observedParamNamespaces := []string{}
	validator.RegisterDefinition(&nativeTypeParamPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		lock.Lock()
		defer lock.Unlock()

		evaluations.Add(1)
		if p, ok := versionedParams.(*v1.ConfigMap); ok {
			observedParamNamespaces = append(observedParamNamespaces, p.Namespace)
			return ValidateResult{
				Decisions: []PolicyDecision{
					{
						Action:  ActionDeny,
						Message: "correct type",
					},
				},
			}
		}
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Incorrect param type",
				},
			},
		}
	})

	configMapParam := &v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "replicas-test.example.com",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Data: map[string]string{
			"coolkey": "default",
		},
	}
	configMapParam2 := &v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "replicas-test.example.com",
			Namespace:       "mynamespace",
			ResourceVersion: "1",
		},
		Data: map[string]string{
			"coolkey": "mynamespace",
		},
	}
	configMapParam3 := &v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "replicas-test.example.com",
			Namespace:       "othernamespace",
			ResourceVersion: "1",
		},
		Data: map[string]string{
			"coolkey": "othernamespace",
		},
	}
	require.NoError(t, tracker.Create(definitionsGVR, &nativeTypeParamPolicy, nativeTypeParamPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, &namespaceParamBinding, namespaceParamBinding.Namespace))
	require.NoError(t, tracker.Add(configMapParam))
	require.NoError(t, tracker.Add(configMapParam2))
	require.NoError(t, tracker.Add(configMapParam3))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			&namespaceParamBinding, &nativeTypeParamPolicy, configMapParam, configMapParam2, configMapParam3))

	// Object is irrelevant/unchecked for this test. Just test that
	// the evaluator is executed with correct namespace, and returns admit
	// meaning the params passed was a configmap
	err := handler.Validate(
		testContext,
		attributeRecord(nil, configMapParam, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	func() {
		lock.Lock()
		defer lock.Unlock()
		require.ErrorContains(t, err, "correct type")
		require.EqualValues(t, 1, compiles.Load())
		require.EqualValues(t, 1, evaluations.Load())
	}()

	err = handler.Validate(
		testContext,
		attributeRecord(nil, configMapParam2, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	func() {
		lock.Lock()
		defer lock.Unlock()
		require.ErrorContains(t, err, "correct type")
		require.EqualValues(t, 1, compiles.Load())
		require.EqualValues(t, 2, evaluations.Load())
	}()

	err = handler.Validate(
		testContext,
		attributeRecord(nil, configMapParam3, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	func() {
		lock.Lock()
		defer lock.Unlock()
		require.ErrorContains(t, err, "correct type")
		require.EqualValues(t, 1, compiles.Load())
		require.EqualValues(t, 3, evaluations.Load())
	}()

	err = handler.Validate(
		testContext,
		attributeRecord(nil, configMapParam, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	func() {
		lock.Lock()
		defer lock.Unlock()
		require.ErrorContains(t, err, "correct type")
		require.EqualValues(t, []string{"default", "mynamespace", "othernamespace", "default"}, observedParamNamespaces)
		require.EqualValues(t, 1, compiles.Load())
		require.EqualValues(t, 4, evaluations.Load())
	}()
}

func TestParamRef(t *testing.T) {
	for _, paramIsClusterScoped := range []bool{false, true} {
		for _, nameIsSet := range []bool{false, true} {
			for _, namespaceIsSet := range []bool{false, true} {
				if paramIsClusterScoped && namespaceIsSet {
					// Skip invalid configuration
					continue
				}

				for _, selectorIsSet := range []bool{false, true} {
					if selectorIsSet && nameIsSet {
						// SKip invalid configuration
						continue
					}

					for _, denyNotFound := range []bool{false, true} {

						name := "ParamRef"

						if paramIsClusterScoped {
							name = "ClusterScoped" + name
						}

						if nameIsSet {
							name = name + "WithName"
						} else if selectorIsSet {
							name = name + "WithLabelSelector"
						} else {
							name = name + "WithEverythingSelector"
						}

						if namespaceIsSet {
							name = name + "WithNamespace"
						}

						if denyNotFound {
							name = name + "DenyNotFound"
						} else {
							name = name + "AllowNotFound"
						}

						t.Run(name, func(t *testing.T) {
							// Test creating a policy with a cluster or namesapce-scoped param
							// and binding with the provided configuration. Test will ensure
							// that the provided configuration is capable of matching
							// params as expected, and not matching params when not expected.
							// Also ensures the NotFound setting works as expected with this particular
							// configuration of ParamRef when all the previously
							// matched params are deleted.
							testParamRefCase(t, paramIsClusterScoped, nameIsSet, namespaceIsSet, selectorIsSet, denyNotFound)
						})
					}
				}
			}
		}
	}
}

// testParamRefCase constructs a ParamRef and policy with appropriate ParamKind
// for the given parameters, then constructs a scenario with several matching/non-matching params
// of varying names, namespaces, labels.
//
// Test then selects subset of params that should match provided configuration
// and ensuers those params are the only ones used.
//
// Also ensures NotFound action is enforced correctly by deleting all found
// params and ensuring the Action is used.
//
// This test is not meant to test every possible scenario of matching/not matching:
// only that each ParamRef CAN be evaluated correctly for both cluster scoped
// and namespace-scoped request kinds, and that the failure action is correctly
// applied.
func testParamRefCase(t *testing.T, paramIsClusterScoped, nameIsSet, namespaceIsSet, selectorIsSet, denyNotFound bool) {
	// Create a cluster scoped and a namespace scoped CRD
	policy := *denyPolicy
	binding := *denyBinding
	binding.Spec.ParamRef = &v1beta1.ParamRef{}
	paramRef := binding.Spec.ParamRef

	shouldErrorOnClusterScopedRequests := !namespaceIsSet && !paramIsClusterScoped

	matchingParamName := "replicas-test.example.com"
	matchingNamespace := "mynamespace"
	nonMatchingNamespace := "othernamespace"

	matchingLabels := labels.Set{"doesitmatch": "yes"}
	nonmatchingLabels := labels.Set{"doesitmatch": "no"}
	otherNonmatchingLabels := labels.Set{"notaffiliated": "no"}

	if paramIsClusterScoped {
		policy.Spec.ParamKind = &v1beta1.ParamKind{
			APIVersion: clusterScopedParamsGVK.GroupVersion().String(),
			Kind:       clusterScopedParamsGVK.Kind,
		}
	} else {
		policy.Spec.ParamKind = &v1beta1.ParamKind{
			APIVersion: paramsGVK.GroupVersion().String(),
			Kind:       paramsGVK.Kind,
		}
	}

	if nameIsSet {
		paramRef.Name = matchingParamName
	} else if selectorIsSet {
		paramRef.Selector = metav1.SetAsLabelSelector(matchingLabels)
	} else {
		paramRef.Selector = &metav1.LabelSelector{}
	}

	if namespaceIsSet {
		paramRef.Namespace = matchingNamespace
	}

	if denyNotFound {
		paramRef.ParameterNotFoundAction = ptrTo(v1beta1.DenyAction)
	} else {
		paramRef.ParameterNotFoundAction = ptrTo(v1beta1.AllowAction)
	}

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}

	var matchedParams []runtime.Object
	paramLock := sync.Mutex{}
	observeParam := func(p runtime.Object) {
		paramLock.Lock()
		defer paramLock.Unlock()
		matchedParams = append(matchedParams, p)
	}
	getAndResetObservedParams := func() []runtime.Object {
		paramLock.Lock()
		defer paramLock.Unlock()
		oldParams := matchedParams
		matchedParams = nil
		return oldParams
	}

	compiler.RegisterDefinition(&policy, func(ea []cel.ExpressionAccessor, ovd cel.OptionalVariableDeclarations) cel.Filter {
		return &fakeFilter{
			keyId: policy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(&policy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		observeParam(versionedParams)
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: "Denied by policy",
				},
			},
		}
	})

	handler, paramTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Create library of params to try to fool the controller
	params := []*unstructured.Unstructured{
		newParam(matchingParamName, v1.NamespaceDefault, nonmatchingLabels),
		newParam(matchingParamName, matchingNamespace, nonmatchingLabels),
		newParam(matchingParamName, nonMatchingNamespace, nonmatchingLabels),

		newParam(matchingParamName+"1", v1.NamespaceDefault, matchingLabels),
		newParam(matchingParamName+"1", matchingNamespace, matchingLabels),
		newParam(matchingParamName+"1", nonMatchingNamespace, matchingLabels),

		newParam(matchingParamName+"2", v1.NamespaceDefault, otherNonmatchingLabels),
		newParam(matchingParamName+"2", matchingNamespace, otherNonmatchingLabels),
		newParam(matchingParamName+"2", nonMatchingNamespace, otherNonmatchingLabels),

		newParam(matchingParamName+"3", v1.NamespaceDefault, otherNonmatchingLabels),
		newParam(matchingParamName+"3", matchingNamespace, matchingLabels),
		newParam(matchingParamName+"3", nonMatchingNamespace, matchingLabels),

		newClusterScopedParam(matchingParamName, matchingLabels),
		newClusterScopedParam(matchingParamName+"1", nonmatchingLabels),
		newClusterScopedParam(matchingParamName+"2", otherNonmatchingLabels),
		newClusterScopedParam(matchingParamName+"3", matchingLabels),
		newClusterScopedParam(matchingParamName+"4", nonmatchingLabels),
		newClusterScopedParam(matchingParamName+"5", otherNonmatchingLabels),
	}

	require.NoError(t, tracker.Create(definitionsGVR, &policy, policy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, &binding, binding.Namespace))
	require.NoError(t, waitForReconcile(context.TODO(), controller, &policy, &binding))

	for _, p := range params {
		paramTracker.Add(p)
	}

	namespacedRequestObject := newParam("some param", nonMatchingNamespace, nil)
	clusterScopedRequestObject := newClusterScopedParam("other param", nil)

	// Validate a namespaced object, and verify that the params being validated
	// are the ones we would expect
	var expectedParamsForNamespacedRequest []*unstructured.Unstructured
	for _, p := range params {
		if p.GetAPIVersion() != policy.Spec.ParamKind.APIVersion || p.GetKind() != policy.Spec.ParamKind.Kind {
			continue
		} else if len(paramRef.Name) > 0 && p.GetName() != paramRef.Name {
			continue
		} else if len(paramRef.Namespace) > 0 && p.GetNamespace() != paramRef.Namespace {
			continue
		}

		if !paramIsClusterScoped {
			if len(paramRef.Namespace) == 0 && p.GetNamespace() != namespacedRequestObject.GetNamespace() {
				continue
			}
		}

		if paramRef.Selector != nil {
			ls := p.GetLabels()
			matched := true

			for k, v := range paramRef.Selector.MatchLabels {
				if l, hasLabel := ls[k]; !hasLabel {
					matched = false
					break
				} else if l != v {
					matched = false
					break
				}
			}

			// Empty selector matches everything
			if len(paramRef.Selector.MatchExpressions) == 0 && len(paramRef.Selector.MatchLabels) == 0 {
				matched = true
			}

			if !matched {
				continue
			}
		}

		expectedParamsForNamespacedRequest = append(expectedParamsForNamespacedRequest, p)
		require.NoError(t, waitForReconcile(context.TODO(), controller, p))
	}
	require.NotEmpty(t, expectedParamsForNamespacedRequest, "all test cases should match at least one param")
	require.ErrorContains(t, handler.Validate(context.TODO(), attributeRecord(nil, namespacedRequestObject, admission.Create), &admission.RuntimeObjectInterfaces{}), "Denied by policy")
	require.ElementsMatch(t, expectedParamsForNamespacedRequest, getAndResetObservedParams(), "should exactly match expected params")

	// Validate a cluster-scoped object, and verify that the params being validated
	// are the ones we would expect
	var expectedParamsForClusterScopedRequest []*unstructured.Unstructured
	for _, p := range params {
		if shouldErrorOnClusterScopedRequests {
			continue
		} else if p.GetAPIVersion() != policy.Spec.ParamKind.APIVersion || p.GetKind() != policy.Spec.ParamKind.Kind {
			continue
		} else if len(paramRef.Name) > 0 && p.GetName() != paramRef.Name {
			continue
		} else if len(paramRef.Namespace) > 0 && p.GetNamespace() != paramRef.Namespace {
			continue
		} else if !paramIsClusterScoped && len(paramRef.Namespace) == 0 && p.GetNamespace() != v1.NamespaceDefault {
			continue
		}

		if paramRef.Selector != nil {
			ls := p.GetLabels()
			matched := true
			for k, v := range paramRef.Selector.MatchLabels {
				if l, hasLabel := ls[k]; !hasLabel {
					matched = false
					break
				} else if l != v {
					matched = false
					break
				}
			}

			// Empty selector matches everything
			if len(paramRef.Selector.MatchExpressions) == 0 && len(paramRef.Selector.MatchLabels) == 0 {
				matched = true
			}

			if !matched {
				continue
			}
		}

		expectedParamsForClusterScopedRequest = append(expectedParamsForClusterScopedRequest, p)
		require.NoError(t, waitForReconcile(context.TODO(), controller, p))
	}

	err := handler.Validate(context.TODO(), attributeRecord(nil, clusterScopedRequestObject, admission.Create), &admission.RuntimeObjectInterfaces{})
	if shouldErrorOnClusterScopedRequests {
		// Cannot validate cliuster-scoped resources against a paramRef that sets namespace
		require.ErrorContains(t, err, "failed to configure binding: cannot use namespaced paramRef in policy binding that matches cluster-scoped resources")
	} else {
		require.NotEmpty(t, expectedParamsForClusterScopedRequest, "all test cases should match at least one param")
		require.ErrorContains(t, err, "Denied by policy")
	}
	require.ElementsMatch(t, expectedParamsForClusterScopedRequest, getAndResetObservedParams(), "should exactly match expected params")

	// Remove all params matched by namespaced, and cluster-scoped validation.
	// Validate again to make sure NotFoundAction is respected
	var deleted []runtime.Object
	for _, p := range expectedParamsForNamespacedRequest {
		if paramIsClusterScoped {
			require.NoError(t, paramTracker.Delete(paramsGVK.GroupVersion().WithResource("clusterscopedparamsconfigs"), p.GetNamespace(), p.GetName()))
		} else {
			require.NoError(t, paramTracker.Delete(paramsGVK.GroupVersion().WithResource("paramsconfigs"), p.GetNamespace(), p.GetName()))
		}
		deleted = append(deleted, p)
	}

	for _, p := range expectedParamsForClusterScopedRequest {
		// Tracker.Delete docs says it wont raise error for not found, but its implmenetation
		// pretty plainly does...
		rsrsc := "paramsconfigs"
		if paramIsClusterScoped {
			rsrsc = "clusterscopedparamsconfigs"
		}
		if err := paramTracker.Delete(paramsGVK.GroupVersion().WithResource(rsrsc), p.GetNamespace(), p.GetName()); err != nil && !k8serrors.IsNotFound(err) {
			require.NoError(t, err)
			deleted = append(deleted, p)
		}
	}
	require.NoError(t, waitForReconcileDeletion(context.TODO(), controller, deleted...))

	controller.refreshPolicies()

	// Check that NotFound is working correctly for both namespaeed & non-namespaced
	// request object
	err = handler.Validate(context.TODO(), attributeRecord(nil, namespacedRequestObject, admission.Create), &admission.RuntimeObjectInterfaces{})
	if denyNotFound {
		require.ErrorContains(t, err, "no params found for policy binding with `Deny` parameterNotFoundAction")
	} else {
		require.NoError(t, err, "Allow not found expects no error when no params found. Policy should have been skipped")
	}
	require.Empty(t, getAndResetObservedParams(), "policy should not have been evaluated")

	err = handler.Validate(context.TODO(), attributeRecord(nil, clusterScopedRequestObject, admission.Create), &admission.RuntimeObjectInterfaces{})
	if shouldErrorOnClusterScopedRequests {
		require.ErrorContains(t, err, "failed to configure binding: cannot use namespaced paramRef in policy binding that matches cluster-scoped resources")

	} else if denyNotFound {
		require.ErrorContains(t, err, "no params found for policy binding with `Deny` parameterNotFoundAction")
	} else {
		require.NoError(t, err, "Allow not found expects no error when no params found. Policy should have been skipped")
	}
	require.Empty(t, getAndResetObservedParams(), "policy should not have been evaluated")
}

// If the ParamKind is ClusterScoped, and namespace param is used.
// This is a Configuration Error of the policy
func TestNamespaceParamRefClusterScopedParamError(t *testing.T) {
	reset()
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, _, tracker, controller := setupFakeTest(t, compiler, matcher)

	compiles := atomic.Int64{}
	evaluations := atomic.Int64{}

	// Use ValidatingAdmissionPolicy for param type since it is cluster-scoped
	nativeTypeParamPolicy := *denyPolicy
	nativeTypeParamPolicy.Spec.ParamKind = &v1beta1.ParamKind{
		APIVersion: "admissionregistration.k8s.io/v1beta1",
		Kind:       "ValidatingAdmissionPolicy",
	}

	namespaceParamBinding := *denyBinding
	namespaceParamBinding.Spec.ParamRef = &v1beta1.ParamRef{
		Name:      "other-param-to-use-with-no-label.example.com",
		Namespace: "mynamespace",
	}

	compiler.RegisterDefinition(&nativeTypeParamPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {
		compiles.Add(1)

		return &fakeFilter{
			keyId: nativeTypeParamPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(&nativeTypeParamPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		evaluations.Add(1)
		if _, ok := versionedParams.(*v1beta1.ValidatingAdmissionPolicy); ok {
			return ValidateResult{
				Decisions: []PolicyDecision{
					{
						Action:  ActionAdmit,
						Message: "correct type",
					},
				},
			}
		}
		return ValidateResult{
			Decisions: []PolicyDecision{
				{
					Action:  ActionDeny,
					Message: fmt.Sprintf("Incorrect param type %T", versionedParams),
				},
			},
		}
	})

	require.NoError(t, tracker.Create(definitionsGVR, &nativeTypeParamPolicy, nativeTypeParamPolicy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, &namespaceParamBinding, namespaceParamBinding.Namespace))
	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			&namespaceParamBinding, &nativeTypeParamPolicy))

	// Object is irrelevant/unchecked for this test. Just test that
	// the evaluator is executed with correct namespace, and returns admit
	// meaning the params passed was a configmap
	err := handler.Validate(
		testContext,
		attributeRecord(nil, fakeParams, admission.Create),
		&admission.RuntimeObjectInterfaces{},
	)

	require.ErrorContains(t, err, "paramRef.namespace must not be provided for a cluster-scoped `paramKind`")
	require.EqualValues(t, 1, compiles.Load())
	require.EqualValues(t, 0, evaluations.Load())
}

func TestAuditAnnotations(t *testing.T) {
	testContext, testContextCancel := context.WithCancel(context.Background())
	defer testContextCancel()

	compiler := &fakeCompiler{}
	validator := &fakeValidator{}
	matcher := &fakeMatcher{
		DefaultMatch: true,
	}
	handler, paramsTracker, tracker, controller := setupFakeTest(t, compiler, matcher)

	// Push some fake
	policy := *denyPolicy

	compiler.RegisterDefinition(denyPolicy, func([]cel.ExpressionAccessor, cel.OptionalVariableDeclarations) cel.Filter {

		return &fakeFilter{
			keyId: denyPolicy.Spec.Validations[0].Expression,
		}
	})

	validator.RegisterDefinition(denyPolicy, func(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, versionedParams runtime.Object, namespace *v1.Namespace, runtimeCELCostBudget int64, authz authorizer.Authorizer) ValidateResult {
		o, err := meta.Accessor(versionedParams)
		if err != nil {
			t.Fatal(err)
		}
		exampleValue := "normal-value"
		if o.GetName() == "replicas-test2.example.com" {
			exampleValue = "special-value"
		}
		return ValidateResult{
			AuditAnnotations: []PolicyAuditAnnotation{
				{
					Key:    "example-key",
					Value:  exampleValue,
					Action: AuditAnnotationActionPublish,
				},
				{
					Key:    "excluded-key",
					Value:  "excluded-value",
					Action: AuditAnnotationActionExclude,
				},
				{
					Key:    "error-key",
					Action: AuditAnnotationActionError,
					Error:  "example error",
				},
			},
		}
	})

	fakeParams2 := fakeParams.DeepCopy()
	fakeParams2.SetName("replicas-test2.example.com")
	denyBinding2 := denyBinding.DeepCopy()
	denyBinding2.SetName("denybinding2.example.com")
	denyBinding2.Spec.ParamRef.Name = fakeParams2.GetName()

	fakeParams3 := fakeParams.DeepCopy()
	fakeParams3.SetName("replicas-test3.example.com")
	denyBinding3 := denyBinding.DeepCopy()
	denyBinding3.SetName("denybinding3.example.com")
	denyBinding3.Spec.ParamRef.Name = fakeParams3.GetName()

	require.NoError(t, paramsTracker.Add(fakeParams))
	require.NoError(t, paramsTracker.Add(fakeParams2))
	require.NoError(t, paramsTracker.Add(fakeParams3))
	require.NoError(t, tracker.Create(definitionsGVR, &policy, policy.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding, denyBinding.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding2, denyBinding2.Namespace))
	require.NoError(t, tracker.Create(bindingsGVR, denyBinding3, denyBinding3.Namespace))

	// Wait for controller to reconcile given objects
	require.NoError(t,
		waitForReconcile(
			testContext, controller,
			denyBinding, denyBinding2, denyBinding3, denyPolicy, fakeParams, fakeParams2, fakeParams3))
	attr := attributeRecord(nil, fakeParams, admission.Create)
	err := handler.Validate(
		testContext,
		attr,
		&admission.RuntimeObjectInterfaces{},
	)

	annotations := attr.GetAnnotations(auditinternal.LevelMetadata)
	require.Equal(t, 1, len(annotations))
	value := annotations[policy.Name+"/example-key"]
	parts := strings.Split(value, ", ")
	require.Equal(t, 2, len(parts))
	require.Contains(t, parts, "normal-value", "special-value")

	require.ErrorContains(t, err, "example error")
}

// FakeAttributes decorates admission.Attributes. It's used to trace the added annotations.
type FakeAttributes struct {
	admission.Attributes
	annotations map[string]string
	mutex       sync.Mutex
}

// AddAnnotation adds an annotation key value pair to FakeAttributes
func (f *FakeAttributes) AddAnnotation(k, v string) error {
	return f.AddAnnotationWithLevel(k, v, auditinternal.LevelMetadata)
}

// AddAnnotationWithLevel adds an annotation key value pair to FakeAttributes
func (f *FakeAttributes) AddAnnotationWithLevel(k, v string, _ auditinternal.Level) error {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	if err := f.Attributes.AddAnnotation(k, v); err != nil {
		return err
	}
	if f.annotations == nil {
		f.annotations = make(map[string]string)
	}
	f.annotations[k] = v
	return nil
}

// GetAnnotations reads annotations from FakeAttributes
func (f *FakeAttributes) GetAnnotations(_ auditinternal.Level) map[string]string {
	f.mutex.Lock()
	defer f.mutex.Unlock()
	annotations := make(map[string]string, len(f.annotations))
	for k, v := range f.annotations {
		annotations[k] = v
	}
	return annotations
}

type warningRecorder struct {
	sync.Mutex
	warnings sets.Set[string]
}

func newWarningRecorder() *warningRecorder {
	return &warningRecorder{warnings: sets.New[string]()}
}

func (r *warningRecorder) AddWarning(_, text string) {
	r.Lock()
	defer r.Unlock()
	r.warnings.Insert(text)
	return
}

func (r *warningRecorder) hasWarning(text string) bool {
	r.Lock()
	defer r.Unlock()
	return r.warnings.Has(text)
}

func (r *warningRecorder) len() int {
	r.Lock()
	defer r.Unlock()
	return len(r.warnings)
}

type fakeAuthorizer struct{}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "", nil
}
