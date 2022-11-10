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

package validatingadmissionpolicy

import (
	"fmt"
	"reflect"
	"strings"
	"time"

	celtypes "github.com/google/cel-go/common/types"
	"github.com/google/cel-go/interpreter"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/api/admissionregistration/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

var _ ValidatorCompiler = &CELValidatorCompiler{}
var _ matching.MatchCriteria = &matchCriteria{}

type matchCriteria struct {
	constraints *v1alpha1.MatchResources
}

// GetParsedNamespaceSelector returns the converted LabelSelector which implements labels.Selector
func (m *matchCriteria) GetParsedNamespaceSelector() (labels.Selector, error) {
	return metav1.LabelSelectorAsSelector(m.constraints.NamespaceSelector)
}

// GetParsedObjectSelector returns the converted LabelSelector which implements labels.Selector
func (m *matchCriteria) GetParsedObjectSelector() (labels.Selector, error) {
	return metav1.LabelSelectorAsSelector(m.constraints.ObjectSelector)
}

// GetMatchResources returns the matchConstraints
func (m *matchCriteria) GetMatchResources() v1alpha1.MatchResources {
	return *m.constraints
}

// CELValidatorCompiler implement the interface ValidatorCompiler.
type CELValidatorCompiler struct {
	Matcher *matching.Matcher
}

// DefinitionMatches returns whether this ValidatingAdmissionPolicy matches the provided admission resource request
func (c *CELValidatorCompiler) DefinitionMatches(a admission.Attributes, o admission.ObjectInterfaces, definition *v1alpha1.ValidatingAdmissionPolicy) (bool, schema.GroupVersionKind, error) {
	criteria := matchCriteria{constraints: definition.Spec.MatchConstraints}
	return c.Matcher.Matches(a, o, &criteria)
}

// BindingMatches returns whether this ValidatingAdmissionPolicyBinding matches the provided admission resource request
func (c *CELValidatorCompiler) BindingMatches(a admission.Attributes, o admission.ObjectInterfaces, binding *v1alpha1.ValidatingAdmissionPolicyBinding) (bool, error) {
	if binding.Spec.MatchResources == nil {
		return true, nil
	}
	criteria := matchCriteria{constraints: binding.Spec.MatchResources}
	isMatch, _, err := c.Matcher.Matches(a, o, &criteria)
	return isMatch, err
}

// ValidateInitialization checks if Matcher is initialized.
func (c *CELValidatorCompiler) ValidateInitialization() error {
	return c.Matcher.ValidateInitialization()
}

type validationActivation struct {
	object, oldObject, params, request interface{}
}

// ResolveName returns a value from the activation by qualified name, or false if the name
// could not be found.
func (a *validationActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case ObjectVarName:
		return a.object, true
	case OldObjectVarName:
		return a.oldObject, true
	case ParamsVarName:
		return a.params, true
	case RequestVarName:
		return a.request, true
	default:
		return nil, false
	}
}

// Parent returns the parent of the current activation, may be nil.
// If non-nil, the parent will be searched during resolve calls.
func (a *validationActivation) Parent() interpreter.Activation {
	return nil
}

// Compile compiles the cel expression defined in ValidatingAdmissionPolicy
func (c *CELValidatorCompiler) Compile(p *v1alpha1.ValidatingAdmissionPolicy) Validator {
	if len(p.Spec.Validations) == 0 {
		return nil
	}
	hasParam := false
	if p.Spec.ParamKind != nil {
		hasParam = true
	}
	compilationResults := make([]CompilationResult, len(p.Spec.Validations))
	for i, validation := range p.Spec.Validations {
		compilationResults[i] = CompileValidatingPolicyExpression(validation.Expression, hasParam)
	}
	return &CELValidator{policy: p, compilationResults: compilationResults}
}

// CELValidator implements the Validator interface
type CELValidator struct {
	policy             *v1alpha1.ValidatingAdmissionPolicy
	compilationResults []CompilationResult
}

func convertObjectToUnstructured(obj interface{}) (*unstructured.Unstructured, error) {
	if obj == nil || reflect.ValueOf(obj).IsNil() {
		return &unstructured.Unstructured{Object: nil}, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: ret}, nil
}

func objectToResolveVal(r runtime.Object) (interface{}, error) {
	if r == nil || reflect.ValueOf(r).IsNil() {
		return nil, nil
	}
	v, err := convertObjectToUnstructured(r)
	if err != nil {
		return nil, err
	}
	return v.Object, nil
}

func policyDecisionActionForError(f v1alpha1.FailurePolicyType) policyDecisionAction {
	if f == v1alpha1.Ignore {
		return actionAdmit
	}
	return actionDeny
}

// Validate validates all cel expressions in Validator and returns a PolicyDecision for each CEL expression or returns an error.
// An error will be returned if failed to convert the object/oldObject/params/request to unstructured.
// Each PolicyDecision will have a decision and a message.
// policyDecision.message will be empty if the decision is allowed and no error met.
func (v *CELValidator) Validate(a admission.Attributes, o admission.ObjectInterfaces, versionedParams runtime.Object, matchKind schema.GroupVersionKind) ([]policyDecision, error) {
	// TODO: replace unstructured with ref.Val for CEL variables when native type support is available

	decisions := make([]policyDecision, len(v.compilationResults))
	var err error
	versionedAttr, err := generic.NewVersionedAttributes(a, matchKind, o)
	if err != nil {
		return nil, err
	}
	oldObjectVal, err := objectToResolveVal(versionedAttr.VersionedOldObject)
	if err != nil {
		return nil, err
	}
	objectVal, err := objectToResolveVal(versionedAttr.VersionedObject)
	if err != nil {
		return nil, err
	}
	paramsVal, err := objectToResolveVal(versionedParams)
	if err != nil {
		return nil, err
	}
	request := createAdmissionRequest(versionedAttr.Attributes)
	requestVal, err := convertObjectToUnstructured(request)
	if err != nil {
		return nil, err
	}
	va := &validationActivation{
		object:    objectVal,
		oldObject: oldObjectVal,
		params:    paramsVal,
		request:   requestVal.Object,
	}

	var f v1alpha1.FailurePolicyType
	if v.policy.Spec.FailurePolicy == nil {
		f = v1alpha1.Fail
	} else {
		f = *v.policy.Spec.FailurePolicy
	}

	for i, compilationResult := range v.compilationResults {
		validation := v.policy.Spec.Validations[i]

		var policyDecision = &decisions[i]

		if compilationResult.Error != nil {
			policyDecision.action = policyDecisionActionForError(f)
			policyDecision.evaluation = evalError
			policyDecision.message = fmt.Sprintf("compilation error: %v", compilationResult.Error)
			continue
		}
		if compilationResult.Program == nil {
			policyDecision.action = policyDecisionActionForError(f)
			policyDecision.evaluation = evalError
			policyDecision.message = "unexpected internal error compiling expression"
			continue
		}
		t1 := time.Now()
		evalResult, _, err := compilationResult.Program.Eval(va)
		elapsed := time.Since(t1)
		policyDecision.elapsed = elapsed
		if err != nil {
			policyDecision.action = policyDecisionActionForError(f)
			policyDecision.evaluation = evalError
			policyDecision.message = fmt.Sprintf("expression '%v' resulted in error: %v", v.policy.Spec.Validations[i].Expression, err)
		} else if evalResult != celtypes.True {
			policyDecision.action = actionDeny
			if validation.Reason == nil {
				policyDecision.reason = metav1.StatusReasonInvalid
			} else {
				policyDecision.reason = *validation.Reason
			}
			if len(validation.Message) > 0 {
				policyDecision.message = strings.TrimSpace(validation.Message)
			} else {
				policyDecision.message = fmt.Sprintf("failed expression: %v", strings.TrimSpace(validation.Expression))
			}

		} else {
			policyDecision.action = actionAdmit
			policyDecision.evaluation = evalAdmit
		}
	}

	return decisions, nil
}

func createAdmissionRequest(attr admission.Attributes) *admissionv1.AdmissionRequest {
	// FIXME: how to get resource GVK, GVR and subresource?
	gvk := attr.GetKind()
	gvr := attr.GetResource()
	subresource := attr.GetSubresource()

	requestGVK := attr.GetKind()
	requestGVR := attr.GetResource()
	requestSubResource := attr.GetSubresource()

	aUserInfo := attr.GetUserInfo()
	var userInfo authenticationv1.UserInfo
	if aUserInfo != nil {
		userInfo = authenticationv1.UserInfo{
			Extra:    make(map[string]authenticationv1.ExtraValue),
			Groups:   aUserInfo.GetGroups(),
			UID:      aUserInfo.GetUID(),
			Username: aUserInfo.GetName(),
		}
		// Convert the extra information in the user object
		for key, val := range aUserInfo.GetExtra() {
			userInfo.Extra[key] = authenticationv1.ExtraValue(val)
		}
	}

	dryRun := attr.IsDryRun()

	return &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{
			Group:   gvk.Group,
			Kind:    gvk.Kind,
			Version: gvk.Version,
		},
		Resource: metav1.GroupVersionResource{
			Group:    gvr.Group,
			Resource: gvr.Resource,
			Version:  gvr.Version,
		},
		SubResource: subresource,
		RequestKind: &metav1.GroupVersionKind{
			Group:   requestGVK.Group,
			Kind:    requestGVK.Kind,
			Version: requestGVK.Version,
		},
		RequestResource: &metav1.GroupVersionResource{
			Group:    requestGVR.Group,
			Resource: requestGVR.Resource,
			Version:  requestGVR.Version,
		},
		RequestSubResource: requestSubResource,
		Name:               attr.GetName(),
		Namespace:          attr.GetNamespace(),
		Operation:          admissionv1.Operation(attr.GetOperation()),
		UserInfo:           userInfo,
		// Leave Object and OldObject unset since we don't provide access to them via request
		DryRun: &dryRun,
		Options: runtime.RawExtension{
			Object: attr.GetOperationOptions(),
		},
	}
}
