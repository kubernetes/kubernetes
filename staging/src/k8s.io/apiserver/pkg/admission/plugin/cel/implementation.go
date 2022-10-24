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
	"fmt"
	"strings"
	"sync"

	cel_go "github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	apiservercel "k8s.io/apiserver/pkg/cel"
	v1 "k8s.io/client-go/listers/core/v1"
)

type MyContext struct {
	namespaceLister v1.NamespaceLister
}

type ValidatingAdmissionPolicyDefinition struct {
	v1alpha1.ValidatingAdmissionPolicy

	context MyContext
}

func (p ValidatingAdmissionPolicyDefinition) Matches(a admission.Attributes) bool {
	// TODO cici37: call the matches function when it is done
	//return matches(a, p.Spec.MatchConstraints, &p.context)
	return true
}

var (
	initEnvOnce sync.Once
	initEnv     *cel_go.Env
	initEnvErr  error
)

type validationActivation struct {
	object    ref.Val
	oldObject ref.Val
	params    ref.Val
}

func (a *validationActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case ObjectVarName:
		return a.object, true
	case OldObjectVarName:
		return a.oldObject, true
	case ParamsVarName:
		return a.params, true
	default:
		return nil, false
	}
}

func (a *validationActivation) Parent() interpreter.Activation {
	return nil
}

func (p ValidatingAdmissionPolicyDefinition) Compile(objectConverter ValidatorCompiler, mapper meta.RESTMapper) (EvaluatorFunc, error) {
	if len(p.Spec.Validations) == 0 {
		return nil, nil
	}
	hasParam := false
	if p.GetParamSource() != nil {
		hasParam = true
	}
	var compilationResults []CompilationResult
	for i, validation := range p.Spec.Validations {
		compilationResults[i] = CompileValidatingPolicyExpression(validation.Expression, hasParam)
	}

	evaluator_func := func(a admission.Attributes, params *unstructured.Unstructured) ([]PolicyDecision, error) {
		decisions := make([]PolicyDecision, len(p.Spec.Validations))

		objectVal, err := objectConverter.ValueForObject(a.GetObject(), apiservercel.DynType)
		if err != nil {
			return nil, err
		}
		var oldObjectVal ref.Val
		if a.GetOldObject() != nil {
			oldObjectVal, err = objectConverter.ValueForObject(a.GetOldObject(), apiservercel.DynType)
			if err != nil {
				return nil, err
			}
		}

		var paramsVal ref.Val
		if hasParam {
			paramsVal, err = objectConverter.ValueForObject(params, apiservercel.DynType)
			if err != nil {
				return nil, err
			}
		}
		// TODO: pass in Request?

		va := &validationActivation{
			object:    objectVal,
			oldObject: oldObjectVal,
			params:    paramsVal,
		}

		for i, compilationResult := range compilationResults {
			var policyDecision = &decisions[i]
			if compilationResult.Program == nil {
				continue
			}
			if compilationResult.Error != nil {
				policyDecision.Message = fmt.Sprintf("compilation error: %v", compilationResult.Error)
				continue
			}
			if compilationResult.TransitionRule && a.GetOldObject() == nil {
				continue
			}
			evalResult, _, err := compilationResult.Program.Eval(va)
			if err != nil {
				policyDecision.Message = fmt.Sprintf("evaluation error: %v", err)
			}
			if evalResult != types.True {
				policyDecision.Kind = Deny
				// FIXME: properly format the error message.
				policyDecision.Message = fmt.Sprintf("validation failed: %v, reason: %v", p.Spec.Validations[i].Message, p.Spec.Validations[i].Reason)
			} else {
				policyDecision.Kind = Admit
			}
		}

		return decisions, nil
	}

	return evaluator_func, nil
}

func (p ValidatingAdmissionPolicyDefinition) GetParamSource() *schema.GroupVersionKind {
	if p.Spec.ParamKind == nil {
		return nil
	}
	group, version, found := strings.Cut(p.Spec.ParamKind.APIVersion, "/")
	if !found || version == "" {
		panic("invalid apiVersion")
	}
	return &schema.GroupVersionKind{
		Group:   group,
		Version: version,
		Kind:    p.Spec.ParamKind.Kind,
	}
}

func (p ValidatingAdmissionPolicyDefinition) GetFailurePolicy() FailurePolicy {
	switch *p.Spec.FailurePolicy {
	case v1alpha1.Fail:
		return Fail
	case v1alpha1.Ignore:
		return Ignore
	default:
		panic("unknown failure policy")
	}
}

func NewPolicyFrom(obj interface{}) PolicyDefinition {
	// TODO cici37: err handling?
	return &ValidatingAdmissionPolicyDefinition{
		ValidatingAdmissionPolicy: obj.(v1alpha1.ValidatingAdmissionPolicy),
	}
}

type ValidatingAdmissionPolicyBinding struct {
	v1alpha1.ValidatingAdmissionPolicyBinding

	context MyContext
}

func (p *ValidatingAdmissionPolicyBinding) Matches(a admission.Attributes) bool {
	// TODO cici37: call Matches when it is done
	//return Matches(a, p.Spec.MatchResources, &p.context)
	return true
}

func (p *ValidatingAdmissionPolicyBinding) GetTargetDefinition() (namespace, name string) {
	return p.Namespace, p.Spec.PolicyName
}

func (p *ValidatingAdmissionPolicyBinding) GetTargetParams() (namespace, name string) {
	if p.Spec.ParamRef == nil {
		return "", ""
	}
	return p.Spec.ParamRef.Namespace, p.Spec.ParamRef.Name
}

func NewBindingFrom(obj interface{}) PolicyBinding {
	// TODO cici37: err handling?
	return &ValidatingAdmissionPolicyBinding{
		ValidatingAdmissionPolicyBinding: obj.(v1alpha1.ValidatingAdmissionPolicyBinding),
	}
}
