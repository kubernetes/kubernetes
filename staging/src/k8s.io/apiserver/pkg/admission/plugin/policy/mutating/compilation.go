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

package mutating

import (
	"context"
	"fmt"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/interpreter"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/admission"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/mutation"
	mutationunstructured "k8s.io/apiserver/pkg/cel/mutation/unstructured"
)

// compilePolicy compiles the policy into a PolicyEvaluator
// any error is stored and delayed until invocation.
func compilePolicy(policy *Policy) PolicyEvaluator {
	// removal not yet supported
	e := &evaluator{policy: policy}
	e.compiledEvaluator, e.err = e.compile(plugincel.OptionalVariableDeclarations{HasParams: e.hasParams()})
	return e.Invoke
}

type evaluator struct {
	policy *Policy

	compiledEvaluator *compiledEvaluator
	// err holds the error during the creation of compiledEvaluator
	err error
}

type compiledEvaluator struct {
	programs []cel.Program
}

func (e *evaluator) hasParams() bool {
	return e.policy.Spec.ParamKind != nil
}

func (e *evaluator) compile(vars plugincel.OptionalVariableDeclarations) (*compiledEvaluator, error) {
	envSet, err := createEnvSet(vars)
	if err != nil {
		return nil, err
	}
	env, err := envSet.Env(environment.StoredExpressions)
	if err != nil {
		return nil, err
	}
	var programs []cel.Program
	for _, m := range e.policy.Spec.Mutations {
		if m.PatchType != "Apply" {
			return nil, fmt.Errorf("unsupported mutation type %q", m.PatchType)
		}
		ast, issues := env.Compile(m.Expression)
		if issues != nil {
			return nil, fmt.Errorf("cannot compile CEL expression: %v", issues)
		}
		program, err := env.Program(ast)
		if err != nil {
			return nil, fmt.Errorf("cannot initiate program: %w", err)
		}
		programs = append(programs, program)
	}
	return &compiledEvaluator{
		programs: programs,
	}, nil
}

func (e *evaluator) Invoke(ctx context.Context, matchedResource schema.GroupVersionResource, versionedAttr *admission.VersionedAttributes, o admission.ObjectInterfaces, versionedParams runtime.Object, namespace *v1.Namespace, typeConverter managedfields.TypeConverter, runtimeCELCostBudget int64) (patch.Application, error) {
	err := e.err
	if err != nil {
		return nil, err
	}
	a := new(activation)
	objectGVK := versionedAttr.GetKind()
	err = a.SetObject(versionedAttr.GetObject())
	if err != nil {
		return nil, err
	}
	err = a.SetOldObject(versionedAttr.GetOldObject())
	if err != nil {
		return nil, err
	}
	err = a.SetParams(versionedParams)
	if err != nil {
		return nil, err
	}
	for _, p := range e.compiledEvaluator.programs {
		v, _, err := p.ContextEval(ctx, a)
		if err != nil {
			return nil, err
		}
		value, ok := v.Value().(map[string]any)
		if !ok {
			return nil, fmt.Errorf("unexpected evaluation result type: %t", v.Value())
		}
		o := unstructured.Unstructured{Object: a.object}
		p := unstructured.Unstructured{Object: value}
		p.SetGroupVersionKind(objectGVK)
		patched, err := patch.NewSMD(typeConverter, &o, &p)
		if err != nil {
			return nil, err
		}
		err = a.SetObject(patched.GetPatchedObject())
		if err != nil {
			return nil, err
		}
	}
	return &patchedObject{patchedObject: &unstructured.Unstructured{Object: a.object}}, nil
}

func createEnvSet(vars plugincel.OptionalVariableDeclarations) (*environment.EnvSet, error) {
	_, option := mutation.NewTypeProviderAndEnvOption(&mutationunstructured.TypeResolver{})
	options := []cel.EnvOption{option, cel.Variable("object", cel.DynType), cel.Variable("oldObject", cel.DynType)}
	if vars.HasParams {
		options = append(options, cel.Variable("params", cel.DynType))
	}
	return environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()).Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 30),
		EnvOptions:        options,
	})
}

type activation struct {
	// object is the current version of the incoming request object.
	// For the first mutation, this is the original object in the request.
	// For the second mutation and afterward, this is the object after previous mutations.
	object map[string]any

	// oldObject is the oldObject of the incoming request, or null if oldObject is not present
	// in the incoming request, i.e. for CREATE requests.
	// This is NOT the object before any mutation.
	oldObject map[string]any

	// params is the resolved params that is referred by the policy.
	// It is null if the policy does not refer to any params.
	params map[string]any
}

func (a *activation) ResolveName(name string) (any, bool) {
	switch name {
	case "object":
		return a.object, true
	case "oldObject":
		return a.oldObject, true
	case "params":
		return a.params, true
	}
	return nil, false
}

func (a *activation) Parent() interpreter.Activation {
	return nil
}

func (a *activation) SetObject(object runtime.Object) error {
	var err error
	if object == nil {
		return nil
	}
	a.object, err = runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	return err
}

func (a *activation) SetOldObject(oldObject runtime.Object) error {
	var err error
	if oldObject == nil {
		return nil
	}
	a.oldObject, err = runtime.DefaultUnstructuredConverter.ToUnstructured(oldObject)
	return err
}

func (a *activation) SetParams(params runtime.Object) error {
	var err error
	if params == nil {
		return nil
	}
	a.params, err = runtime.DefaultUnstructuredConverter.ToUnstructured(params)
	return err
}

type patchedObject struct {
	patchedObject runtime.Object
}

func (p *patchedObject) GetPatchedObject() runtime.Object {
	return p.patchedObject
}
