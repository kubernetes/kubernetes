/*
Copyright The Kubernetes Authors.

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

package celtest

import (
	"fmt"
	"os"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	sigsyaml "sigs.k8s.io/yaml"
)

// policyMeta extracts the Kind to dispatch parsing. Not a real K8s type
// because flat-format policies have no TypeMeta.
type policyMeta struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
}

// policyFile represents the flat variables/validations format that is not
// a real Kubernetes resource.
type policyFile struct {
	Variables        []admissionregistrationv1.Variable        `json:"variables"`
	Validations      []admissionregistrationv1.Validation      `json:"validations"`
	MatchConditions  []admissionregistrationv1.MatchCondition  `json:"matchConditions"`
	AuditAnnotations []admissionregistrationv1.AuditAnnotation `json:"auditAnnotations"`
}

// admissionInputFile represents a YAML file containing the runtime inputs for
// admission CEL evaluation.
type admissionInputFile struct {
	Object          map[string]interface{}        `json:"object,omitempty"`
	OldObject       map[string]interface{}        `json:"oldObject,omitempty"`
	Params          map[string]interface{}        `json:"params,omitempty"`
	Request         *admissionv1.AdmissionRequest `json:"request,omitempty"`
	Namespace       *corev1.Namespace             `json:"namespace,omitempty"`
	NamespaceObject *corev1.Namespace             `json:"namespaceObject,omitempty"`
}

// parseAdmissionPolicy dispatches YAML parsing based on the document's Kind field.
func parseAdmissionPolicy(data []byte) (*AdmissionPolicy, error) {
	var meta policyMeta
	if err := sigsyaml.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("parsing policy YAML: %w", err)
	}

	if meta.Kind == "" {
		return parseFlatPolicy(data)
	}

	switch meta.Kind {
	case "ValidatingAdmissionPolicy":
		return parseValidatingAdmissionPolicy(data)
	case "MutatingAdmissionPolicy":
		return parseMutatingAdmissionPolicy(data)
	case "ValidatingWebhookConfiguration":
		return parseValidatingWebhookConfiguration(data)
	case "MutatingWebhookConfiguration":
		return parseMutatingWebhookConfiguration(data)
	}

	return nil, fmt.Errorf("unsupported policy source kind %q", meta.Kind)
}

// parseFlatPolicy parses a YAML document that contains bare variables and
// validations fields without a Kubernetes resource wrapper.
func parseFlatPolicy(data []byte) (*AdmissionPolicy, error) {
	var flat policyFile
	if err := sigsyaml.Unmarshal(data, &flat); err != nil {
		return nil, fmt.Errorf("parsing flat policy YAML: %w", err)
	}

	policy := &AdmissionPolicy{}
	policy.hasParams = true
	policy.hasParamsSet = true
	if err := appendVariables(policy, flat.Variables); err != nil {
		return nil, err
	}
	if err := appendValidations(policy, "validations", flat.Validations); err != nil {
		return nil, err
	}
	if err := appendMatchConditions(policy, "matchConditions", flat.MatchConditions); err != nil {
		return nil, err
	}
	if err := appendAuditAnnotations(policy, "auditAnnotations", flat.AuditAnnotations); err != nil {
		return nil, err
	}
	if len(policy.variables) == 0 && len(policy.validations) == 0 && len(policy.matchConditions) == 0 && len(policy.auditAnnotations) == 0 {
		return nil, fmt.Errorf("policy file does not contain CEL expressions")
	}
	return policy, nil
}

// parseValidatingAdmissionPolicy parses a ValidatingAdmissionPolicy YAML document.
func parseValidatingAdmissionPolicy(data []byte) (*AdmissionPolicy, error) {
	var vap admissionregistrationv1.ValidatingAdmissionPolicy
	if err := sigsyaml.Unmarshal(data, &vap); err != nil {
		return nil, fmt.Errorf("parsing ValidatingAdmissionPolicy: %w", err)
	}

	return NewFromValidatingAdmissionPolicy(&vap)
}

// NewFromValidatingAdmissionPolicy converts a typed ValidatingAdmissionPolicy
// into an AdmissionPolicy that can be evaluated by this package.
func NewFromValidatingAdmissionPolicy(vap *admissionregistrationv1.ValidatingAdmissionPolicy) (*AdmissionPolicy, error) {
	if vap == nil {
		return nil, fmt.Errorf("ValidatingAdmissionPolicy is nil")
	}
	policy := &AdmissionPolicy{}
	policy.hasParams = vap.Spec.ParamKind != nil
	policy.hasParamsSet = true
	if err := appendVariables(policy, vap.Spec.Variables); err != nil {
		return nil, err
	}
	if err := appendMatchConditions(policy, "spec.matchConditions", vap.Spec.MatchConditions); err != nil {
		return nil, err
	}
	if err := appendValidations(policy, "spec.validations", vap.Spec.Validations); err != nil {
		return nil, err
	}
	if err := appendAuditAnnotations(policy, "spec.auditAnnotations", vap.Spec.AuditAnnotations); err != nil {
		return nil, err
	}
	if len(policy.variables) == 0 && len(policy.validations) == 0 && len(policy.matchConditions) == 0 && len(policy.auditAnnotations) == 0 {
		return nil, fmt.Errorf("ValidatingAdmissionPolicy does not contain CEL expressions")
	}
	return policy, nil
}

// parseMutatingAdmissionPolicy parses a MutatingAdmissionPolicy YAML document.
func parseMutatingAdmissionPolicy(data []byte) (*AdmissionPolicy, error) {
	var map_ admissionregistrationv1.MutatingAdmissionPolicy
	if err := sigsyaml.Unmarshal(data, &map_); err != nil {
		return nil, fmt.Errorf("parsing MutatingAdmissionPolicy: %w", err)
	}

	return NewFromMutatingAdmissionPolicy(&map_)
}

// NewFromMutatingAdmissionPolicy converts a typed MutatingAdmissionPolicy into
// an AdmissionPolicy that can be evaluated by this package.
func NewFromMutatingAdmissionPolicy(mutatingPolicy *admissionregistrationv1.MutatingAdmissionPolicy) (*AdmissionPolicy, error) {
	if mutatingPolicy == nil {
		return nil, fmt.Errorf("MutatingAdmissionPolicy is nil")
	}
	policy := &AdmissionPolicy{}
	policy.hasParams = mutatingPolicy.Spec.ParamKind != nil
	policy.hasParamsSet = true
	if err := appendVariables(policy, mutatingPolicy.Spec.Variables); err != nil {
		return nil, err
	}
	if err := appendMatchConditions(policy, "spec.matchConditions", mutatingPolicy.Spec.MatchConditions); err != nil {
		return nil, err
	}
	if err := appendMutations(policy, "spec.mutations", mutatingPolicy.Spec.Mutations); err != nil {
		return nil, err
	}
	if len(policy.variables) == 0 && len(policy.mutations) == 0 && len(policy.matchConditions) == 0 {
		return nil, fmt.Errorf("MutatingAdmissionPolicy does not contain CEL expressions")
	}
	return policy, nil
}

// parseValidatingWebhookConfiguration parses a ValidatingWebhookConfiguration
// YAML document, extracting matchConditions expressions.
func parseValidatingWebhookConfiguration(data []byte) (*AdmissionPolicy, error) {
	var config admissionregistrationv1.ValidatingWebhookConfiguration
	if err := sigsyaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing ValidatingWebhookConfiguration: %w", err)
	}

	return NewFromValidatingWebhookConfiguration(&config)
}

// NewFromValidatingWebhookConfiguration converts a typed
// ValidatingWebhookConfiguration into an AdmissionPolicy containing its
// matchConditions.
func NewFromValidatingWebhookConfiguration(config *admissionregistrationv1.ValidatingWebhookConfiguration) (*AdmissionPolicy, error) {
	if config == nil {
		return nil, fmt.Errorf("ValidatingWebhookConfiguration is nil")
	}
	policy := &AdmissionPolicy{}
	policy.hasParams = false
	policy.hasParamsSet = true
	for webhookIndex, webhook := range config.Webhooks {
		if err := appendMatchConditions(policy, fmt.Sprintf("webhooks[%d].matchConditions", webhookIndex), webhook.MatchConditions); err != nil {
			return nil, err
		}
	}
	if len(policy.matchConditions) == 0 {
		return nil, fmt.Errorf("ValidatingWebhookConfiguration does not contain CEL expressions")
	}
	return policy, nil
}

// parseMutatingWebhookConfiguration parses a MutatingWebhookConfiguration
// YAML document, extracting matchConditions expressions.
func parseMutatingWebhookConfiguration(data []byte) (*AdmissionPolicy, error) {
	var config admissionregistrationv1.MutatingWebhookConfiguration
	if err := sigsyaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing MutatingWebhookConfiguration: %w", err)
	}

	return NewFromMutatingWebhookConfiguration(&config)
}

// NewFromMutatingWebhookConfiguration converts a typed
// MutatingWebhookConfiguration into an AdmissionPolicy containing its
// matchConditions.
func NewFromMutatingWebhookConfiguration(config *admissionregistrationv1.MutatingWebhookConfiguration) (*AdmissionPolicy, error) {
	if config == nil {
		return nil, fmt.Errorf("MutatingWebhookConfiguration is nil")
	}
	policy := &AdmissionPolicy{}
	policy.hasParams = false
	policy.hasParamsSet = true
	for webhookIndex, webhook := range config.Webhooks {
		if err := appendMatchConditions(policy, fmt.Sprintf("webhooks[%d].matchConditions", webhookIndex), webhook.MatchConditions); err != nil {
			return nil, err
		}
	}
	if len(policy.matchConditions) == 0 {
		return nil, fmt.Errorf("MutatingWebhookConfiguration does not contain CEL expressions")
	}
	return policy, nil
}

func appendVariables(policy *AdmissionPolicy, variables []admissionregistrationv1.Variable) error {
	for _, item := range variables {
		if item.Name == "" {
			return fmt.Errorf("variable missing name")
		}
		if item.Expression == "" {
			return fmt.Errorf("variable %q missing expression", item.Name)
		}
		policy.variables = append(policy.variables, variable{
			Name:       item.Name,
			Expression: item.Expression,
		})
	}
	return nil
}

func appendValidations(policy *AdmissionPolicy, prefix string, validations []admissionregistrationv1.Validation) error {
	for index, item := range validations {
		if item.Expression == "" {
			return fmt.Errorf("validation %s[%d] missing expression", prefix, index)
		}
		policy.validations = append(policy.validations, validation{
			Path:              fmt.Sprintf("%s[%d]", prefix, index),
			Expression:        item.Expression,
			Message:           item.Message,
			MessageExpression: item.MessageExpression,
		})
	}
	return nil
}

func appendMatchConditions(policy *AdmissionPolicy, prefix string, conditions []admissionregistrationv1.MatchCondition) error {
	for index, condition := range conditions {
		if condition.Expression == "" {
			return fmt.Errorf("match condition %s[%d] missing expression", prefix, index)
		}
		policy.matchConditions = append(policy.matchConditions, matchCondition{
			Path:       fmt.Sprintf("%s[%d]", prefix, index),
			Name:       condition.Name,
			Expression: condition.Expression,
		})
	}
	return nil
}

func appendAuditAnnotations(policy *AdmissionPolicy, prefix string, annotations []admissionregistrationv1.AuditAnnotation) error {
	for index, annotation := range annotations {
		if annotation.Key == "" {
			return fmt.Errorf("audit annotation %s[%d] missing key", prefix, index)
		}
		if annotation.ValueExpression == "" {
			return fmt.Errorf("audit annotation %s[%d] missing valueExpression", prefix, index)
		}
		policy.auditAnnotations = append(policy.auditAnnotations, auditAnnotation{
			Path:            fmt.Sprintf("%s[%d]", prefix, index),
			Key:             annotation.Key,
			ValueExpression: annotation.ValueExpression,
		})
	}
	return nil
}

// parseAdmissionPolicyFile reads a YAML file from disk and parses it.
func parseAdmissionPolicyFile(path string) (*AdmissionPolicy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading policy file %s: %w", path, err)
	}
	return parseAdmissionPolicy(data)
}

func parseAdmissionInput(data []byte) (*AdmissionInput, error) {
	var file admissionInputFile
	if err := sigsyaml.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parsing admission input YAML: %w", err)
	}

	if file.Namespace != nil && file.NamespaceObject != nil {
		return nil, fmt.Errorf("admission input may set either namespace or namespaceObject, not both")
	}
	namespace := file.Namespace
	if namespace == nil {
		namespace = file.NamespaceObject
	}

	return &AdmissionInput{
		object:    file.Object,
		oldObject: file.OldObject,
		params:    file.Params,
		request:   file.Request,
		namespace: namespace,
	}, nil
}

func parseAdmissionInputFile(path string) (*AdmissionInput, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading admission input file %s: %w", path, err)
	}
	return parseAdmissionInput(data)
}

func appendMutations(policy *AdmissionPolicy, prefix string, mutations []admissionregistrationv1.Mutation) error {
	for index, item := range mutations {
		if item.PatchType == "" {
			return fmt.Errorf("mutation %s[%d] missing patchType", prefix, index)
		}
		var expression string
		switch item.PatchType {
		case admissionregistrationv1.PatchTypeApplyConfiguration:
			if item.ApplyConfiguration == nil || item.ApplyConfiguration.Expression == "" {
				return fmt.Errorf("mutation %s[%d] missing applyConfiguration.expression", prefix, index)
			}
			expression = item.ApplyConfiguration.Expression
		case admissionregistrationv1.PatchTypeJSONPatch:
			if item.JSONPatch == nil || item.JSONPatch.Expression == "" {
				return fmt.Errorf("mutation %s[%d] missing jsonPatch.expression", prefix, index)
			}
			expression = item.JSONPatch.Expression
		default:
			return fmt.Errorf("mutation %s[%d] unsupported patchType %q", prefix, index, item.PatchType)
		}
		policy.mutations = append(policy.mutations, mutation{
			Path:       fmt.Sprintf("%s[%d]", prefix, index),
			PatchType:  string(item.PatchType),
			Expression: expression,
		})
	}
	return nil
}
