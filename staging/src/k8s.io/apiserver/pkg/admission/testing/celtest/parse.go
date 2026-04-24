/*
Copyright 2026 The Kubernetes Authors.

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

	"gopkg.in/yaml.v3"
)

type policyMeta struct {
	APIVersion string `yaml:"apiVersion"`
	Kind       string `yaml:"kind"`
}

type policyFile struct {
	Variables   []variableEntry   `yaml:"variables"`
	Validations []validationEntry `yaml:"validations"`
}

type variableEntry struct {
	Name       string `yaml:"name"`
	Expression string `yaml:"expression"`
}

type validationEntry struct {
	Expression        string `yaml:"expression"`
	Message           string `yaml:"message"`
	MessageExpression string `yaml:"messageExpression"`
}

type validatingAdmissionPolicyFile struct {
	Spec struct {
		ParamKind   *struct{}         `yaml:"paramKind"`
		Variables   []variableEntry   `yaml:"variables"`
		Validations []validationEntry `yaml:"validations"`
	} `yaml:"spec"`
}

type webhookConfigurationFile struct {
	Webhooks []struct {
		MatchConditions []struct {
			Expression string `yaml:"expression"`
		} `yaml:"matchConditions"`
	} `yaml:"webhooks"`
}

// parseAdmissionPolicy dispatches YAML parsing based on the document's Kind field.
func parseAdmissionPolicy(data []byte) (*AdmissionPolicy, error) {
	var meta policyMeta
	if err := yaml.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("parsing policy YAML: %w", err)
	}

	if meta.Kind == "" {
		return parseFlatPolicy(data)
	}

	switch meta.Kind {
	case "ValidatingAdmissionPolicy":
		return parseValidatingAdmissionPolicy(data)
	case "MutatingAdmissionPolicy":
		return nil, fmt.Errorf("MutatingAdmissionPolicy parsing is not yet implemented in this prototype")
	case "ValidatingWebhookConfiguration", "MutatingWebhookConfiguration":
		return parseWebhookConfiguration(data)
	}

	return nil, fmt.Errorf("unsupported policy source kind %q", meta.Kind)
}

// parseFlatPolicy parses a YAML document that contains bare variables and
// validations fields without a Kubernetes resource wrapper.
func parseFlatPolicy(data []byte) (*AdmissionPolicy, error) {
	var policyFile policyFile
	if err := yaml.Unmarshal(data, &policyFile); err != nil {
		return nil, fmt.Errorf("parsing fallback policy YAML: %w", err)
	}

	policy := &AdmissionPolicy{}
	policy.hasParams = true
	policy.hasParamsSet = true
	if err := appendVariables(policy, policyFile.Variables); err != nil {
		return nil, err
	}
	if err := appendValidations(policy, "validations", policyFile.Validations); err != nil {
		return nil, err
	}
	if len(policy.Variables) == 0 && len(policy.Validations) == 0 {
		return nil, fmt.Errorf("policy file does not contain variables or validations")
	}
	return policy, nil
}

// parseValidatingAdmissionPolicy parses a ValidatingAdmissionPolicy YAML document.
func parseValidatingAdmissionPolicy(data []byte) (*AdmissionPolicy, error) {
	var policyFile validatingAdmissionPolicyFile
	if err := yaml.Unmarshal(data, &policyFile); err != nil {
		return nil, fmt.Errorf("parsing ValidatingAdmissionPolicy: %w", err)
	}

	policy := &AdmissionPolicy{}
	policy.hasParams = policyFile.Spec.ParamKind != nil
	policy.hasParamsSet = true
	if err := appendVariables(policy, policyFile.Spec.Variables); err != nil {
		return nil, err
	}
	if err := appendValidations(policy, "spec.validations", policyFile.Spec.Validations); err != nil {
		return nil, err
	}
	if len(policy.Variables) == 0 && len(policy.Validations) == 0 {
		return nil, fmt.Errorf("ValidatingAdmissionPolicy does not contain variables or validations")
	}
	return policy, nil
}

// parseWebhookConfiguration parses a ValidatingWebhookConfiguration or
// MutatingWebhookConfiguration YAML document, extracting matchConditions
// expressions as validations.
func parseWebhookConfiguration(data []byte) (*AdmissionPolicy, error) {
	var config webhookConfigurationFile
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parsing webhook configuration: %w", err)
	}

	policy := &AdmissionPolicy{}
	policy.hasParams = false
	policy.hasParamsSet = true
	for webhookIndex, webhook := range config.Webhooks {
		for conditionIndex, condition := range webhook.MatchConditions {
			if condition.Expression == "" {
				return nil, fmt.Errorf("webhooks[%d].matchConditions[%d] missing expression", webhookIndex, conditionIndex)
			}
			policy.Validations = append(policy.Validations, Validation{
				Path:       fmt.Sprintf("webhooks[%d].matchConditions[%d]", webhookIndex, conditionIndex),
				Expression: condition.Expression,
			})
		}
	}
	if len(policy.Validations) == 0 {
		return nil, fmt.Errorf("webhook configuration does not contain matchConditions")
	}
	return policy, nil
}

func appendVariables(policy *AdmissionPolicy, variables []variableEntry) error {
	for _, variable := range variables {
		if variable.Name == "" {
			return fmt.Errorf("variable missing name")
		}
		if variable.Expression == "" {
			return fmt.Errorf("variable %q missing expression", variable.Name)
		}
		policy.Variables = append(policy.Variables, Variable{
			Name:       variable.Name,
			Expression: variable.Expression,
		})
	}
	return nil
}

func appendValidations(policy *AdmissionPolicy, prefix string, validations []validationEntry) error {
	for index, validation := range validations {
		if validation.Expression == "" {
			return fmt.Errorf("validation %s[%d] missing expression", prefix, index)
		}
		policy.Validations = append(policy.Validations, Validation{
			Path:              fmt.Sprintf("%s[%d]", prefix, index),
			Expression:        validation.Expression,
			Message:           validation.Message,
			MessageExpression: validation.MessageExpression,
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
