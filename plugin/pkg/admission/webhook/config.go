/*
Copyright 2017 The Kubernetes Authors.

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

// Package webhook checks a webhook for configured operation admission
package webhook

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apiserver/pkg/admission"
)

const (
	defaultFailAction         = Deny
	defaultRetryBackoff       = time.Duration(500) * time.Millisecond
	errInvalidFailAction      = "webhook rule (%d) has an invalid fail action: %s should be either %v or %v"
	errInvalidRuleOperation   = "webhook rule (%d) has an invalid operation (%d): %s"
	errInvalidRuleType        = "webhook rule (%d) has an invalid type: %s should be either %v or %v"
	errMissingKubeConfigFile  = "kubeConfigFile is required"
	errOneRuleRequired        = "webhook requires at least one rule defined"
	errRetryBackoffOutOfRange = "webhook retry backoff is invalid: %v should be between %v and %v"
	minRetryBackoff           = time.Duration(1) * time.Millisecond
	maxRetryBackoff           = time.Duration(5) * time.Minute
)

func normalizeConfig(config *GenericAdmissionWebhookConfig) error {
	allowFailAction := string(Allow)
	denyFailAction := string(Deny)
	connectOperation := string(admission.Connect)
	createOperation := string(admission.Create)
	deleteOperation := string(admission.Delete)
	updateOperation := string(admission.Update)
	sendRuleType := string(Send)
	skipRuleType := string(Skip)

	// Validate the kubeConfigFile property is present
	if config.KubeConfigFile == "" {
		return fmt.Errorf(errMissingKubeConfigFile)
	}

	// Normalize and validate the retry backoff
	if config.RetryBackoff == 0 {
		config.RetryBackoff = defaultRetryBackoff
	} else {
		// Unmarshalling gives nanoseconds so convert to milliseconds
		config.RetryBackoff *= time.Millisecond
	}

	if config.RetryBackoff < minRetryBackoff || config.RetryBackoff > maxRetryBackoff {
		return fmt.Errorf(errRetryBackoffOutOfRange, config.RetryBackoff, minRetryBackoff, maxRetryBackoff)
	}

	// Validate that there is at least one rule
	if len(config.Rules) == 0 {
		return fmt.Errorf(errOneRuleRequired)
	}

	for i, rule := range config.Rules {
		// Normalize and validate the fail action
		failAction := strings.ToUpper(string(rule.FailAction))

		switch failAction {
		case "":
			config.Rules[i].FailAction = defaultFailAction
		case allowFailAction:
			config.Rules[i].FailAction = Allow
		case denyFailAction:
			config.Rules[i].FailAction = Deny
		default:
			return fmt.Errorf(errInvalidFailAction, i, rule.FailAction, Allow, Deny)
		}

		// Normalize and validate the rule operation(s)
		for j, operation := range rule.Operations {
			operation := strings.ToUpper(string(operation))

			switch operation {
			case connectOperation:
				config.Rules[i].Operations[j] = admission.Connect
			case createOperation:
				config.Rules[i].Operations[j] = admission.Create
			case deleteOperation:
				config.Rules[i].Operations[j] = admission.Delete
			case updateOperation:
				config.Rules[i].Operations[j] = admission.Update
			default:
				return fmt.Errorf(errInvalidRuleOperation, i, j, rule.Operations[j])
			}
		}

		// Normalize and validate the rule type
		ruleType := strings.ToUpper(string(rule.Type))

		switch ruleType {
		case sendRuleType:
			config.Rules[i].Type = Send
		case skipRuleType:
			config.Rules[i].Type = Skip
		default:
			return fmt.Errorf(errInvalidRuleType, i, rule.Type, Send, Skip)
		}
	}

	return nil
}
