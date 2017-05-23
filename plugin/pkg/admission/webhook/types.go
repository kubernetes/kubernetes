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
	"time"

	"k8s.io/apiserver/pkg/admission"
)

// FailAction is the action to take whenever a webhook fails and there are no more retries available
type FailAction string

const (
	// Allow is the fail action taken whenever the webhook call fails and there are no more retries
	Allow FailAction = "ALLOW"
	// Deny is the fail action taken whenever the webhook call fails and there are no more retries
	Deny FailAction = "DENY"
)

// GenericAdmissionWebhookConfig holds configuration for an admission webhook
type GenericAdmissionWebhookConfig struct {
	KubeConfigFile string        `json:"kubeConfigFile"`
	RetryBackoff   time.Duration `json:"retryBackoff"`
	Rules          []Rule        `json:"rules"`
}

// Rule is the type defining an admission rule in the admission controller configuration file
type Rule struct {
	// APIGroups is a list of API groups that contain the resource this rule applies to.
	APIGroups []string `json:"apiGroups"`
	// FailAction is the action to take whenever the webhook fails and there are no more retries (Default: DENY)
	FailAction FailAction `json:"failAction"`
	// Namespaces is a list of namespaces this rule applies to.
	Namespaces []string `json:"namespaces"`
	// Operations is a list of admission operations this rule applies to.
	Operations []admission.Operation `json:"operations"`
	// Resources is a list of resources this rule applies to.
	Resources []string `json:"resources"`
	// ResourceNames is a list of resource names this rule applies to.
	ResourceNames []string `json:"resourceNames"`
	// Type is the admission rule type
	Type RuleType `json:"type"`
}

// RuleType is the type of admission rule
type RuleType string

const (
	// Send is the rule type for when a matching admission.Attributes should be sent to the webhook
	Send RuleType = "SEND"
	// Skip is the rule type for when a matching admission.Attributes should not be sent to the webhook
	Skip RuleType = "SKIP"
)
