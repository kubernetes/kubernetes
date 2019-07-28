/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	"strings"

	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/kubernetes/pkg/apis/auditregistration"
)

// ValidateAuditSink validates the AuditSinks
func ValidateAuditSink(as *auditregistration.AuditSink) field.ErrorList {
	allErrs := genericvalidation.ValidateObjectMeta(&as.ObjectMeta, false, genericvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateAuditSinkSpec(as.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateAuditSinkSpec validates the sink spec for audit
func ValidateAuditSinkSpec(s auditregistration.AuditSinkSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, ValidatePolicy(s.Policy, fldPath.Child("policy"))...)
	allErrs = append(allErrs, ValidateWebhook(s.Webhook, fldPath.Child("webhook"))...)
	return allErrs
}

// ValidateWebhook validates the webhook
func ValidateWebhook(w auditregistration.Webhook, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if w.Throttle != nil {
		allErrs = append(allErrs, ValidateWebhookThrottleConfig(w.Throttle, fldPath.Child("throttle"))...)
	}

	cc := w.ClientConfig
	switch {
	case (cc.URL == nil) == (cc.Service == nil):
		allErrs = append(allErrs, field.Required(fldPath.Child("clientConfig"), "exactly one of url or service is required"))
	case cc.URL != nil:
		allErrs = append(allErrs, webhook.ValidateWebhookURL(fldPath.Child("clientConfig").Child("url"), *cc.URL, false)...)
	case cc.Service != nil:
		allErrs = append(allErrs, webhook.ValidateWebhookService(fldPath.Child("clientConfig").Child("service"), cc.Service.Name, cc.Service.Namespace, cc.Service.Path, cc.Service.Port)...)
	}
	return allErrs
}

// ValidateWebhookThrottleConfig validates the throttle config
func ValidateWebhookThrottleConfig(c *auditregistration.WebhookThrottleConfig, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if c.QPS != nil && *c.QPS <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("qps"), c.QPS, "qps must be a positive number"))
	}
	if c.Burst != nil && *c.Burst <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("burst"), c.Burst, "burst must be a positive number"))
	}
	return allErrs
}

// ValidatePolicy validates the audit policy
func ValidatePolicy(policy auditregistration.Policy, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, validateStages(policy.Stages, fldPath.Child("stages"))...)
	allErrs = append(allErrs, validateLevel(policy.Level, fldPath.Child("level"))...)
	if policy.Level != auditregistration.LevelNone && len(policy.Stages) == 0 {
		return field.ErrorList{field.Required(fldPath.Child("stages"), "")}
	}
	return allErrs
}

var validLevels = sets.NewString(
	string(auditregistration.LevelNone),
	string(auditregistration.LevelMetadata),
	string(auditregistration.LevelRequest),
	string(auditregistration.LevelRequestResponse),
)

var validStages = sets.NewString(
	string(auditregistration.StageRequestReceived),
	string(auditregistration.StageResponseStarted),
	string(auditregistration.StageResponseComplete),
	string(auditregistration.StagePanic),
)

func validateLevel(level auditregistration.Level, fldPath *field.Path) field.ErrorList {
	if string(level) == "" {
		return field.ErrorList{field.Required(fldPath, "")}
	}
	if !validLevels.Has(string(level)) {
		return field.ErrorList{field.NotSupported(fldPath, level, validLevels.List())}
	}
	return nil
}

func validateStages(stages []auditregistration.Stage, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for i, stage := range stages {
		if !validStages.Has(string(stage)) {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), string(stage), "allowed stages are "+strings.Join(validStages.List(), ",")))
		}
	}
	return allErrs
}

// ValidateAuditSinkUpdate validates an update to the object
func ValidateAuditSinkUpdate(newC, oldC *auditregistration.AuditSink) field.ErrorList {
	return ValidateAuditSink(newC)
}
