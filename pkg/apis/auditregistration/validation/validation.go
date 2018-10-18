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
	"fmt"
	"net/url"
	"strings"

	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
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
	allErrs = append(allErrs, ValidatePolicy(s.Policy, field.NewPath("policy"))...)
	allErrs = append(allErrs, ValidateWebhook(s.Webhook, field.NewPath("webhook"))...)
	return allErrs
}

// ValidateWebhook validates the webhook
func ValidateWebhook(w auditregistration.Webhook, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if w.Throttle != nil {
		allErrs = append(allErrs, ValidateWebhookThrottleConfig(w.Throttle, fldPath.Child("throttle"))...)
	}
	allErrs = append(allErrs, ValidateWebhookClientConfig(&w.ClientConfig, fldPath.Child("clientConfig"))...)
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

// ValidateWebhookClientConfig validates the WebhookClientConfig
// note: this is largely copy/paste inheritance from admissionregistration with subtle changes
func ValidateWebhookClientConfig(cc *auditregistration.WebhookClientConfig, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	if (cc.URL == nil) == (cc.Service == nil) {
		allErrors = append(allErrors, field.Required(fldPath.Child("url"), "exactly one of url or service is required"))
	}

	if cc.URL != nil {
		const form = "; desired format: https://host[/path]"
		if u, err := url.Parse(*cc.URL); err != nil {
			allErrors = append(allErrors, field.Required(fldPath.Child("url"), "url must be a valid URL: "+err.Error()+form))
		} else {
			if len(u.Host) == 0 {
				allErrors = append(allErrors, field.Invalid(fldPath.Child("url"), u.Host, "host must be provided"+form))
			}
			if u.User != nil {
				allErrors = append(allErrors, field.Invalid(fldPath.Child("url"), u.User.String(), "user information is not permitted in the URL"))
			}
			if len(u.Fragment) != 0 {
				allErrors = append(allErrors, field.Invalid(fldPath.Child("url"), u.Fragment, "fragments are not permitted in the URL"))
			}
			if len(u.RawQuery) != 0 {
				allErrors = append(allErrors, field.Invalid(fldPath.Child("url"), u.RawQuery, "query parameters are not permitted in the URL"))
			}
		}
	}

	if cc.Service != nil {
		allErrors = append(allErrors, validateWebhookService(cc.Service, fldPath.Child("service"))...)
	}
	return allErrors
}

// note: this is copy/paste inheritance from admissionregistration
func validateWebhookService(svc *auditregistration.ServiceReference, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList

	if len(svc.Name) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("name"), "service name is required"))
	}

	if len(svc.Namespace) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("namespace"), "service namespace is required"))
	}

	if svc.Path == nil {
		return allErrors
	}

	// TODO: replace below with url.Parse + verifying that host is empty?

	urlPath := *svc.Path
	if urlPath == "/" || len(urlPath) == 0 {
		return allErrors
	}
	if urlPath == "//" {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, "segment[0] may not be empty"))
		return allErrors
	}

	if !strings.HasPrefix(urlPath, "/") {
		allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, "must start with a '/'"))
	}

	urlPathToCheck := urlPath[1:]
	if strings.HasSuffix(urlPathToCheck, "/") {
		urlPathToCheck = urlPathToCheck[:len(urlPathToCheck)-1]
	}
	steps := strings.Split(urlPathToCheck, "/")
	for i, step := range steps {
		if len(step) == 0 {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, fmt.Sprintf("segment[%d] may not be empty", i)))
			continue
		}
		failures := validation.IsDNS1123Subdomain(step)
		for _, failure := range failures {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("path"), urlPath, fmt.Sprintf("segment[%d]: %v", i, failure)))
		}
	}

	return allErrors
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
