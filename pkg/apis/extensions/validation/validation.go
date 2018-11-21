/*
Copyright 2015 The Kubernetes Authors.

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
	"net"
	"regexp"
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// ValidateIngress tests if required fields in the Ingress are set.
func ValidateIngress(ingress *extensions.Ingress) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ingress.ObjectMeta, true, ValidateIngressName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIngressName validates that the given name can be used as an Ingress name.
var ValidateIngressName = apimachineryvalidation.NameIsDNSSubdomain

func validateIngressTLS(spec *extensions.IngressSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO: Perform a more thorough validation of spec.TLS.Hosts that takes
	// the wildcard spec from RFC 6125 into account.
	for _, itls := range spec.TLS {
		for i, host := range itls.Hosts {
			if strings.Contains(host, "*") {
				for _, msg := range validation.IsWildcardDNS1123Subdomain(host) {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("hosts"), host, msg))
				}
				continue
			}
			for _, msg := range validation.IsDNS1123Subdomain(host) {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("hosts"), host, msg))
			}
		}
	}

	return allErrs
}

// ValidateIngressSpec tests if required fields in the IngressSpec are set.
func ValidateIngressSpec(spec *extensions.IngressSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO: Is a default backend mandatory?
	if spec.Backend != nil {
		allErrs = append(allErrs, validateIngressBackend(spec.Backend, fldPath.Child("backend"))...)
	} else if len(spec.Rules) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Rules, "either `backend` or `rules` must be specified"))
	}
	if len(spec.Rules) > 0 {
		allErrs = append(allErrs, validateIngressRules(spec.Rules, fldPath.Child("rules"))...)
	}
	if len(spec.TLS) > 0 {
		allErrs = append(allErrs, validateIngressTLS(spec, fldPath.Child("tls"))...)
	}
	return allErrs
}

// ValidateIngressUpdate tests if required fields in the Ingress are set.
func ValidateIngressUpdate(ingress, oldIngress *extensions.Ingress) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIngressStatusUpdate tests if required fields in the Ingress are set when updating status.
func ValidateIngressStatusUpdate(ingress, oldIngress *extensions.Ingress) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apivalidation.ValidateLoadBalancerStatus(&ingress.Status.LoadBalancer, field.NewPath("status", "loadBalancer"))...)
	return allErrs
}

func validateIngressRules(ingressRules []extensions.IngressRule, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ingressRules) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}
	for i, ih := range ingressRules {
		if len(ih.Host) > 0 {
			if isIP := (net.ParseIP(ih.Host) != nil); isIP {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("host"), ih.Host, "must be a DNS name, not an IP address"))
			}
			// TODO: Ports and ips are allowed in the host part of a url
			// according to RFC 3986, consider allowing them.
			if strings.Contains(ih.Host, "*") {
				for _, msg := range validation.IsWildcardDNS1123Subdomain(ih.Host) {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("host"), ih.Host, msg))
				}
				continue
			}
			for _, msg := range validation.IsDNS1123Subdomain(ih.Host) {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("host"), ih.Host, msg))
			}
		}
		allErrs = append(allErrs, validateIngressRuleValue(&ih.IngressRuleValue, fldPath.Index(0))...)
	}
	return allErrs
}

func validateIngressRuleValue(ingressRule *extensions.IngressRuleValue, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if ingressRule.HTTP != nil {
		allErrs = append(allErrs, validateHTTPIngressRuleValue(ingressRule.HTTP, fldPath.Child("http"))...)
	}
	return allErrs
}

func validateHTTPIngressRuleValue(httpIngressRuleValue *extensions.HTTPIngressRuleValue, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(httpIngressRuleValue.Paths) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("paths"), ""))
	}
	for i, rule := range httpIngressRuleValue.Paths {
		if len(rule.Path) > 0 {
			if !strings.HasPrefix(rule.Path, "/") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("paths").Index(i).Child("path"), rule.Path, "must be an absolute path"))
			}
			// TODO: More draconian path regex validation.
			// Path must be a valid regex. This is the basic requirement.
			// In addition to this any characters not allowed in a path per
			// RFC 3986 section-3.3 cannot appear as a literal in the regex.
			// Consider the example: http://host/valid?#bar, everything after
			// the last '/' is a valid regex that matches valid#bar, which
			// isn't a valid path, because the path terminates at the first ?
			// or #. A more sophisticated form of validation would detect that
			// the user is confusing url regexes with path regexes.
			_, err := regexp.CompilePOSIX(rule.Path)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("paths").Index(i).Child("path"), rule.Path, "must be a valid regex"))
			}
		}
		allErrs = append(allErrs, validateIngressBackend(&rule.Backend, fldPath.Child("backend"))...)
	}
	return allErrs
}

// validateIngressBackend tests if a given backend is valid.
func validateIngressBackend(backend *extensions.IngressBackend, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// All backends must reference a single local service by name, and a single service port by name or number.
	if len(backend.ServiceName) == 0 {
		return append(allErrs, field.Required(fldPath.Child("serviceName"), ""))
	} else {
		for _, msg := range apivalidation.ValidateServiceName(backend.ServiceName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("serviceName"), backend.ServiceName, msg))
		}
	}
	allErrs = append(allErrs, apivalidation.ValidatePortNumOrName(backend.ServicePort, fldPath.Child("servicePort"))...)
	return allErrs
}
