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

package validation

import (
	"fmt"
	"net"
	"strings"

	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	pathvalidation "k8s.io/apimachinery/pkg/api/validation/path"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/networking"
)

const (
	annotationIngressClass       = "kubernetes.io/ingress.class"
	maxLenIngressClassController = 250
)

var (
	supportedPathTypes = sets.NewString(
		string(networking.PathTypeExact),
		string(networking.PathTypePrefix),
		string(networking.PathTypeImplementationSpecific),
	)
	invalidPathSequences = []string{"//", "/./", "/../", "%2f", "%2F"}
	invalidPathSuffixes  = []string{"/..", "/."}
)

// ValidateNetworkPolicyName can be used to check whether the given networkpolicy
// name is valid.
func ValidateNetworkPolicyName(name string, prefix bool) []string {
	return apimachineryvalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidateNetworkPolicyPort validates a NetworkPolicyPort
func ValidateNetworkPolicyPort(port *networking.NetworkPolicyPort, portPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if port.Protocol != nil && *port.Protocol != api.ProtocolTCP && *port.Protocol != api.ProtocolUDP && *port.Protocol != api.ProtocolSCTP {
		allErrs = append(allErrs, field.NotSupported(portPath.Child("protocol"), *port.Protocol, []string{string(api.ProtocolTCP), string(api.ProtocolUDP), string(api.ProtocolSCTP)}))
	}
	if port.Port != nil {
		if port.Port.Type == intstr.Int {
			for _, msg := range validation.IsValidPortNum(int(port.Port.IntVal)) {
				allErrs = append(allErrs, field.Invalid(portPath.Child("port"), port.Port.IntVal, msg))
			}
		} else {
			for _, msg := range validation.IsValidPortName(port.Port.StrVal) {
				allErrs = append(allErrs, field.Invalid(portPath.Child("port"), port.Port.StrVal, msg))
			}
		}
	}

	return allErrs
}

// ValidateNetworkPolicyPeer validates a NetworkPolicyPeer
func ValidateNetworkPolicyPeer(peer *networking.NetworkPolicyPeer, peerPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	numPeers := 0

	if peer.PodSelector != nil {
		numPeers++
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(peer.PodSelector, peerPath.Child("podSelector"))...)
	}
	if peer.NamespaceSelector != nil {
		numPeers++
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(peer.NamespaceSelector, peerPath.Child("namespaceSelector"))...)
	}
	if peer.IPBlock != nil {
		numPeers++
		allErrs = append(allErrs, ValidateIPBlock(peer.IPBlock, peerPath.Child("ipBlock"))...)
	}

	if numPeers == 0 {
		allErrs = append(allErrs, field.Required(peerPath, "must specify a peer"))
	} else if numPeers > 1 && peer.IPBlock != nil {
		allErrs = append(allErrs, field.Forbidden(peerPath, "may not specify both ipBlock and another peer"))
	}

	return allErrs
}

// ValidateNetworkPolicySpec tests if required fields in the networkpolicy spec are set.
func ValidateNetworkPolicySpec(spec *networking.NetworkPolicySpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(&spec.PodSelector, fldPath.Child("podSelector"))...)

	// Validate ingress rules.
	for i, ingress := range spec.Ingress {
		ingressPath := fldPath.Child("ingress").Index(i)
		for i, port := range ingress.Ports {
			portPath := ingressPath.Child("ports").Index(i)
			allErrs = append(allErrs, ValidateNetworkPolicyPort(&port, portPath)...)
		}
		for i, from := range ingress.From {
			fromPath := ingressPath.Child("from").Index(i)
			allErrs = append(allErrs, ValidateNetworkPolicyPeer(&from, fromPath)...)
		}
	}
	// Validate egress rules
	for i, egress := range spec.Egress {
		egressPath := fldPath.Child("egress").Index(i)
		for i, port := range egress.Ports {
			portPath := egressPath.Child("ports").Index(i)
			allErrs = append(allErrs, ValidateNetworkPolicyPort(&port, portPath)...)
		}
		for i, to := range egress.To {
			toPath := egressPath.Child("to").Index(i)
			allErrs = append(allErrs, ValidateNetworkPolicyPeer(&to, toPath)...)
		}
	}
	// Validate PolicyTypes
	allowed := sets.NewString(string(networking.PolicyTypeIngress), string(networking.PolicyTypeEgress))
	if len(spec.PolicyTypes) > len(allowed) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("policyTypes"), &spec.PolicyTypes, "may not specify more than two policyTypes"))
		return allErrs
	}
	for i, pType := range spec.PolicyTypes {
		policyPath := fldPath.Child("policyTypes").Index(i)
		if !allowed.Has(string(pType)) {
			allErrs = append(allErrs, field.NotSupported(policyPath, pType, []string{string(networking.PolicyTypeIngress), string(networking.PolicyTypeEgress)}))
		}
	}
	return allErrs
}

// ValidateNetworkPolicy validates a networkpolicy.
func ValidateNetworkPolicy(np *networking.NetworkPolicy) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&np.ObjectMeta, true, ValidateNetworkPolicyName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateNetworkPolicySpec(&np.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateNetworkPolicyUpdate tests if an update to a NetworkPolicy is valid.
func ValidateNetworkPolicyUpdate(update, old *networking.NetworkPolicy) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, ValidateNetworkPolicySpec(&update.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIPBlock validates a cidr and the except fields of an IpBlock NetworkPolicyPeer
func ValidateIPBlock(ipb *networking.IPBlock, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ipb.CIDR) == 0 || ipb.CIDR == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("cidr"), ""))
		return allErrs
	}
	cidrIPNet, err := apivalidation.ValidateCIDR(ipb.CIDR)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("cidr"), ipb.CIDR, "not a valid CIDR"))
		return allErrs
	}
	exceptCIDR := ipb.Except
	for i, exceptIP := range exceptCIDR {
		exceptPath := fldPath.Child("except").Index(i)
		exceptCIDR, err := apivalidation.ValidateCIDR(exceptIP)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(exceptPath, exceptIP, "not a valid CIDR"))
			return allErrs
		}
		cidrMaskLen, _ := cidrIPNet.Mask.Size()
		exceptMaskLen, _ := exceptCIDR.Mask.Size()
		if !cidrIPNet.Contains(exceptCIDR.IP) || cidrMaskLen >= exceptMaskLen {
			allErrs = append(allErrs, field.Invalid(exceptPath, exceptIP, "must be a strict subset of `cidr`"))
		}
	}
	return allErrs
}

// ValidateIngressName validates that the given name can be used as an Ingress
// name.
var ValidateIngressName = apimachineryvalidation.NameIsDNSSubdomain

// IngressValidationOptions cover beta to GA transitions for HTTP PathType
type IngressValidationOptions struct {
	// AllowInvalidSecretName indicates whether spec.tls[*].secretName values that are not valid Secret names should be allowed
	AllowInvalidSecretName bool

	// AllowInvalidWildcardHostRule indicates whether invalid rule values are allowed in rules with wildcard hostnames
	AllowInvalidWildcardHostRule bool
}

// ValidateIngress validates Ingresses on create and update.
func validateIngress(ingress *networking.Ingress, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ingress.ObjectMeta, true, ValidateIngressName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, field.NewPath("spec"), opts, requestGV)...)
	return allErrs
}

// ValidateIngressCreate validates Ingresses on create.
func ValidateIngressCreate(ingress *networking.Ingress, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	var opts IngressValidationOptions
	opts = IngressValidationOptions{
		AllowInvalidSecretName:       allowInvalidSecretName(requestGV, nil),
		AllowInvalidWildcardHostRule: allowInvalidWildcardHostRule(requestGV, nil),
	}
	allErrs = append(allErrs, validateIngress(ingress, opts, requestGV)...)
	annotationVal, annotationIsSet := ingress.Annotations[annotationIngressClass]
	if annotationIsSet && ingress.Spec.IngressClassName != nil {
		annotationPath := field.NewPath("annotations").Child(annotationIngressClass)
		allErrs = append(allErrs, field.Invalid(annotationPath, annotationVal, "can not be set when the class field is also set"))
	}
	return allErrs
}

// ValidateIngressUpdate validates ingresses on update.
func ValidateIngressUpdate(ingress, oldIngress *networking.Ingress, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, field.NewPath("metadata"))
	var opts IngressValidationOptions
	opts = IngressValidationOptions{
		AllowInvalidSecretName:       allowInvalidSecretName(requestGV, oldIngress),
		AllowInvalidWildcardHostRule: allowInvalidWildcardHostRule(requestGV, oldIngress),
	}

	allErrs = append(allErrs, validateIngress(ingress, opts, requestGV)...)
	return allErrs
}

func validateIngressTLS(spec *networking.IngressSpec, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO: Perform a more thorough validation of spec.TLS.Hosts that takes
	// the wildcard spec from RFC 6125 into account.
	for tlsIndex, itls := range spec.TLS {
		for i, host := range itls.Hosts {
			if strings.Contains(host, "*") {
				for _, msg := range validation.IsWildcardDNS1123Subdomain(host) {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(tlsIndex).Child("hosts").Index(i), host, msg))
				}
				continue
			}
			for _, msg := range validation.IsDNS1123Subdomain(host) {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(tlsIndex).Child("hosts").Index(i), host, msg))
			}
		}

		if !opts.AllowInvalidSecretName {
			for _, msg := range validateTLSSecretName(itls.SecretName) {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(tlsIndex).Child("secretName"), itls.SecretName, msg))
			}
		}
	}

	return allErrs
}

// defaultBackendFieldName returns the name of the field used for defaultBackend
// in the provided GroupVersion.
func defaultBackendFieldName(gv schema.GroupVersion) string {
	switch gv {
	case networkingv1beta1.SchemeGroupVersion, extensionsv1beta1.SchemeGroupVersion:
		return "backend"
	default:
		return "defaultBackend"
	}
}

// ValidateIngressSpec tests if required fields in the IngressSpec are set.
func ValidateIngressSpec(spec *networking.IngressSpec, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(spec.Rules) == 0 && spec.DefaultBackend == nil {
		errMsg := fmt.Sprintf("either `%s` or `rules` must be specified", defaultBackendFieldName(requestGV))
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Rules, errMsg))
	}
	if spec.DefaultBackend != nil {
		allErrs = append(allErrs, validateIngressBackend(spec.DefaultBackend, fldPath.Child(defaultBackendFieldName(requestGV)), opts, requestGV)...)
	}
	if len(spec.Rules) > 0 {
		allErrs = append(allErrs, validateIngressRules(spec.Rules, fldPath.Child("rules"), opts, requestGV)...)
	}
	if len(spec.TLS) > 0 {
		allErrs = append(allErrs, validateIngressTLS(spec, fldPath.Child("tls"), opts)...)
	}
	if spec.IngressClassName != nil {
		for _, msg := range ValidateIngressClassName(*spec.IngressClassName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("ingressClassName"), *spec.IngressClassName, msg))
		}
	}
	return allErrs
}

// ValidateIngressStatusUpdate tests if required fields in the Ingress are set when updating status.
func ValidateIngressStatusUpdate(ingress, oldIngress *networking.Ingress) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apivalidation.ValidateLoadBalancerStatus(&ingress.Status.LoadBalancer, field.NewPath("status", "loadBalancer"))...)
	return allErrs
}

func validateIngressRules(ingressRules []networking.IngressRule, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ingressRules) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}
	for i, ih := range ingressRules {
		wildcardHost := false
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
				wildcardHost = true
			} else {
				for _, msg := range validation.IsDNS1123Subdomain(ih.Host) {
					allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("host"), ih.Host, msg))
				}
			}
		}

		if !wildcardHost || !opts.AllowInvalidWildcardHostRule {
			allErrs = append(allErrs, validateIngressRuleValue(&ih.IngressRuleValue, fldPath.Index(i), opts, requestGV)...)
		}
	}
	return allErrs
}

func validateIngressRuleValue(ingressRule *networking.IngressRuleValue, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	if ingressRule.HTTP != nil {
		allErrs = append(allErrs, validateHTTPIngressRuleValue(ingressRule.HTTP, fldPath.Child("http"), opts, requestGV)...)
	}
	return allErrs
}

func validateHTTPIngressRuleValue(httpIngressRuleValue *networking.HTTPIngressRuleValue, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(httpIngressRuleValue.Paths) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("paths"), ""))
	}
	for i, path := range httpIngressRuleValue.Paths {
		allErrs = append(allErrs, validateHTTPIngressPath(&path, fldPath.Child("paths").Index(i), opts, requestGV)...)
	}
	return allErrs
}

func validateHTTPIngressPath(path *networking.HTTPIngressPath, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}

	if path.PathType == nil {
		return append(allErrs, field.Required(fldPath.Child("pathType"), "pathType must be specified"))
	}

	switch *path.PathType {
	case networking.PathTypeExact, networking.PathTypePrefix:
		if !strings.HasPrefix(path.Path, "/") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("path"), path.Path, "must be an absolute path"))
		}
		if len(path.Path) > 0 {
			for _, invalidSeq := range invalidPathSequences {
				if strings.Contains(path.Path, invalidSeq) {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("path"), path.Path, fmt.Sprintf("must not contain '%s'", invalidSeq)))
				}
			}

			for _, invalidSuff := range invalidPathSuffixes {
				if strings.HasSuffix(path.Path, invalidSuff) {
					allErrs = append(allErrs, field.Invalid(fldPath.Child("path"), path.Path, fmt.Sprintf("cannot end with '%s'", invalidSuff)))
				}
			}
		}
	case networking.PathTypeImplementationSpecific:
		if len(path.Path) > 0 {
			if !strings.HasPrefix(path.Path, "/") {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("path"), path.Path, "must be an absolute path"))
			}
		}
	default:
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("pathType"), *path.PathType, supportedPathTypes.List()))
	}
	allErrs = append(allErrs, validateIngressBackend(&path.Backend, fldPath.Child("backend"), opts, requestGV)...)
	return allErrs
}

// numberPortField returns the field path to a service port number field
// relative to a backend struct in the provided GroupVersion
func numberPortField(numberPortFieldPath *field.Path, gv schema.GroupVersion) *field.Path {
	switch gv {
	case networkingv1beta1.SchemeGroupVersion, extensionsv1beta1.SchemeGroupVersion:
		return numberPortFieldPath.Child("servicePort")
	default:
		return numberPortFieldPath.Child("service", "port", "number")
	}
}

// namedPortField returns the field path to a service port name field
// relative to a backend struct in the provided GroupVersion
func namedPortField(namedPortFieldPath *field.Path, gv schema.GroupVersion) *field.Path {
	switch gv {
	case networkingv1beta1.SchemeGroupVersion, extensionsv1beta1.SchemeGroupVersion:
		return namedPortFieldPath.Child("servicePort")
	default:
		return namedPortFieldPath.Child("service", "port", "name")
	}
}

// serviceNameFieldPath returns the name of the field used for a
// service name in the provided GroupVersion.
func serviceNameFieldPath(backendFieldPath *field.Path, gv schema.GroupVersion) *field.Path {
	switch gv {
	case networkingv1beta1.SchemeGroupVersion, extensionsv1beta1.SchemeGroupVersion:
		return backendFieldPath.Child("serviceName")
	default:
		return backendFieldPath.Child("service", "name")
	}
}

// validateIngressBackend tests if a given backend is valid.
func validateIngressBackend(backend *networking.IngressBackend, fldPath *field.Path, opts IngressValidationOptions, requestGV schema.GroupVersion) field.ErrorList {
	allErrs := field.ErrorList{}

	hasResourceBackend := backend.Resource != nil
	hasServiceBackend := backend.Service != nil

	switch {
	case hasResourceBackend && hasServiceBackend:
		return append(allErrs, field.Invalid(fldPath, "", "cannot set both resource and service backends"))
	case hasResourceBackend:
		allErrs = append(allErrs, validateIngressTypedLocalObjectReference(backend.Resource, fldPath.Child("resource"))...)
	case hasServiceBackend:

		if len(backend.Service.Name) == 0 {
			allErrs = append(allErrs, field.Required(serviceNameFieldPath(fldPath, requestGV), ""))
		} else {
			for _, msg := range apivalidation.ValidateServiceName(backend.Service.Name, false) {
				allErrs = append(allErrs, field.Invalid(serviceNameFieldPath(fldPath, requestGV), backend.Service.Name, msg))
			}
		}

		hasPortName := len(backend.Service.Port.Name) > 0
		hasPortNumber := backend.Service.Port.Number != 0
		if hasPortName && hasPortNumber {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "cannot set both port name & port number"))
		} else if hasPortName {
			for _, msg := range validation.IsValidPortName(backend.Service.Port.Name) {
				allErrs = append(allErrs, field.Invalid(namedPortField(fldPath, requestGV), backend.Service.Port.Name, msg))
			}
		} else if hasPortNumber {
			for _, msg := range validation.IsValidPortNum(int(backend.Service.Port.Number)) {
				allErrs = append(allErrs, field.Invalid(numberPortField(fldPath, requestGV), backend.Service.Port.Number, msg))
			}
		} else {
			allErrs = append(allErrs, field.Required(fldPath, "port name or number is required"))
		}
	default:
		allErrs = append(allErrs, field.Invalid(fldPath, "", "resource or service backend is required"))
	}
	return allErrs
}

// ValidateIngressClassName validates that the given name can be used as an
// IngressClass name.
var ValidateIngressClassName = apimachineryvalidation.NameIsDNSSubdomain

// ValidateIngressClass ensures that IngressClass resources are valid.
func ValidateIngressClass(ingressClass *networking.IngressClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ingressClass.ObjectMeta, false, ValidateIngressClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateIngressClassSpec(&ingressClass.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateIngressClassUpdate ensures that IngressClass updates are valid.
func ValidateIngressClassUpdate(newIngressClass, oldIngressClass *networking.IngressClass) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&newIngressClass.ObjectMeta, &oldIngressClass.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateIngressClassSpecUpdate(&newIngressClass.Spec, &oldIngressClass.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateIngressClass(newIngressClass)...)
	return allErrs
}

// validateIngressClassSpec ensures that IngressClassSpec fields are valid.
func validateIngressClassSpec(spec *networking.IngressClassSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(spec.Controller) > maxLenIngressClassController {
		allErrs = append(allErrs, field.TooLong(fldPath.Child("controller"), spec.Controller, maxLenIngressClassController))
	}
	allErrs = append(allErrs, validation.IsDomainPrefixedPath(fldPath.Child("controller"), spec.Controller)...)
	allErrs = append(allErrs, validateIngressTypedLocalObjectReference(spec.Parameters, fldPath.Child("parameters"))...)
	return allErrs
}

// validateIngressClassSpecUpdate ensures that IngressClassSpec updates are
// valid.
func validateIngressClassSpecUpdate(newSpec, oldSpec *networking.IngressClassSpec, fldPath *field.Path) field.ErrorList {
	return apivalidation.ValidateImmutableField(newSpec.Controller, oldSpec.Controller, fldPath.Child("controller"))
}

// validateIngressTypedLocalObjectReference ensures that Parameters fields are valid.
func validateIngressTypedLocalObjectReference(params *api.TypedLocalObjectReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if params == nil {
		return allErrs
	}

	if params.APIGroup != nil {
		for _, msg := range validation.IsDNS1123Subdomain(*params.APIGroup) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("apiGroup"), *params.APIGroup, msg))
		}
	}

	if params.Kind == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("kind"), "kind is required"))
	} else {
		for _, msg := range pathvalidation.IsValidPathSegmentName(params.Kind) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), params.Kind, msg))
		}
	}

	if params.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "name is required"))
	} else {
		for _, msg := range pathvalidation.IsValidPathSegmentName(params.Name) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), params.Name, msg))
		}
	}

	return allErrs
}

func allowInvalidSecretName(gv schema.GroupVersion, oldIngress *networking.Ingress) bool {
	if gv == networkingv1beta1.SchemeGroupVersion || gv == extensionsv1beta1.SchemeGroupVersion {
		// backwards compatibility with released API versions that allowed invalid names
		return true
	}
	if oldIngress != nil {
		for _, tls := range oldIngress.Spec.TLS {
			if len(validateTLSSecretName(tls.SecretName)) > 0 {
				// backwards compatibility with existing persisted object
				return true
			}
		}
	}
	return false
}

func validateTLSSecretName(name string) []string {
	if len(name) == 0 {
		return nil
	}
	return apivalidation.ValidateSecretName(name, false)
}

func allowInvalidWildcardHostRule(gv schema.GroupVersion, oldIngress *networking.Ingress) bool {
	if gv == networkingv1beta1.SchemeGroupVersion || gv == extensionsv1beta1.SchemeGroupVersion {
		// backwards compatibility with released API versions that allowed invalid rules
		return true
	}
	if oldIngress != nil {
		for _, rule := range oldIngress.Spec.Rules {
			if strings.Contains(rule.Host, "*") && len(validateIngressRuleValue(&rule.IngressRuleValue, nil, IngressValidationOptions{}, gv)) > 0 {
				// backwards compatibility with existing invalid data
				return true
			}
		}
	}
	return false
}
