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
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	pathvalidation "k8s.io/apimachinery/pkg/api/validation/path"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/networking"
	netutils "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
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

	supportedIngressClassParametersReferenceScopes = sets.NewString(
		networking.IngressClassParametersReferenceScopeNamespace,
		networking.IngressClassParametersReferenceScopeCluster,
	)
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
			if port.EndPort != nil {
				if *port.EndPort < port.Port.IntVal {
					allErrs = append(allErrs, field.Invalid(portPath.Child("endPort"), port.Port.IntVal, "must be greater than or equal to `port`"))
				}
				for _, msg := range validation.IsValidPortNum(int(*port.EndPort)) {
					allErrs = append(allErrs, field.Invalid(portPath.Child("endPort"), *port.EndPort, msg))
				}
			}
		} else {
			if port.EndPort != nil {
				allErrs = append(allErrs, field.Invalid(portPath.Child("endPort"), *port.EndPort, "may not be specified when `port` is non-numeric"))
			}
			for _, msg := range validation.IsValidPortName(port.Port.StrVal) {
				allErrs = append(allErrs, field.Invalid(portPath.Child("port"), port.Port.StrVal, msg))
			}
		}
	} else {
		if port.EndPort != nil {
			allErrs = append(allErrs, field.Invalid(portPath.Child("endPort"), *port.EndPort, "may not be specified when `port` is not specified"))
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

// ValidateNetworkPolicyStatusUpdate tests if an update to a NetworkPolicy status is valid
func ValidateNetworkPolicyStatusUpdate(status, oldstatus networking.NetworkPolicyStatus, fldPath *field.Path) field.ErrorList {
	return unversionedvalidation.ValidateConditions(status.Conditions, fldPath.Child("conditions"))
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
func validateIngress(ingress *networking.Ingress, opts IngressValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ingress.ObjectMeta, true, ValidateIngressName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, field.NewPath("spec"), opts)...)
	return allErrs
}

// ValidateIngressCreate validates Ingresses on create.
func ValidateIngressCreate(ingress *networking.Ingress) field.ErrorList {
	allErrs := field.ErrorList{}
	opts := IngressValidationOptions{
		AllowInvalidSecretName:       false,
		AllowInvalidWildcardHostRule: false,
	}
	allErrs = append(allErrs, validateIngress(ingress, opts)...)
	annotationVal, annotationIsSet := ingress.Annotations[annotationIngressClass]
	if annotationIsSet && ingress.Spec.IngressClassName != nil {
		annotationPath := field.NewPath("annotations").Child(annotationIngressClass)
		allErrs = append(allErrs, field.Invalid(annotationPath, annotationVal, "can not be set when the class field is also set"))
	}
	return allErrs
}

// ValidateIngressUpdate validates ingresses on update.
func ValidateIngressUpdate(ingress, oldIngress *networking.Ingress) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, field.NewPath("metadata"))
	opts := IngressValidationOptions{
		AllowInvalidSecretName:       allowInvalidSecretName(oldIngress),
		AllowInvalidWildcardHostRule: allowInvalidWildcardHostRule(oldIngress),
	}

	allErrs = append(allErrs, validateIngress(ingress, opts)...)
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

// ValidateIngressSpec tests if required fields in the IngressSpec are set.
func ValidateIngressSpec(spec *networking.IngressSpec, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(spec.Rules) == 0 && spec.DefaultBackend == nil {
		errMsg := fmt.Sprintf("either `%s` or `rules` must be specified", "defaultBackend")
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Rules, errMsg))
	}
	if spec.DefaultBackend != nil {
		allErrs = append(allErrs, validateIngressBackend(spec.DefaultBackend, fldPath.Child("defaultBackend"), opts)...)
	}
	if len(spec.Rules) > 0 {
		allErrs = append(allErrs, validateIngressRules(spec.Rules, fldPath.Child("rules"), opts)...)
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

func validateIngressRules(ingressRules []networking.IngressRule, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(ingressRules) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}
	for i, ih := range ingressRules {
		wildcardHost := false
		if len(ih.Host) > 0 {
			if isIP := (netutils.ParseIPSloppy(ih.Host) != nil); isIP {
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
			allErrs = append(allErrs, validateIngressRuleValue(&ih.IngressRuleValue, fldPath.Index(i), opts)...)
		}
	}
	return allErrs
}

func validateIngressRuleValue(ingressRule *networking.IngressRuleValue, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if ingressRule.HTTP != nil {
		allErrs = append(allErrs, validateHTTPIngressRuleValue(ingressRule.HTTP, fldPath.Child("http"), opts)...)
	}
	return allErrs
}

func validateHTTPIngressRuleValue(httpIngressRuleValue *networking.HTTPIngressRuleValue, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(httpIngressRuleValue.Paths) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("paths"), ""))
	}
	for i, path := range httpIngressRuleValue.Paths {
		allErrs = append(allErrs, validateHTTPIngressPath(&path, fldPath.Child("paths").Index(i), opts)...)
	}
	return allErrs
}

func validateHTTPIngressPath(path *networking.HTTPIngressPath, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
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
	allErrs = append(allErrs, validateIngressBackend(&path.Backend, fldPath.Child("backend"), opts)...)
	return allErrs
}

// validateIngressBackend tests if a given backend is valid.
func validateIngressBackend(backend *networking.IngressBackend, fldPath *field.Path, opts IngressValidationOptions) field.ErrorList {
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
			allErrs = append(allErrs, field.Required(fldPath.Child("service", "name"), ""))
		} else {
			for _, msg := range apivalidation.ValidateServiceName(backend.Service.Name, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("service", "name"), backend.Service.Name, msg))
			}
		}

		hasPortName := len(backend.Service.Port.Name) > 0
		hasPortNumber := backend.Service.Port.Number != 0
		if hasPortName && hasPortNumber {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "cannot set both port name & port number"))
		} else if hasPortName {
			for _, msg := range validation.IsValidPortName(backend.Service.Port.Name) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("service", "port", "name"), backend.Service.Port.Name, msg))
			}
		} else if hasPortNumber {
			for _, msg := range validation.IsValidPortNum(int(backend.Service.Port.Number)) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("service", "port", "number"), backend.Service.Port.Number, msg))
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
	allErrs = append(allErrs, validateIngressClassParametersReference(spec.Parameters, fldPath.Child("parameters"))...)
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

// validateIngressClassParametersReference ensures that Parameters fields are valid.
// Parameters was previously of type `TypedLocalObjectReference` and used
// `validateIngressTypedLocalObjectReference()`. This function extends validation
// for additional fields introduced for namespace-scoped references.
func validateIngressClassParametersReference(params *networking.IngressClassParametersReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if params == nil {
		return allErrs
	}

	allErrs = append(allErrs, validateIngressTypedLocalObjectReference(&api.TypedLocalObjectReference{
		APIGroup: params.APIGroup,
		Kind:     params.Kind,
		Name:     params.Name,
	}, fldPath)...)

	if params.Scope == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("scope"), ""))
		return allErrs
	}

	if params.Scope != nil || params.Namespace != nil {
		scope := utilpointer.StringPtrDerefOr(params.Scope, "")

		if !supportedIngressClassParametersReferenceScopes.Has(scope) {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("scope"), scope,
				supportedIngressClassParametersReferenceScopes.List()))
		} else {

			if scope == networking.IngressClassParametersReferenceScopeNamespace {
				if params.Namespace == nil {
					allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), "`parameters.scope` is set to 'Namespace'"))
				} else {
					for _, msg := range apivalidation.ValidateNamespaceName(*params.Namespace, false) {
						allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), *params.Namespace, msg))
					}
				}
			}

			if scope == networking.IngressClassParametersReferenceScopeCluster && params.Namespace != nil {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("namespace"), "`parameters.scope` is set to 'Cluster'"))
			}
		}
	}

	return allErrs
}

func allowInvalidSecretName(oldIngress *networking.Ingress) bool {
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

func allowInvalidWildcardHostRule(oldIngress *networking.Ingress) bool {
	if oldIngress != nil {
		for _, rule := range oldIngress.Spec.Rules {
			if strings.Contains(rule.Host, "*") && len(validateIngressRuleValue(&rule.IngressRuleValue, nil, IngressValidationOptions{})) > 0 {
				// backwards compatibility with existing invalid data
				return true
			}
		}
	}
	return false
}

// ValidateClusterCIDRConfigName validates that the given name can be used as an
// ClusterCIDRConfig name.
var ValidateClusterCIDRConfigName = apimachineryvalidation.NameIsDNSLabel

// ValidateClusterCIDRConfig validates a clusterCIDRConfig.
func ValidateClusterCIDRConfig(ccc *networking.ClusterCIDRConfig) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&ccc.ObjectMeta, false, ValidateClusterCIDRConfigName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateClusterCIDRConfigSpec(&ccc.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateClusterCIDRConfigSpec validates clusterCIDRConfig Spec.
func ValidateClusterCIDRConfigSpec(spec *networking.ClusterCIDRConfigSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.NodeSelector != nil {
		allErrs = append(allErrs, apivalidation.ValidateNodeSelector(spec.NodeSelector, fldPath.Child("nodeSelector"))...)
	}

	maxIPv4PerNodeMaskSize := int32(32)
	maxIPv6PerNodeMaskSize := int32(128)

	hasIPv4 := spec.IPv4 != nil
	hasIPv6 := spec.IPv6 != nil

	// Validate if CIDR is configured for at least one IP Family(IPv4/IPv6).
	if !hasIPv4 && !hasIPv6 {
		allErrs = append(allErrs, field.Required(fldPath, "one or both of `ipv4` and `ipv6` must be configured"))
	}

	// Validate configured IPv4 CIDR and PerNodeMaskSize.
	if hasIPv4 {
		allErrs = append(allErrs, validateCIDRConfig(spec.IPv4, fldPath, maxIPv4PerNodeMaskSize, "ipv4")...)
	}

	// Validate configured IPv6 CIDR and PerNodeMaskSize.
	if hasIPv6 {
		allErrs = append(allErrs, validateCIDRConfig(spec.IPv6, fldPath, maxIPv6PerNodeMaskSize, "ipv6")...)
	}

	// Validate PerNodeMaskSize if both IPv4 and IPv6 CIDRs are configured.
	// IPv4.PerNodeMaskSize and IPv6.PerNodeMaskSize must specify the same number of IP addresses:
	// 32 - spec.IPv4.PerNodeMaskSize == 128 - spec.IPv6.PerNodeMaskSize
	// Unequal allocatable IP addresses will lead to wastage of IP addresses.
	// Also, this being one of the tie-breaker rules for ClusterCIDConfigs, it is
	// necessary to have equal number of allocatable IP addresses per node.
	if hasIPv4 && hasIPv6 {
		if maxIPv4PerNodeMaskSize-spec.IPv4.PerNodeMaskSize != maxIPv6PerNodeMaskSize-spec.IPv6.PerNodeMaskSize {
			allErrs = append(allErrs, field.Invalid(fldPath, *spec, "ipv4.perNodeMaskSize and ipv6.perNodeMaskSize must allocate the same number of IPs"))
		}
	}
	return allErrs
}

func validateCIDRConfig(cidrConfig *networking.CIDRConfig, fldPath *field.Path, maxPerNodeMaskSize int32, ipFamily string) field.ErrorList {
	allErrs := field.ErrorList{}
	if cidrConfig.CIDR == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child(ipFamily, "cidr"), ""))
		return allErrs
	} else {
		switch ipFamily {
		case "ipv4":
			if !netutils.IsIPv4CIDRString(cidrConfig.CIDR) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child(ipFamily, "cidr"), *cidrConfig, "must be a valid IPv4 CIDR"))
				return allErrs
			}
		case "ipv6":
			if !netutils.IsIPv6CIDRString(cidrConfig.CIDR) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child(ipFamily, "cidr"), *cidrConfig, "must be a valid IPv6 CIDR"))
				return allErrs
			}
		}
	}

	if cidrConfig.PerNodeMaskSize == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child(ipFamily, "perNodeMaskSize"), ""))
	} else {
		_, cidr, _ := netutils.ParseCIDRSloppy(cidrConfig.CIDR)
		maskSize, _ := cidr.Mask.Size()
		if cidrConfig.PerNodeMaskSize < int32(maskSize) || cidrConfig.PerNodeMaskSize > maxPerNodeMaskSize {
			allErrs = append(allErrs, field.Invalid(fldPath.Child(ipFamily, "perNodeMaskSize"), *cidrConfig, fmt.Sprintf("must be greater than %d and less than or equal to %d", maskSize, maxPerNodeMaskSize)))
		}
	}

	return allErrs
}

// ValidateClusterCIDRConfigUpdate tests if an update to a ClusterCIDRConfig is valid.
func ValidateClusterCIDRConfigUpdate(update, old *networking.ClusterCIDRConfig) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateClusterCIDRConfigUpdateSpec(&update.Spec, &old.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateClusterCIDRConfigUpdateSpec(update, old *networking.ClusterCIDRConfigSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, apivalidation.ValidateImmutableField(update.NodeSelector, old.NodeSelector, fldPath.Child("nodeSelector"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(update.IPv4, old.IPv4, fldPath.Child("ipv4"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(update.IPv6, old.IPv6, fldPath.Child("ipv6"))...)

	return allErrs
}
