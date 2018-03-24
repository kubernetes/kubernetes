/*
Copyright 2014 The Kubernetes Authors.

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

	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/master/ports"
)

// ValidateServiceName can be used to check whether the given service name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateServiceName = NameIsDNS1035Label

var supportedSessionAffinityType = sets.NewString(string(core.ServiceAffinityClientIP), string(core.ServiceAffinityNone))
var supportedServiceType = sets.NewString(string(core.ServiceTypeClusterIP), string(core.ServiceTypeNodePort),
	string(core.ServiceTypeLoadBalancer), string(core.ServiceTypeExternalName))

// ValidateService tests if required fields/annotations of a Service are valid.
func ValidateService(service *core.Service) field.ErrorList {
	allErrs := ValidateObjectMeta(&service.ObjectMeta, true, ValidateServiceName, field.NewPath("metadata"))

	specPath := field.NewPath("spec")
	isHeadlessService := service.Spec.ClusterIP == core.ClusterIPNone
	if len(service.Spec.Ports) == 0 && !isHeadlessService && service.Spec.Type != core.ServiceTypeExternalName {
		allErrs = append(allErrs, field.Required(specPath.Child("ports"), ""))
	}
	switch service.Spec.Type {
	case core.ServiceTypeLoadBalancer:
		for ix := range service.Spec.Ports {
			port := &service.Spec.Ports[ix]
			// This is a workaround for broken cloud environments that
			// over-open firewalls.  Hopefully it can go away when more clouds
			// understand containers better.
			if port.Port == ports.KubeletPort {
				portPath := specPath.Child("ports").Index(ix)
				allErrs = append(allErrs, field.Invalid(portPath, port.Port, fmt.Sprintf("may not expose port %v externally since it is used by kubelet", ports.KubeletPort)))
			}
		}
		if service.Spec.ClusterIP == "None" {
			allErrs = append(allErrs, field.Invalid(specPath.Child("clusterIP"), service.Spec.ClusterIP, "may not be set to 'None' for LoadBalancer services"))
		}
	case core.ServiceTypeNodePort:
		if service.Spec.ClusterIP == "None" {
			allErrs = append(allErrs, field.Invalid(specPath.Child("clusterIP"), service.Spec.ClusterIP, "may not be set to 'None' for NodePort services"))
		}
	case core.ServiceTypeExternalName:
		if service.Spec.ClusterIP != "" {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("clusterIP"), "must be empty for ExternalName services"))
		}
		if len(service.Spec.ExternalName) > 0 {
			allErrs = append(allErrs, ValidateDNS1123Subdomain(service.Spec.ExternalName, specPath.Child("externalName"))...)
		} else {
			allErrs = append(allErrs, field.Required(specPath.Child("externalName"), ""))
		}
	}

	allPortNames := sets.String{}
	portsPath := specPath.Child("ports")
	for i := range service.Spec.Ports {
		portPath := portsPath.Index(i)
		allErrs = append(allErrs, validateServicePort(&service.Spec.Ports[i], len(service.Spec.Ports) > 1, isHeadlessService, &allPortNames, portPath)...)
	}

	if service.Spec.Selector != nil {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabels(service.Spec.Selector, specPath.Child("selector"))...)
	}

	if len(service.Spec.SessionAffinity) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("sessionAffinity"), ""))
	} else if !supportedSessionAffinityType.Has(string(service.Spec.SessionAffinity)) {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("sessionAffinity"), service.Spec.SessionAffinity, supportedSessionAffinityType.List()))
	}

	if service.Spec.SessionAffinity == core.ServiceAffinityClientIP {
		allErrs = append(allErrs, validateClientIPAffinityConfig(service.Spec.SessionAffinityConfig, specPath.Child("sessionAffinityConfig"))...)
	} else if service.Spec.SessionAffinity == core.ServiceAffinityNone {
		if service.Spec.SessionAffinityConfig != nil {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("sessionAffinityConfig"), fmt.Sprintf("must not be set when session affinity is %s", string(core.ServiceAffinityNone))))
		}
	}

	if helper.IsServiceIPSet(service) {
		if ip := net.ParseIP(service.Spec.ClusterIP); ip == nil {
			allErrs = append(allErrs, field.Invalid(specPath.Child("clusterIP"), service.Spec.ClusterIP, "must be empty, 'None', or a valid IP address"))
		}
	}

	ipPath := specPath.Child("externalIPs")
	for i, ip := range service.Spec.ExternalIPs {
		idxPath := ipPath.Index(i)
		if msgs := validation.IsValidIP(ip); len(msgs) != 0 {
			for i := range msgs {
				allErrs = append(allErrs, field.Invalid(idxPath, ip, msgs[i]))
			}
		} else {
			allErrs = append(allErrs, validateNonSpecialIP(ip, idxPath)...)
		}
	}

	if len(service.Spec.Type) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("type"), ""))
	} else if !supportedServiceType.Has(string(service.Spec.Type)) {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("type"), service.Spec.Type, supportedServiceType.List()))
	}

	if service.Spec.Type == core.ServiceTypeLoadBalancer {
		portsPath := specPath.Child("ports")
		includeProtocols := sets.NewString()
		for i := range service.Spec.Ports {
			portPath := portsPath.Index(i)
			if !supportedPortProtocols.Has(string(service.Spec.Ports[i].Protocol)) {
				allErrs = append(allErrs, field.Invalid(portPath.Child("protocol"), service.Spec.Ports[i].Protocol, "cannot create an external load balancer with non-TCP/UDP ports"))
			} else {
				includeProtocols.Insert(string(service.Spec.Ports[i].Protocol))
			}
		}
		if includeProtocols.Len() > 1 {
			allErrs = append(allErrs, field.Invalid(portsPath, service.Spec.Ports, "cannot create an external load balancer with mix protocols"))
		}
	}

	if service.Spec.Type == core.ServiceTypeClusterIP {
		portsPath := specPath.Child("ports")
		for i := range service.Spec.Ports {
			portPath := portsPath.Index(i)
			if service.Spec.Ports[i].NodePort != 0 {
				allErrs = append(allErrs, field.Forbidden(portPath.Child("nodePort"), "may not be used when `type` is 'ClusterIP'"))
			}
		}
	}

	// Check for duplicate NodePorts, considering (protocol,port) pairs
	portsPath = specPath.Child("ports")
	nodePorts := make(map[core.ServicePort]bool)
	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		if port.NodePort == 0 {
			continue
		}
		portPath := portsPath.Index(i)
		var key core.ServicePort
		key.Protocol = port.Protocol
		key.NodePort = port.NodePort
		_, found := nodePorts[key]
		if found {
			allErrs = append(allErrs, field.Duplicate(portPath.Child("nodePort"), port.NodePort))
		}
		nodePorts[key] = true
	}

	// Check for duplicate Ports, considering (protocol,port) pairs
	portsPath = specPath.Child("ports")
	ports := make(map[core.ServicePort]bool)
	for i, port := range service.Spec.Ports {
		portPath := portsPath.Index(i)
		key := core.ServicePort{Protocol: port.Protocol, Port: port.Port}
		_, found := ports[key]
		if found {
			allErrs = append(allErrs, field.Duplicate(portPath, key))
		}
		ports[key] = true
	}

	// Validate SourceRange field and annotation
	_, ok := service.Annotations[core.AnnotationLoadBalancerSourceRangesKey]
	if len(service.Spec.LoadBalancerSourceRanges) > 0 || ok {
		var fieldPath *field.Path
		var val string
		if len(service.Spec.LoadBalancerSourceRanges) > 0 {
			fieldPath = specPath.Child("LoadBalancerSourceRanges")
			val = fmt.Sprintf("%v", service.Spec.LoadBalancerSourceRanges)
		} else {
			fieldPath = field.NewPath("metadata", "annotations").Key(core.AnnotationLoadBalancerSourceRangesKey)
			val = service.Annotations[core.AnnotationLoadBalancerSourceRangesKey]
		}
		if service.Spec.Type != core.ServiceTypeLoadBalancer {
			allErrs = append(allErrs, field.Forbidden(fieldPath, "may only be used when `type` is 'LoadBalancer'"))
		}
		_, err := apiservice.GetLoadBalancerSourceRanges(service)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fieldPath, val, "must be a list of IP ranges. For example, 10.240.0.0/24,10.250.0.0/24 "))
		}
	}

	allErrs = append(allErrs, validateServiceExternalTrafficFieldsValue(service)...)

	return allErrs
}

// ValidateServiceUpdate tests if required fields in the service are set during an update
func ValidateServiceUpdate(service, oldService *core.Service) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&service.ObjectMeta, &oldService.ObjectMeta, field.NewPath("metadata"))

	// ClusterIP should be immutable for services using it (every type other than ExternalName)
	// which do not have ClusterIP assigned yet (empty string value)
	if service.Spec.Type != core.ServiceTypeExternalName {
		if oldService.Spec.Type != core.ServiceTypeExternalName && oldService.Spec.ClusterIP != "" {
			allErrs = append(allErrs, ValidateImmutableField(service.Spec.ClusterIP, oldService.Spec.ClusterIP, field.NewPath("spec", "clusterIP"))...)
		}
	}

	allErrs = append(allErrs, ValidateService(service)...)
	return allErrs
}

// ValidateServiceStatusUpdate tests if required fields in the Service are set when updating status.
func ValidateServiceStatusUpdate(service, oldService *core.Service) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&service.ObjectMeta, &oldService.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLoadBalancerStatus(&service.Status.LoadBalancer, field.NewPath("status", "loadBalancer"))...)
	return allErrs
}

// ValidateServiceExternalTrafficFieldsCombination validates if ExternalTrafficPolicy,
// HealthCheckNodePort and Type combination are legal. For update, it should be called
// after clearing externalTraffic related fields for the ease of transitioning between
// different service types.
func ValidateServiceExternalTrafficFieldsCombination(service *core.Service) field.ErrorList {
	allErrs := field.ErrorList{}

	if service.Spec.Type != core.ServiceTypeLoadBalancer &&
		service.Spec.Type != core.ServiceTypeNodePort &&
		service.Spec.ExternalTrafficPolicy != "" {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "externalTrafficPolicy"), service.Spec.ExternalTrafficPolicy,
			"ExternalTrafficPolicy can only be set on NodePort and LoadBalancer service"))
	}

	if !apiservice.NeedsHealthCheck(service) &&
		service.Spec.HealthCheckNodePort != 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "healthCheckNodePort"), service.Spec.HealthCheckNodePort,
			"HealthCheckNodePort can only be set on LoadBalancer service with ExternalTrafficPolicy=Local"))
	}

	return allErrs
}

// ValidateLoadBalancerStatus validates required fields on a LoadBalancerStatus
func ValidateLoadBalancerStatus(status *core.LoadBalancerStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, ingress := range status.Ingress {
		idxPath := fldPath.Child("ingress").Index(i)
		if len(ingress.IP) > 0 {
			if isIP := (net.ParseIP(ingress.IP) != nil); !isIP {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("ip"), ingress.IP, "must be a valid IP address"))
			}
		}
		if len(ingress.Hostname) > 0 {
			for _, msg := range validation.IsDNS1123Subdomain(ingress.Hostname) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostname"), ingress.Hostname, msg))
			}
			if isIP := (net.ParseIP(ingress.Hostname) != nil); isIP {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostname"), ingress.Hostname, "must be a DNS name, not an IP address"))
			}
		}
	}
	return allErrs
}

func ValidatePortNumOrName(port intstr.IntOrString, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if port.Type == intstr.Int {
		for _, msg := range validation.IsValidPortNum(port.IntValue()) {
			allErrs = append(allErrs, field.Invalid(fldPath, port.IntValue(), msg))
		}
	} else if port.Type == intstr.String {
		for _, msg := range validation.IsValidPortName(port.StrVal) {
			allErrs = append(allErrs, field.Invalid(fldPath, port.StrVal, msg))
		}
	} else {
		allErrs = append(allErrs, field.InternalError(fldPath, fmt.Errorf("unknown type: %v", port.Type)))
	}
	return allErrs
}

func validateServicePort(sp *core.ServicePort, requireName, isHeadlessService bool, allNames *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if requireName && len(sp.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else if len(sp.Name) != 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(sp.Name, fldPath.Child("name"))...)
		if allNames.Has(sp.Name) {
			allErrs = append(allErrs, field.Duplicate(fldPath.Child("name"), sp.Name))
		} else {
			allNames.Insert(sp.Name)
		}
	}

	for _, msg := range validation.IsValidPortNum(int(sp.Port)) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("port"), sp.Port, msg))
	}

	if len(sp.Protocol) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("protocol"), ""))
	} else if !supportedPortProtocols.Has(string(sp.Protocol)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("protocol"), sp.Protocol, supportedPortProtocols.List()))
	}

	allErrs = append(allErrs, ValidatePortNumOrName(sp.TargetPort, fldPath.Child("targetPort"))...)

	// in the v1 API, targetPorts on headless services were tolerated.
	// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
	//
	// if isHeadlessService {
	// 	if sp.TargetPort.Type == intstr.String || (sp.TargetPort.Type == intstr.Int && sp.Port != sp.TargetPort.IntValue()) {
	// 		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetPort"), sp.TargetPort, "must be equal to the value of 'port' when clusterIP = None"))
	// 	}
	// }

	return allErrs
}

// validateServiceExternalTrafficFieldsValue validates ExternalTraffic related annotations
// have legal value.
func validateServiceExternalTrafficFieldsValue(service *core.Service) field.ErrorList {
	allErrs := field.ErrorList{}

	// Check first class fields.
	if service.Spec.ExternalTrafficPolicy != "" &&
		service.Spec.ExternalTrafficPolicy != core.ServiceExternalTrafficPolicyTypeCluster &&
		service.Spec.ExternalTrafficPolicy != core.ServiceExternalTrafficPolicyTypeLocal {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec").Child("externalTrafficPolicy"), service.Spec.ExternalTrafficPolicy,
			fmt.Sprintf("ExternalTrafficPolicy must be empty, %v or %v", core.ServiceExternalTrafficPolicyTypeCluster, core.ServiceExternalTrafficPolicyTypeLocal)))
	}
	if service.Spec.HealthCheckNodePort < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("spec").Child("healthCheckNodePort"), service.Spec.HealthCheckNodePort,
			"HealthCheckNodePort must be not less than 0"))
	}

	return allErrs
}

func validateClientIPAffinityConfig(config *core.SessionAffinityConfig, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if config == nil {
		allErrs = append(allErrs, field.Required(fldPath, fmt.Sprintf("when session affinity type is %s", core.ServiceAffinityClientIP)))
		return allErrs
	}
	if config.ClientIP == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("clientIP"), fmt.Sprintf("when session affinity type is %s", core.ServiceAffinityClientIP)))
		return allErrs
	}
	if config.ClientIP.TimeoutSeconds == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("clientIP").Child("timeoutSeconds"), fmt.Sprintf("when session affinity type is %s", core.ServiceAffinityClientIP)))
		return allErrs
	}
	allErrs = append(allErrs, validateAffinityTimeout(config.ClientIP.TimeoutSeconds, fldPath.Child("clientIP").Child("timeoutSeconds"))...)

	return allErrs
}

func validateAffinityTimeout(timeout *int32, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if *timeout <= 0 || *timeout > core.MaxClientIPServiceAffinitySeconds {
		allErrs = append(allErrs, field.Invalid(fldPath, timeout, fmt.Sprintf("must be greater than 0 and less than %d", core.MaxClientIPServiceAffinitySeconds)))
	}
	return allErrs
}
