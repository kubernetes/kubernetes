/*
Copyright 2022 The Kubernetes Authors.

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

package service

import (
	"fmt"
	"net/netip"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

func GetWarningsForService(service, oldService *api.Service) []string {
	if service == nil {
		return nil
	}
	var warnings []string

	if _, ok := service.Annotations[api.DeprecatedAnnotationTopologyAwareHints]; ok {
		warnings = append(warnings, fmt.Sprintf("annotation %s is deprecated, please use %s instead", api.DeprecatedAnnotationTopologyAwareHints, api.AnnotationTopologyMode))
	}

	if helper.IsServiceIPSet(service) {
		for i, clusterIP := range service.Spec.ClusterIPs {
			warnings = append(warnings, getWarningsForIP(field.NewPath("spec").Child("clusterIPs").Index(i), clusterIP)...)
		}
	}

	for i, externalIP := range service.Spec.ExternalIPs {
		warnings = append(warnings, getWarningsForIP(field.NewPath("spec").Child("externalIPs").Index(i), externalIP)...)
	}

	if len(service.Spec.LoadBalancerIP) > 0 {
		warnings = append(warnings, getWarningsForIP(field.NewPath("spec").Child("loadBalancerIP"), service.Spec.LoadBalancerIP)...)
	}

	for i, cidr := range service.Spec.LoadBalancerSourceRanges {
		warnings = append(warnings, getWarningsForCIDR(field.NewPath("spec").Child("loadBalancerSourceRanges").Index(i), cidr)...)
	}

	return warnings
}

func getWarningsForIP(fieldPath *field.Path, address string) []string {
	// IPv4 addresses with leading zeros CVE-2021-29923 are not valid in golang since 1.17
	// This will also warn about possible future changes on the golang std library
	// xref: https://issues.k8s.io/108074
	ip, err := netip.ParseAddr(address)
	if err != nil {
		return []string{fmt.Sprintf("%s: IP address was accepted, but will be invalid in a future Kubernetes release: %v", fieldPath, err)}
	}
	// A Recommendation for IPv6 Address Text Representation
	//
	// "All of the above examples represent the same IPv6 address.  This
	// flexibility has caused many problems for operators, systems
	// engineers, and customers.
	// ..."
	// https://datatracker.ietf.org/doc/rfc5952/
	if ip.Is6() && ip.String() != address {
		return []string{fmt.Sprintf("%s: IPv6 address %q is not in RFC 5952 canonical format (%q), which may cause controller apply-loops", fieldPath, address, ip.String())}
	}
	return []string{}
}

func getWarningsForCIDR(fieldPath *field.Path, cidr string) []string {
	// IPv4 addresses with leading zeros CVE-2021-29923 are not valid in golang since 1.17
	// This will also warn about possible future changes on the golang std library
	// xref: https://issues.k8s.io/108074
	prefix, err := netip.ParsePrefix(cidr)
	if err != nil {
		return []string{fmt.Sprintf("%s: IP prefix was accepted, but will be invalid in a future Kubernetes release: %v", fieldPath, err)}
	}
	// A Recommendation for IPv6 Address Text Representation
	//
	// "All of the above examples represent the same IPv6 address.  This
	// flexibility has caused many problems for operators, systems
	// engineers, and customers.
	// ..."
	// https://datatracker.ietf.org/doc/rfc5952/
	if prefix.Addr().Is6() && prefix.String() != cidr {
		return []string{fmt.Sprintf("%s: IPv6 prefix %q is not in RFC 5952 canonical format (%q), which may cause controller apply-loops", fieldPath, cidr, prefix.String())}
	}
	return []string{}
}
