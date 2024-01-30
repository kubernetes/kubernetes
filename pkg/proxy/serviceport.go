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

package proxy

import (
	"fmt"
	"net"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	apiservice "k8s.io/kubernetes/pkg/api/v1/service"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	netutils "k8s.io/utils/net"
)

// ServicePort is an interface which abstracts information about a service.
type ServicePort interface {
	// String returns service string.  An example format can be: `IP:Port/Protocol`.
	String() string
	// ClusterIP returns service cluster IP in net.IP format.
	ClusterIP() net.IP
	// Port returns service port if present. If return 0 means not present.
	Port() int
	// SessionAffinityType returns service session affinity type
	SessionAffinityType() v1.ServiceAffinity
	// StickyMaxAgeSeconds returns service max connection age
	StickyMaxAgeSeconds() int
	// ExternalIPs returns service ExternalIPs
	ExternalIPs() []net.IP
	// LoadBalancerVIPs returns service LoadBalancerIPs which are VIP mode
	LoadBalancerVIPs() []net.IP
	// Protocol returns service protocol.
	Protocol() v1.Protocol
	// LoadBalancerSourceRanges returns service LoadBalancerSourceRanges if present empty array if not
	LoadBalancerSourceRanges() []*net.IPNet
	// HealthCheckNodePort returns service health check node port if present.  If return 0, it means not present.
	HealthCheckNodePort() int
	// NodePort returns a service Node port if present. If return 0, it means not present.
	NodePort() int
	// ExternalPolicyLocal returns if a service has only node local endpoints for external traffic.
	ExternalPolicyLocal() bool
	// InternalPolicyLocal returns if a service has only node local endpoints for internal traffic.
	InternalPolicyLocal() bool
	// HintsAnnotation returns the value of the v1.DeprecatedAnnotationTopologyAwareHints annotation.
	HintsAnnotation() string
	// ExternallyAccessible returns true if the service port is reachable via something
	// other than ClusterIP (NodePort/ExternalIP/LoadBalancer)
	ExternallyAccessible() bool
	// UsesClusterEndpoints returns true if the service port ever sends traffic to
	// endpoints based on "Cluster" traffic policy
	UsesClusterEndpoints() bool
	// UsesLocalEndpoints returns true if the service port ever sends traffic to
	// endpoints based on "Local" traffic policy
	UsesLocalEndpoints() bool
}

// BaseServicePortInfo contains base information that defines a service.
// This could be used directly by proxier while processing services,
// or can be used for constructing a more specific ServiceInfo struct
// defined by the proxier if needed.
type BaseServicePortInfo struct {
	clusterIP                net.IP
	port                     int
	protocol                 v1.Protocol
	nodePort                 int
	loadBalancerVIPs         []net.IP
	sessionAffinityType      v1.ServiceAffinity
	stickyMaxAgeSeconds      int
	externalIPs              []net.IP
	loadBalancerSourceRanges []*net.IPNet
	healthCheckNodePort      int
	externalPolicyLocal      bool
	internalPolicyLocal      bool
	hintsAnnotation          string
}

var _ ServicePort = &BaseServicePortInfo{}

// String is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) String() string {
	return fmt.Sprintf("%s:%d/%s", bsvcPortInfo.clusterIP, bsvcPortInfo.port, bsvcPortInfo.protocol)
}

// ClusterIP is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) ClusterIP() net.IP {
	return bsvcPortInfo.clusterIP
}

// Port is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) Port() int {
	return bsvcPortInfo.port
}

// SessionAffinityType is part of the ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) SessionAffinityType() v1.ServiceAffinity {
	return bsvcPortInfo.sessionAffinityType
}

// StickyMaxAgeSeconds is part of the ServicePort interface
func (bsvcPortInfo *BaseServicePortInfo) StickyMaxAgeSeconds() int {
	return bsvcPortInfo.stickyMaxAgeSeconds
}

// Protocol is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) Protocol() v1.Protocol {
	return bsvcPortInfo.protocol
}

// LoadBalancerSourceRanges is part of ServicePort interface
func (bsvcPortInfo *BaseServicePortInfo) LoadBalancerSourceRanges() []*net.IPNet {
	return bsvcPortInfo.loadBalancerSourceRanges
}

// HealthCheckNodePort is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) HealthCheckNodePort() int {
	return bsvcPortInfo.healthCheckNodePort
}

// NodePort is part of the ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) NodePort() int {
	return bsvcPortInfo.nodePort
}

// ExternalIPs is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) ExternalIPs() []net.IP {
	return bsvcPortInfo.externalIPs
}

// LoadBalancerVIPs is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) LoadBalancerVIPs() []net.IP {
	return bsvcPortInfo.loadBalancerVIPs
}

// ExternalPolicyLocal is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) ExternalPolicyLocal() bool {
	return bsvcPortInfo.externalPolicyLocal
}

// InternalPolicyLocal is part of ServicePort interface
func (bsvcPortInfo *BaseServicePortInfo) InternalPolicyLocal() bool {
	return bsvcPortInfo.internalPolicyLocal
}

// HintsAnnotation is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) HintsAnnotation() string {
	return bsvcPortInfo.hintsAnnotation
}

// ExternallyAccessible is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) ExternallyAccessible() bool {
	return bsvcPortInfo.nodePort != 0 || len(bsvcPortInfo.loadBalancerVIPs) != 0 || len(bsvcPortInfo.externalIPs) != 0
}

// UsesClusterEndpoints is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) UsesClusterEndpoints() bool {
	// The service port uses Cluster endpoints if the internal traffic policy is "Cluster",
	// or if it accepts external traffic at all. (Even if the external traffic policy is
	// "Local", we need Cluster endpoints to implement short circuiting.)
	return !bsvcPortInfo.internalPolicyLocal || bsvcPortInfo.ExternallyAccessible()
}

// UsesLocalEndpoints is part of ServicePort interface.
func (bsvcPortInfo *BaseServicePortInfo) UsesLocalEndpoints() bool {
	return bsvcPortInfo.internalPolicyLocal || (bsvcPortInfo.externalPolicyLocal && bsvcPortInfo.ExternallyAccessible())
}

func newBaseServiceInfo(service *v1.Service, ipFamily v1.IPFamily, port *v1.ServicePort) *BaseServicePortInfo {
	externalPolicyLocal := apiservice.ExternalPolicyLocal(service)
	internalPolicyLocal := apiservice.InternalPolicyLocal(service)

	var stickyMaxAgeSeconds int
	if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP {
		// Kube-apiserver side guarantees SessionAffinityConfig won't be nil when session affinity type is ClientIP
		stickyMaxAgeSeconds = int(*service.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	}

	clusterIP := proxyutil.GetClusterIPByFamily(ipFamily, service)
	info := &BaseServicePortInfo{
		clusterIP:           netutils.ParseIPSloppy(clusterIP),
		port:                int(port.Port),
		protocol:            port.Protocol,
		nodePort:            int(port.NodePort),
		sessionAffinityType: service.Spec.SessionAffinity,
		stickyMaxAgeSeconds: stickyMaxAgeSeconds,
		externalPolicyLocal: externalPolicyLocal,
		internalPolicyLocal: internalPolicyLocal,
	}

	// v1.DeprecatedAnnotationTopologyAwareHints has precedence over v1.AnnotationTopologyMode.
	var exists bool
	info.hintsAnnotation, exists = service.Annotations[v1.DeprecatedAnnotationTopologyAwareHints]
	if !exists {
		info.hintsAnnotation = service.Annotations[v1.AnnotationTopologyMode]
	}

	// filter external ips, source ranges and ingress ips
	// prior to dual stack services, this was considered an error, but with dual stack
	// services, this is actually expected. Hence we downgraded from reporting by events
	// to just log lines with high verbosity
	ipFamilyMap := proxyutil.MapIPsByIPFamily(service.Spec.ExternalIPs)
	info.externalIPs = ipFamilyMap[ipFamily]

	// Log the IPs not matching the ipFamily
	if ips, ok := ipFamilyMap[proxyutil.OtherIPFamily(ipFamily)]; ok && len(ips) > 0 {
		klog.V(4).InfoS("Service change tracker ignored the following external IPs for given service as they don't match IP Family",
			"ipFamily", ipFamily, "externalIPs", ips, "service", klog.KObj(service))
	}

	cidrFamilyMap := proxyutil.MapCIDRsByIPFamily(service.Spec.LoadBalancerSourceRanges)
	info.loadBalancerSourceRanges = cidrFamilyMap[ipFamily]
	// Log the CIDRs not matching the ipFamily
	if cidrs, ok := cidrFamilyMap[proxyutil.OtherIPFamily(ipFamily)]; ok && len(cidrs) > 0 {
		klog.V(4).InfoS("Service change tracker ignored the following load balancer source ranges for given Service as they don't match IP Family",
			"ipFamily", ipFamily, "loadBalancerSourceRanges", cidrs, "service", klog.KObj(service))
	}

	// Obtain Load Balancer Ingress
	var invalidIPs []net.IP
	for _, ing := range service.Status.LoadBalancer.Ingress {
		if ing.IP == "" {
			continue
		}

		// proxy mode load balancers do not need to track the IPs in the service cache
		// and they can also implement IP family translation, so no need to check if
		// the status ingress.IP and the ClusterIP belong to the same family.
		if !proxyutil.IsVIPMode(ing) {
			klog.V(4).InfoS("Service change tracker ignored the following load balancer ingress IP for given Service as it using Proxy mode",
				"ipFamily", ipFamily, "loadBalancerIngressIP", ing.IP, "service", klog.KObj(service))
			continue
		}

		// kube-proxy does not implement IP family translation, skip addresses with
		// different IP family
		ip := netutils.ParseIPSloppy(ing.IP) // (already verified as an IP-address)
		if ingFamily := proxyutil.GetIPFamilyFromIP(ip); ingFamily == ipFamily {
			info.loadBalancerVIPs = append(info.loadBalancerVIPs, ip)
		} else {
			invalidIPs = append(invalidIPs, ip)
		}
	}
	if len(invalidIPs) > 0 {
		klog.V(4).InfoS("Service change tracker ignored the following load balancer ingress IPs for given Service as they don't match the IP Family",
			"ipFamily", ipFamily, "loadBalancerIngressIPs", invalidIPs, "service", klog.KObj(service))
	}

	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p == 0 {
			klog.ErrorS(nil, "Service has no healthcheck nodeport", "service", klog.KObj(service))
		} else {
			info.healthCheckNodePort = int(p)
		}
	}

	return info
}
