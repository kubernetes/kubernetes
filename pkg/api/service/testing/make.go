/*
Copyright 2021 The Kubernetes Authors.

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

package testing

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilpointer "k8s.io/utils/pointer"

	api "k8s.io/kubernetes/pkg/apis/core"
)

// Tweak is a function that modifies a Service.
type Tweak func(*api.Service)

// MakeService helps construct Service objects (which pass API validation) more
// legibly and tersely than a Go struct definition.  By default this produces
// a ClusterIP service with a single port and a trivial selector.  The caller
// can pass any number of tweak functions to further modify the result.
func MakeService(name string, tweaks ...Tweak) *api.Service {
	// NOTE: Any field that would be populated by defaulting needs to be
	// present and valid here.
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"k": "v"},
			SessionAffinity: api.ServiceAffinityNone,
		},
	}
	// Default to ClusterIP
	SetTypeClusterIP(svc)
	// Default to 1 port
	SetPorts(MakeServicePort("", 93, intstr.FromInt(76), api.ProtocolTCP))(svc)

	for _, tweak := range tweaks {
		tweak(svc)
	}

	return svc
}

// SetTypeClusterIP sets the service type to ClusterIP and clears other fields.
func SetTypeClusterIP(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeClusterIP
	for i := range svc.Spec.Ports {
		svc.Spec.Ports[i].NodePort = 0
	}
	svc.Spec.ExternalName = ""
	svc.Spec.ExternalTrafficPolicy = ""
	svc.Spec.AllocateLoadBalancerNodePorts = nil
	internalTrafficPolicy := api.ServiceInternalTrafficPolicyCluster
	svc.Spec.InternalTrafficPolicy = &internalTrafficPolicy
}

// SetTypeNodePort sets the service type to NodePort and clears other fields.
func SetTypeNodePort(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeNodePort
	svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeCluster
	svc.Spec.ExternalName = ""
	svc.Spec.AllocateLoadBalancerNodePorts = nil
	internalTrafficPolicy := api.ServiceInternalTrafficPolicyCluster
	svc.Spec.InternalTrafficPolicy = &internalTrafficPolicy
}

// SetTypeLoadBalancer sets the service type to LoadBalancer and clears other fields.
func SetTypeLoadBalancer(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeLoadBalancer
	svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeCluster
	svc.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(true)
	svc.Spec.ExternalName = ""
	internalTrafficPolicy := api.ServiceInternalTrafficPolicyCluster
	svc.Spec.InternalTrafficPolicy = &internalTrafficPolicy
}

// SetTypeExternalName sets the service type to ExternalName and clears other fields.
func SetTypeExternalName(svc *api.Service) {
	svc.Spec.Type = api.ServiceTypeExternalName
	svc.Spec.ExternalName = "example.com"
	svc.Spec.ExternalTrafficPolicy = ""
	svc.Spec.ClusterIP = ""
	svc.Spec.ClusterIPs = nil
	svc.Spec.AllocateLoadBalancerNodePorts = nil
	svc.Spec.InternalTrafficPolicy = nil
}

// SetPorts sets the service ports list.
func SetPorts(ports ...api.ServicePort) Tweak {
	return func(svc *api.Service) {
		svc.Spec.Ports = ports
	}
}

// MakeServicePort helps construct ServicePort objects which pass API
// validation.
func MakeServicePort(name string, port int, tgtPort intstr.IntOrString, proto api.Protocol) api.ServicePort {
	return api.ServicePort{
		Name:       name,
		Port:       int32(port),
		TargetPort: tgtPort,
		Protocol:   proto,
	}
}

// SetHeadless sets the service as headless and clears other fields.
func SetHeadless(svc *api.Service) {
	SetTypeClusterIP(svc)
	SetClusterIPs(api.ClusterIPNone)(svc)
}

// SetSelector sets the service selector.
func SetSelector(sel map[string]string) Tweak {
	return func(svc *api.Service) {
		svc.Spec.Selector = map[string]string{}
		for k, v := range sel {
			svc.Spec.Selector[k] = v
		}
	}
}

// SetClusterIP sets the service ClusterIP fields.
func SetClusterIP(ip string) Tweak {
	return func(svc *api.Service) {
		svc.Spec.ClusterIP = ip
	}
}

// SetClusterIPs sets the service ClusterIP and ClusterIPs fields.
func SetClusterIPs(ips ...string) Tweak {
	return func(svc *api.Service) {
		svc.Spec.ClusterIP = ips[0]
		svc.Spec.ClusterIPs = ips
	}
}

// SetIPFamilies sets the service IPFamilies field.
func SetIPFamilies(families ...api.IPFamily) Tweak {
	return func(svc *api.Service) {
		svc.Spec.IPFamilies = families
	}
}

// SetIPFamilyPolicy sets the service IPFamilyPolicy field.
func SetIPFamilyPolicy(policy api.IPFamilyPolicy) Tweak {
	return func(svc *api.Service) {
		svc.Spec.IPFamilyPolicy = &policy
	}
}

// SetNodePorts sets the values for each node port, in order.  If less values
// are specified than there are ports, the rest are untouched.
func SetNodePorts(values ...int) Tweak {
	return func(svc *api.Service) {
		for i := range svc.Spec.Ports {
			if i >= len(values) {
				break
			}
			svc.Spec.Ports[i].NodePort = int32(values[i])
		}
	}
}

// SetInternalTrafficPolicy sets the internalTrafficPolicy field for a Service.
func SetInternalTrafficPolicy(policy api.ServiceInternalTrafficPolicyType) Tweak {
	return func(svc *api.Service) {
		svc.Spec.InternalTrafficPolicy = &policy
	}
}

// SetExternalTrafficPolicy sets the externalTrafficPolicy field for a Service.
func SetExternalTrafficPolicy(policy api.ServiceExternalTrafficPolicyType) Tweak {
	return func(svc *api.Service) {
		svc.Spec.ExternalTrafficPolicy = policy
	}
}

// SetAllocateLoadBalancerNodePorts sets the allocate LB node port field.
func SetAllocateLoadBalancerNodePorts(val bool) Tweak {
	return func(svc *api.Service) {
		svc.Spec.AllocateLoadBalancerNodePorts = utilpointer.BoolPtr(val)
	}
}

// SetUniqueNodePorts sets all nodeports to unique values.
func SetUniqueNodePorts(svc *api.Service) {
	for i := range svc.Spec.Ports {
		svc.Spec.Ports[i].NodePort = int32(30000 + i)
	}
}

// SetHealthCheckNodePort sets the healthCheckNodePort field for a Service.
func SetHealthCheckNodePort(value int32) Tweak {
	return func(svc *api.Service) {
		svc.Spec.HealthCheckNodePort = value
	}
}

// SetSessionAffinity sets the SessionAffinity field.
func SetSessionAffinity(affinity api.ServiceAffinity) Tweak {
	return func(svc *api.Service) {
		svc.Spec.SessionAffinity = affinity
		switch affinity {
		case api.ServiceAffinityNone:
			svc.Spec.SessionAffinityConfig = nil
		case api.ServiceAffinityClientIP:
			timeout := int32(10)
			svc.Spec.SessionAffinityConfig = &api.SessionAffinityConfig{
				ClientIP: &api.ClientIPConfig{
					TimeoutSeconds: &timeout,
				},
			}
		}
	}
}

// SetExternalName sets the ExternalName field.
func SetExternalName(val string) Tweak {
	return func(svc *api.Service) {
		svc.Spec.ExternalName = val
	}
}
