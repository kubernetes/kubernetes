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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateService(t *testing.T) {
	testCases := []struct {
		name     string
		tweakSvc func(svc *core.Service) // given a basic valid service, each test case can customize it
		numErrs  int
	}{
		{
			name: "missing namespace",
			tweakSvc: func(s *core.Service) {
				s.Namespace = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid namespace",
			tweakSvc: func(s *core.Service) {
				s.Namespace = "-123"
			},
			numErrs: 1,
		},
		{
			name: "missing name",
			tweakSvc: func(s *core.Service) {
				s.Name = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid name",
			tweakSvc: func(s *core.Service) {
				s.Name = "-123"
			},
			numErrs: 1,
		},
		{
			name: "too long name",
			tweakSvc: func(s *core.Service) {
				s.Name = strings.Repeat("a", 64)
			},
			numErrs: 1,
		},
		{
			name: "invalid generateName",
			tweakSvc: func(s *core.Service) {
				s.GenerateName = "-123"
			},
			numErrs: 1,
		},
		{
			name: "too long generateName",
			tweakSvc: func(s *core.Service) {
				s.GenerateName = strings.Repeat("a", 64)
			},
			numErrs: 1,
		},
		{
			name: "invalid label",
			tweakSvc: func(s *core.Service) {
				s.Labels["NoUppercaseOrSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "invalid annotation",
			tweakSvc: func(s *core.Service) {
				s.Annotations["NoSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "nil selector",
			tweakSvc: func(s *core.Service) {
				s.Spec.Selector = nil
			},
			numErrs: 0,
		},
		{
			name: "invalid selector",
			tweakSvc: func(s *core.Service) {
				s.Spec.Selector["NoSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "missing session affinity",
			tweakSvc: func(s *core.Service) {
				s.Spec.SessionAffinity = ""
			},
			numErrs: 1,
		},
		{
			name: "missing type",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = ""
			},
			numErrs: 1,
		},
		{
			name: "missing ports",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports = nil
			},
			numErrs: 1,
		},
		{
			name: "missing ports but headless",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports = nil
				s.Spec.ClusterIP = core.ClusterIPNone
			},
			numErrs: 0,
		},
		{
			name: "empty port[0] name",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Name = ""
			},
			numErrs: 0,
		},
		{
			name: "empty port[1] name",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "", Protocol: "TCP", Port: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "empty multi-port port[0] name",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Name = ""
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "p", Protocol: "TCP", Port: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid port name",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Name = "INVALID"
			},
			numErrs: 1,
		},
		{
			name: "missing protocol",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Protocol = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid protocol",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Protocol = "INVALID"
			},
			numErrs: 1,
		},
		{
			name: "invalid cluster ip",
			tweakSvc: func(s *core.Service) {
				s.Spec.ClusterIP = "invalid"
			},
			numErrs: 1,
		},
		{
			name: "missing port",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Port = 0
			},
			numErrs: 1,
		},
		{
			name: "invalid port",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Port = 65536
			},
			numErrs: 1,
		},
		{
			name: "invalid TargetPort int",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].TargetPort = intstr.FromInt(65536)
			},
			numErrs: 1,
		},
		{
			name: "valid port headless",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromInt(11722)
				s.Spec.ClusterIP = core.ClusterIPNone
			},
			numErrs: 0,
		},
		{
			name: "invalid port headless 1",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromInt(11721)
				s.Spec.ClusterIP = core.ClusterIPNone
			},
			// in the v1 API, targetPorts on headless services were tolerated.
			// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
			// numErrs: 1,
			numErrs: 0,
		},
		{
			name: "invalid port headless 2",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromString("target")
				s.Spec.ClusterIP = core.ClusterIPNone
			},
			// in the v1 API, targetPorts on headless services were tolerated.
			// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
			// numErrs: 1,
			numErrs: 0,
		},
		{
			name: "invalid publicIPs localhost",
			tweakSvc: func(s *core.Service) {
				s.Spec.ExternalIPs = []string{"127.0.0.1"}
			},
			numErrs: 1,
		},
		{
			name: "invalid publicIPs unspecified",
			tweakSvc: func(s *core.Service) {
				s.Spec.ExternalIPs = []string{"0.0.0.0"}
			},
			numErrs: 1,
		},
		{
			name: "invalid publicIPs loopback",
			tweakSvc: func(s *core.Service) {
				s.Spec.ExternalIPs = []string{"127.0.0.1"}
			},
			numErrs: 1,
		},
		{
			name: "invalid publicIPs host",
			tweakSvc: func(s *core.Service) {
				s.Spec.ExternalIPs = []string{"myhost.mydomain"}
			},
			numErrs: 1,
		},
		{
			name: "dup port name",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Name = "p"
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "p", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid load balancer protocol UDP 1",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports[0].Protocol = "UDP"
			},
			numErrs: 0,
		},
		{
			name: "valid load balancer protocol UDP 2",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports[0] = core.ServicePort{Name: "q", Port: 12345, Protocol: "UDP", TargetPort: intstr.FromInt(12345)}
			},
			numErrs: 0,
		},
		{
			name: "invalid load balancer with mix protocol",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "UDP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid 1",
			tweakSvc: func(s *core.Service) {
				// do nothing
			},
			numErrs: 0,
		},
		{
			name: "valid 2",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].Protocol = "UDP"
				s.Spec.Ports[0].TargetPort = intstr.FromInt(12345)
			},
			numErrs: 0,
		},
		{
			name: "valid 3",
			tweakSvc: func(s *core.Service) {
				s.Spec.Ports[0].TargetPort = intstr.FromString("http")
			},
			numErrs: 0,
		},
		{
			name: "valid cluster ip - none ",
			tweakSvc: func(s *core.Service) {
				s.Spec.ClusterIP = "None"
			},
			numErrs: 0,
		},
		{
			name: "valid cluster ip - empty",
			tweakSvc: func(s *core.Service) {
				s.Spec.ClusterIP = ""
				s.Spec.Ports[0].TargetPort = intstr.FromString("http")
			},
			numErrs: 0,
		},
		{
			name: "valid type - cluster",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
			},
			numErrs: 0,
		},
		{
			name: "valid type - loadbalancer",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer 2 ports",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid external load balancer 2 ports",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "duplicate nodeports",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "r", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "duplicate nodeports (different protocols)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "r", Port: 2, Protocol: "UDP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 0,
		},
		{
			name: "invalid duplicate ports (with same protocol)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(8080)})
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "r", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(80)})
			},
			numErrs: 1,
		},
		{
			name: "valid duplicate ports (with different protocols)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(8080)})
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "r", Port: 12345, Protocol: "UDP", TargetPort: intstr.FromInt(80)})
			},
			numErrs: 0,
		},
		{
			name: "valid type - cluster",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
			},
			numErrs: 0,
		},
		{
			name: "valid type - nodeport",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
			},
			numErrs: 0,
		},
		{
			name: "valid type - loadbalancer",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer 2 ports",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer with NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service with NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service without NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid cluster service without NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "invalid cluster service with NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid public service with duplicate NodePort",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "p1", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "p2", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "valid type=LoadBalancer",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			// For now we open firewalls, and its insecure if we open 10250, remove this
			// when we have better protections in place.
			name: "invalid port type=LoadBalancer",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "kubelet", Port: 10250, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid LoadBalancer source range annotation",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Annotations[core.AnnotationLoadBalancerSourceRangesKey] = "1.2.3.4/8,  5.6.7.8/16"
			},
			numErrs: 0,
		},
		{
			name: "empty LoadBalancer source range annotation",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Annotations[core.AnnotationLoadBalancerSourceRangesKey] = ""
			},
			numErrs: 0,
		},
		{
			name: "invalid LoadBalancer source range annotation (hostname)",
			tweakSvc: func(s *core.Service) {
				s.Annotations[core.AnnotationLoadBalancerSourceRangesKey] = "foo.bar"
			},
			numErrs: 2,
		},
		{
			name: "invalid LoadBalancer source range annotation (invalid CIDR)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Annotations[core.AnnotationLoadBalancerSourceRangesKey] = "1.2.3.4/33"
			},
			numErrs: 1,
		},
		{
			name: "invalid source range for non LoadBalancer type service",
			tweakSvc: func(s *core.Service) {
				s.Spec.LoadBalancerSourceRanges = []string{"1.2.3.4/8", "5.6.7.8/16"}
			},
			numErrs: 1,
		},
		{
			name: "valid LoadBalancer source range",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.LoadBalancerSourceRanges = []string{"1.2.3.4/8", "5.6.7.8/16"}
			},
			numErrs: 0,
		},
		{
			name: "empty LoadBalancer source range",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.LoadBalancerSourceRanges = []string{"   "}
			},
			numErrs: 1,
		},
		{
			name: "invalid LoadBalancer source range",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.LoadBalancerSourceRanges = []string{"foo.bar"}
			},
			numErrs: 1,
		},
		{
			name: "valid ExternalName",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeExternalName
				s.Spec.ClusterIP = ""
				s.Spec.ExternalName = "foo.bar.example.com"
			},
			numErrs: 0,
		},
		{
			name: "invalid ExternalName clusterIP (valid IP)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeExternalName
				s.Spec.ClusterIP = "1.2.3.4"
				s.Spec.ExternalName = "foo.bar.example.com"
			},
			numErrs: 1,
		},
		{
			name: "invalid ExternalName clusterIP (None)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeExternalName
				s.Spec.ClusterIP = "None"
				s.Spec.ExternalName = "foo.bar.example.com"
			},
			numErrs: 1,
		},
		{
			name: "invalid ExternalName (not a DNS name)",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeExternalName
				s.Spec.ClusterIP = ""
				s.Spec.ExternalName = "-123"
			},
			numErrs: 1,
		},
		{
			name: "LoadBalancer type cannot have None ClusterIP",
			tweakSvc: func(s *core.Service) {
				s.Spec.ClusterIP = "None"
				s.Spec.Type = core.ServiceTypeLoadBalancer
			},
			numErrs: 1,
		},
		{
			name: "invalid node port with clusterIP None",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, core.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.ClusterIP = "None"
			},
			numErrs: 1,
		},
		// ESIPP section begins.
		{
			name: "invalid externalTraffic field",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.ExternalTrafficPolicy = "invalid"
			},
			numErrs: 1,
		},
		{
			name: "nagative healthCheckNodePort field",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
				s.Spec.HealthCheckNodePort = -1
			},
			numErrs: 1,
		},
		{
			name: "nagative healthCheckNodePort field",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
				s.Spec.HealthCheckNodePort = 31100
			},
			numErrs: 0,
		},
		// ESIPP section ends.
		{
			name: "invalid timeoutSeconds field",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.SessionAffinity = core.ServiceAffinityClientIP
				s.Spec.SessionAffinityConfig = &core.SessionAffinityConfig{
					ClientIP: &core.ClientIPConfig{
						TimeoutSeconds: newInt32(-1),
					},
				}
			},
			numErrs: 1,
		},
		{
			name: "sessionAffinityConfig can't be set when session affinity is None",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.SessionAffinity = core.ServiceAffinityNone
				s.Spec.SessionAffinityConfig = &core.SessionAffinityConfig{
					ClientIP: &core.ClientIPConfig{
						TimeoutSeconds: newInt32(90),
					},
				}
			},
			numErrs: 1,
		},
	}

	for _, tc := range testCases {
		svc := makeValidService()
		tc.tweakSvc(&svc)
		errs := ValidateService(&svc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, errs.ToAggregate())
		}
	}
}

func TestValidateServiceExternalTrafficFieldsCombination(t *testing.T) {
	testCases := []struct {
		name     string
		tweakSvc func(svc *core.Service) // Given a basic valid service, each test case can customize it.
		numErrs  int
	}{
		{
			name: "valid loadBalancer service with externalTrafficPolicy and healthCheckNodePort set",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
				s.Spec.HealthCheckNodePort = 34567
			},
			numErrs: 0,
		},
		{
			name: "valid nodePort service with externalTrafficPolicy set",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
			},
			numErrs: 0,
		},
		{
			name: "valid clusterIP service with none of externalTrafficPolicy and healthCheckNodePort set",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
			},
			numErrs: 0,
		},
		{
			name: "cannot set healthCheckNodePort field on loadBalancer service with externalTrafficPolicy!=Local",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeLoadBalancer
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeCluster
				s.Spec.HealthCheckNodePort = 34567
			},
			numErrs: 1,
		},
		{
			name: "cannot set healthCheckNodePort field on nodePort service",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeNodePort
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
				s.Spec.HealthCheckNodePort = 34567
			},
			numErrs: 1,
		},
		{
			name: "cannot set externalTrafficPolicy or healthCheckNodePort fields on clusterIP service",
			tweakSvc: func(s *core.Service) {
				s.Spec.Type = core.ServiceTypeClusterIP
				s.Spec.ExternalTrafficPolicy = core.ServiceExternalTrafficPolicyTypeLocal
				s.Spec.HealthCheckNodePort = 34567
			},
			numErrs: 2,
		},
	}

	for _, tc := range testCases {
		svc := makeValidService()
		tc.tweakSvc(&svc)
		errs := ValidateServiceExternalTrafficFieldsCombination(&svc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, errs.ToAggregate())
		}
	}
}

func TestValidateServiceUpdate(t *testing.T) {
	testCases := []struct {
		name     string
		tweakSvc func(oldSvc, newSvc *core.Service) // given basic valid services, each test case can customize them
		numErrs  int
	}{
		{
			name: "no change",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				// do nothing
			},
			numErrs: 0,
		},
		{
			name: "change name",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Name += "2"
			},
			numErrs: 1,
		},
		{
			name: "change namespace",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Namespace += "2"
			},
			numErrs: 1,
		},
		{
			name: "change label valid",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Labels["key"] = "other-value"
			},
			numErrs: 0,
		},
		{
			name: "add label",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Labels["key2"] = "value2"
			},
			numErrs: 0,
		},
		{
			name: "change cluster IP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "8.6.7.5"
			},
			numErrs: 1,
		},
		{
			name: "remove cluster IP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = ""
			},
			numErrs: 1,
		},
		{
			name: "change affinity",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.SessionAffinity = "ClientIP"
				newSvc.Spec.SessionAffinityConfig = &core.SessionAffinityConfig{
					ClientIP: &core.ClientIPConfig{
						TimeoutSeconds: newInt32(90),
					},
				}
			},
			numErrs: 0,
		},
		{
			name: "remove affinity",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.SessionAffinity = ""
			},
			numErrs: 1,
		},
		{
			name: "change type",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "remove type",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.Type = ""
			},
			numErrs: 1,
		},
		{
			name: "change type -> nodeport",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.Type = core.ServiceTypeNodePort
			},
			numErrs: 0,
		},
		{
			name: "add loadBalancerSourceRanges",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.LoadBalancerSourceRanges = []string{"10.0.0.0/8"}
			},
			numErrs: 0,
		},
		{
			name: "update loadBalancerSourceRanges",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				oldSvc.Spec.LoadBalancerSourceRanges = []string{"10.0.0.0/8"}
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.LoadBalancerSourceRanges = []string{"10.100.0.0/16"}
			},
			numErrs: 0,
		},
		{
			name: "LoadBalancer type cannot have None ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				newSvc.Spec.ClusterIP = "None"
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer
			},
			numErrs: 1,
		},
		{
			name: "`None` ClusterIP cannot be changed",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.ClusterIP = "None"
				newSvc.Spec.ClusterIP = "1.2.3.4"
			},
			numErrs: 1,
		},
		{
			name: "`None` ClusterIP cannot be removed",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.ClusterIP = "None"
				newSvc.Spec.ClusterIP = ""
			},
			numErrs: 1,
		},
		{
			name: "Service with ClusterIP type cannot change its set ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with ClusterIP type can change its empty ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with ClusterIP type cannot change its set ClusterIP when changing type to NodePort",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with ClusterIP type can change its empty ClusterIP when changing type to NodePort",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with ClusterIP type cannot change its ClusterIP when changing type to LoadBalancer",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with ClusterIP type can change its empty ClusterIP when changing type to LoadBalancer",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeClusterIP
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with NodePort type cannot change its set ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with NodePort type can change its empty ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with NodePort type cannot change its set ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with NodePort type can change its empty ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with NodePort type cannot change its set ClusterIP when changing type to LoadBalancer",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with NodePort type can change its empty ClusterIP when changing type to LoadBalancer",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with LoadBalancer type cannot change its set ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with LoadBalancer type can change its empty ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeLoadBalancer

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with LoadBalancer type cannot change its set ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with LoadBalancer type can change its empty ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with LoadBalancer type cannot change its set ClusterIP when changing type to NodePort",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 1,
		},
		{
			name: "Service with LoadBalancer type can change its empty ClusterIP when changing type to NodePort",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeLoadBalancer
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with ExternalName type can change its empty ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeExternalName
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "Service with ExternalName type can change its set ClusterIP when changing type to ClusterIP",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeExternalName
				newSvc.Spec.Type = core.ServiceTypeClusterIP

				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "1.2.3.5"
			},
			numErrs: 0,
		},
		{
			name: "invalid node port with clusterIP None",
			tweakSvc: func(oldSvc, newSvc *core.Service) {
				oldSvc.Spec.Type = core.ServiceTypeNodePort
				newSvc.Spec.Type = core.ServiceTypeNodePort

				oldSvc.Spec.Ports = append(oldSvc.Spec.Ports, core.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				newSvc.Spec.Ports = append(newSvc.Spec.Ports, core.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})

				oldSvc.Spec.ClusterIP = ""
				newSvc.Spec.ClusterIP = "None"
			},
			numErrs: 1,
		},
	}

	for _, tc := range testCases {
		oldSvc := makeValidService()
		newSvc := makeValidService()
		tc.tweakSvc(&oldSvc, &newSvc)
		errs := ValidateServiceUpdate(&newSvc, &oldSvc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, errs.ToAggregate())
		}
	}
}

func makeValidService() core.Service {
	return core.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid",
			Namespace:       "valid",
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			ResourceVersion: "1",
		},
		Spec: core.ServiceSpec{
			Selector:        map[string]string{"key": "val"},
			SessionAffinity: "None",
			Type:            core.ServiceTypeClusterIP,
			Ports:           []core.ServicePort{{Name: "p", Protocol: "TCP", Port: 8675, TargetPort: intstr.FromInt(8675)}},
		},
	}
}
