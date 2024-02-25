//go:build !providerless
// +build !providerless

/*
Copyright 2016 The Kubernetes Authors.

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

package gce

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/test/e2e/framework"
	gcecloud "k8s.io/legacy-cloud-providers/gce"
)

// MakeFirewallNameForLBService return the expected firewall name for a LB service.
// This should match the formatting of makeFirewallName() in pkg/cloudprovider/providers/gce/gce_loadbalancer.go
func MakeFirewallNameForLBService(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

// ConstructFirewallForLBService returns the expected GCE firewall rule for a loadbalancer type service
func ConstructFirewallForLBService(svc *v1.Service, nodeTag string) *compute.Firewall {
	if svc.Spec.Type != v1.ServiceTypeLoadBalancer {
		framework.Failf("can not construct firewall rule for non-loadbalancer type service")
	}
	fw := compute.Firewall{}
	fw.Name = MakeFirewallNameForLBService(cloudprovider.DefaultLoadBalancerName(svc))
	fw.TargetTags = []string{nodeTag}
	if svc.Spec.LoadBalancerSourceRanges == nil {
		fw.SourceRanges = []string{"0.0.0.0/0"}
	} else {
		fw.SourceRanges = svc.Spec.LoadBalancerSourceRanges
	}
	for _, sp := range svc.Spec.Ports {
		fw.Allowed = append(fw.Allowed, &compute.FirewallAllowed{
			IPProtocol: strings.ToLower(string(sp.Protocol)),
			Ports:      []string{strconv.Itoa(int(sp.Port))},
		})
	}
	return &fw
}

// MakeHealthCheckFirewallNameForLBService returns the firewall name used by the GCE load
// balancers for performing health checks.
func MakeHealthCheckFirewallNameForLBService(clusterID, name string, isNodesHealthCheck bool) string {
	return gcecloud.MakeHealthCheckFirewallName(clusterID, name, isNodesHealthCheck)
}

// ConstructHealthCheckFirewallForLBService returns the expected GCE firewall rule for a loadbalancer type service
func ConstructHealthCheckFirewallForLBService(clusterID string, svc *v1.Service, nodeTag string, isNodesHealthCheck bool) *compute.Firewall {
	if svc.Spec.Type != v1.ServiceTypeLoadBalancer {
		framework.Failf("can not construct firewall rule for non-loadbalancer type service")
	}
	fw := compute.Firewall{}
	fw.Name = MakeHealthCheckFirewallNameForLBService(clusterID, cloudprovider.DefaultLoadBalancerName(svc), isNodesHealthCheck)
	fw.TargetTags = []string{nodeTag}
	fw.SourceRanges = gcecloud.L4LoadBalancerSrcRanges()
	healthCheckPort := gcecloud.GetNodesHealthCheckPort()
	if !isNodesHealthCheck {
		healthCheckPort = svc.Spec.HealthCheckNodePort
	}
	fw.Allowed = []*compute.FirewallAllowed{
		{
			IPProtocol: "tcp",
			Ports:      []string{fmt.Sprintf("%d", healthCheckPort)},
		},
	}
	return &fw
}

// PackProtocolsPortsFromFirewall packs protocols and ports in an unified way for verification.
func PackProtocolsPortsFromFirewall(alloweds []*compute.FirewallAllowed) []string {
	protocolPorts := []string{}
	for _, allowed := range alloweds {
		for _, port := range allowed.Ports {
			protocolPorts = append(protocolPorts, strings.ToLower(allowed.IPProtocol+"/"+port))
		}
	}
	return protocolPorts
}

type portRange struct {
	protocol string
	min, max int
}

func toPortRange(s string) (pr portRange, err error) {
	protoPorts := strings.Split(s, "/")
	// Set protocol
	pr.protocol = strings.ToUpper(protoPorts[0])

	if len(protoPorts) != 2 {
		return pr, fmt.Errorf("expected a single '/' in %q", s)
	}

	ports := strings.Split(protoPorts[1], "-")
	switch len(ports) {
	case 1:
		v, err := strconv.Atoi(ports[0])
		if err != nil {
			return pr, err
		}
		pr.min, pr.max = v, v
	case 2:
		start, err := strconv.Atoi(ports[0])
		if err != nil {
			return pr, err
		}
		end, err := strconv.Atoi(ports[1])
		if err != nil {
			return pr, err
		}
		pr.min, pr.max = start, end
	default:
		return pr, fmt.Errorf("unexpected range value %q", protoPorts[1])
	}

	return pr, nil
}

// isPortsSubset asserts that the "requiredPorts" are covered by the "coverage" ports.
// requiredPorts - must be single-port, examples: 'tcp/50', 'udp/80'.
// coverage - single or port-range values, example: 'tcp/50', 'udp/80-1000'.
// Returns true if every requiredPort exists in the list of coverage rules.
func isPortsSubset(requiredPorts, coverage []string) error {
	for _, reqPort := range requiredPorts {
		rRange, err := toPortRange(reqPort)
		if err != nil {
			return err
		}
		if rRange.min != rRange.max {
			return fmt.Errorf("requiring a range is not supported: %q", reqPort)
		}

		var covered bool
		for _, c := range coverage {
			cRange, err := toPortRange(c)
			if err != nil {
				return err
			}

			if rRange.protocol != cRange.protocol {
				continue
			}

			if rRange.min >= cRange.min && rRange.min <= cRange.max {
				covered = true
				break
			}
		}

		if !covered {
			return fmt.Errorf("%q is not covered by %v", reqPort, coverage)
		}
	}
	return nil
}

// SameStringArray verifies whether two string arrays have the same strings, return error if not.
// Order does not matter.
// When `include` is set to true, verifies whether result includes all elements from expected.
func SameStringArray(result, expected []string, include bool) error {
	res := sets.NewString(result...)
	exp := sets.NewString(expected...)
	if !include {
		diff := res.Difference(exp)
		if len(diff) != 0 {
			return fmt.Errorf("found differences: %v", diff)
		}
	} else {
		if !res.IsSuperset(exp) {
			return fmt.Errorf("some elements are missing: expected %v, got %v", expected, result)
		}
	}
	return nil
}

// VerifyFirewallRule verifies whether the result firewall is consistent with the expected firewall.
// When `portsSubset` is false, match given ports exactly. Otherwise, only check ports are included.
func VerifyFirewallRule(res, exp *compute.Firewall, network string, portsSubset bool) error {
	if res == nil || exp == nil {
		return fmt.Errorf("res and exp must not be nil")
	}
	if res.Name != exp.Name {
		return fmt.Errorf("incorrect name: %v, expected %v", res.Name, exp.Name)
	}

	actualPorts := PackProtocolsPortsFromFirewall(res.Allowed)
	expPorts := PackProtocolsPortsFromFirewall(exp.Allowed)
	if portsSubset {
		if err := isPortsSubset(expPorts, actualPorts); err != nil {
			return fmt.Errorf("incorrect allowed protocol ports: %w", err)
		}
	} else {
		if err := SameStringArray(actualPorts, expPorts, false); err != nil {
			return fmt.Errorf("incorrect allowed protocols ports: %w", err)
		}
	}

	if err := SameStringArray(res.SourceRanges, exp.SourceRanges, false); err != nil {
		return fmt.Errorf("incorrect source ranges %v, expected %v: %w", res.SourceRanges, exp.SourceRanges, err)
	}
	if err := SameStringArray(res.SourceTags, exp.SourceTags, false); err != nil {
		return fmt.Errorf("incorrect source tags %v, expected %v: %w", res.SourceTags, exp.SourceTags, err)
	}
	if err := SameStringArray(res.TargetTags, exp.TargetTags, false); err != nil {
		return fmt.Errorf("incorrect target tags %v, expected %v: %w", res.TargetTags, exp.TargetTags, err)
	}
	return nil
}

// WaitForFirewallRule waits for the specified firewall existence
func WaitForFirewallRule(ctx context.Context, gceCloud *gcecloud.Cloud, fwName string, exist bool, timeout time.Duration) (*compute.Firewall, error) {
	framework.Logf("Waiting up to %v for firewall %v exist=%v", timeout, fwName, exist)
	var fw *compute.Firewall
	var err error

	condition := func(ctx context.Context) (bool, error) {
		fw, err = gceCloud.GetFirewall(fwName)
		if err != nil && exist ||
			err == nil && !exist ||
			err != nil && !exist && !IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollUntilContextTimeout(ctx, 5*time.Second, timeout, true, condition); err != nil {
		return nil, fmt.Errorf("error waiting for firewall %v exist=%v", fwName, exist)
	}
	return fw, nil
}
