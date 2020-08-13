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
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
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

// GetInstancePrefix returns the INSTANCE_PREFIX env we set for e2e cluster.
// From cluster/gce/config-test.sh, master name is set up using below format:
// MASTER_NAME="${INSTANCE_PREFIX}-master"
func GetInstancePrefix(masterName string) (string, error) {
	if !strings.HasSuffix(masterName, "-master") {
		return "", fmt.Errorf("unexpected master name format: %v", masterName)
	}
	return masterName[:len(masterName)-7], nil
}

// GetClusterName returns the CLUSTER_NAME env we set for e2e cluster.
// From cluster/gce/config-test.sh, cluster name is set up using below format:
// CLUSTER_NAME="${CLUSTER_NAME:-${INSTANCE_PREFIX}}"
func GetClusterName(instancePrefix string) string {
	return instancePrefix
}

// GetE2eFirewalls returns all firewall rules we create for an e2e cluster.
// From cluster/gce/util.sh, all firewall rules should be consistent with the ones created by startup scripts.
func GetE2eFirewalls(masterName, masterTag, nodeTag, network, clusterIPRange string) []*compute.Firewall {
	instancePrefix, err := GetInstancePrefix(masterName)
	framework.ExpectNoError(err)
	clusterName := GetClusterName(instancePrefix)

	fws := []*compute.Firewall{}
	fws = append(fws, &compute.Firewall{
		Name:         clusterName + "-default-internal-master",
		SourceRanges: []string{"10.0.0.0/8"},
		TargetTags:   []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"1-2379"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"2382-65535"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "icmp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         clusterName + "-default-internal-node",
		SourceRanges: []string{"10.0.0.0/8"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"1-65535"},
			},
			{
				IPProtocol: "icmp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         network + "-default-ssh",
		SourceRanges: []string{"0.0.0.0/0"},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"22"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:       masterName + "-etcd",
		SourceTags: []string{masterTag},
		TargetTags: []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"2380"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"2381"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         masterName + "-https",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{masterTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"443"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-all",
		SourceRanges: []string{clusterIPRange},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
			},
			{
				IPProtocol: "udp",
			},
			{
				IPProtocol: "icmp",
			},
			{
				IPProtocol: "esp",
			},
			{
				IPProtocol: "ah",
			},
			{
				IPProtocol: "sctp",
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-http-alt",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"80"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"8080"},
			},
		},
	})
	fws = append(fws, &compute.Firewall{
		Name:         nodeTag + "-nodeports",
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{nodeTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"30000-32767"},
			},
			{
				IPProtocol: "udp",
				Ports:      []string{"30000-32767"},
			},
		},
	})
	return fws
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
	// Sample Network value: https://www.googleapis.com/compute/v1/projects/{project-id}/global/networks/e2e
	if !strings.HasSuffix(res.Network, "/"+network) {
		return fmt.Errorf("incorrect network: %v, expected ends with: %v", res.Network, "/"+network)
	}

	actualPorts := PackProtocolsPortsFromFirewall(res.Allowed)
	expPorts := PackProtocolsPortsFromFirewall(exp.Allowed)
	if portsSubset {
		if err := isPortsSubset(expPorts, actualPorts); err != nil {
			return fmt.Errorf("incorrect allowed protocol ports: %v", err)
		}
	} else {
		if err := SameStringArray(actualPorts, expPorts, false); err != nil {
			return fmt.Errorf("incorrect allowed protocols ports: %v", err)
		}
	}

	if err := SameStringArray(res.SourceRanges, exp.SourceRanges, false); err != nil {
		return fmt.Errorf("incorrect source ranges %v, expected %v: %v", res.SourceRanges, exp.SourceRanges, err)
	}
	if err := SameStringArray(res.SourceTags, exp.SourceTags, false); err != nil {
		return fmt.Errorf("incorrect source tags %v, expected %v: %v", res.SourceTags, exp.SourceTags, err)
	}
	if err := SameStringArray(res.TargetTags, exp.TargetTags, false); err != nil {
		return fmt.Errorf("incorrect target tags %v, expected %v: %v", res.TargetTags, exp.TargetTags, err)
	}
	return nil
}

// WaitForFirewallRule waits for the specified firewall existence
func WaitForFirewallRule(gceCloud *gcecloud.Cloud, fwName string, exist bool, timeout time.Duration) (*compute.Firewall, error) {
	framework.Logf("Waiting up to %v for firewall %v exist=%v", timeout, fwName, exist)
	var fw *compute.Firewall
	var err error

	condition := func() (bool, error) {
		fw, err = gceCloud.GetFirewall(fwName)
		if err != nil && exist ||
			err == nil && !exist ||
			err != nil && !exist && !IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
			return false, nil
		}
		return true, nil
	}

	if err := wait.PollImmediate(5*time.Second, timeout, condition); err != nil {
		return nil, fmt.Errorf("error waiting for firewall %v exist=%v", fwName, exist)
	}
	return fw, nil
}
