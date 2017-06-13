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

package gce

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
)

// Internal Load Balancer

// Instance groups remain legacy named to stay consistent with ingress
func makeInstanceGroupName(clusterID string) string {
	return fmt.Sprintf("k8s-ig--%s", clusterID)
}

func makeBackendServiceName(loadBalancerName, clusterID string, shared bool, scheme lbScheme, protocol v1.Protocol, svcAffinity v1.ServiceAffinity) string {
	if shared {
		hash := sha1.New()

		// For every non-nil option, hash its value. Currently, only service affinity is relevant.
		hash.Write([]byte(string(svcAffinity)))

		hashed := hex.EncodeToString(hash.Sum(nil))
		hashed = hashed[:16]

		// k8s-          4
		// {clusterid}-  17
		// {scheme}-     9   (internal/external)
		// {protocol}-   4   (tcp/udp)
		// nmv1-         5   (naming convention version)
		// {suffix}      16  (hash of settings)
		// -----------------
		//               55  characters used
		return fmt.Sprintf("k8s-%s-%s-%s-nmv1-%s", clusterID, strings.ToLower(string(scheme)), strings.ToLower(string(protocol)), hashed)
	}
	return loadBalancerName
}

func makeHealthCheckName(loadBalancerName, clusterID string, shared bool) string {
	if shared {
		return fmt.Sprintf("k8s-%s-node", clusterID)
	}

	return loadBalancerName
}

func makeHealthCheckFirewallNameFromHC(healthCheckName string) string {
	return healthCheckName + "-hc"
}

func makeHealthCheckFirewallName(loadBalancerName, clusterID string, shared bool) string {
	if shared {
		return fmt.Sprintf("k8s-%s-node-hc", clusterID)
	}
	return loadBalancerName + "-hc"
}

func makeBackendServiceDescription(nm types.NamespacedName, shared bool) string {
	if shared {
		return ""
	}
	return fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, nm.String())
}

// External Load Balancer

// makeNodesHealthCheckName returns name of the health check resource used by
// the GCE load balancers (l4) for performing health checks on nodes.
func makeNodesHealthCheckName(clusterID string) string {
	return fmt.Sprintf("k8s-%v-node", clusterID)
}

func makeHealthCheckDescription(serviceName string) string {
	return fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName)
}

// MakeHealthCheckFirewallName returns the firewall name used by the GCE load
// balancers (l4) for performing health checks.
func MakeHealthCheckFirewallName(clusterID, hcName string, isNodesHealthCheck bool) string {
	if isNodesHealthCheck {
		return makeNodesHealthCheckName(clusterID) + "-http-hc"
	}
	return "k8s-" + hcName + "-http-hc"
}

func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

func makeFirewallDescription(serviceName, ipAddress string) string {
	return fmt.Sprintf(`{"kubernetes.io/service-name":"%s", "kubernetes.io/service-ip":"%s"}`,
		serviceName, ipAddress)
}
