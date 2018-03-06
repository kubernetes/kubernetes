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

package hostport

import (
	"fmt"
	"net"
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/kubernetes/pkg/util/net"
)

const (
	// the hostport chain
	kubeHostportsChain utiliptables.Chain = "KUBE-HOSTPORTS"
	// prefix for hostport chains
	kubeHostportChainPrefix string = "KUBE-HP-"
)

// PortMapping represents a network port in a container
type PortMapping struct {
	Name          string
	HostPort      int32
	ContainerPort int32
	Protocol      v1.Protocol
	HostIP        string
}

// PodPortMapping represents a pod's network state and associated container port mappings
type PodPortMapping struct {
	Namespace    string
	Name         string
	PortMappings []*PortMapping
	HostNetwork  bool
	IP           net.IP
}

// ConstructPodPortMapping creates a PodPortMapping from the ports specified in the pod's
// containers.
func ConstructPodPortMapping(pod *v1.Pod, podIP net.IP) *PodPortMapping {
	portMappings := make([]*PortMapping, 0)
	for _, c := range pod.Spec.Containers {
		for _, port := range c.Ports {
			portMappings = append(portMappings, &PortMapping{
				Name:          port.Name,
				HostPort:      port.HostPort,
				ContainerPort: port.ContainerPort,
				Protocol:      port.Protocol,
				HostIP:        port.HostIP,
			})
		}
	}

	return &PodPortMapping{
		Namespace:    pod.Namespace,
		Name:         pod.Name,
		PortMappings: portMappings,
		HostNetwork:  pod.Spec.HostNetwork,
		IP:           podIP,
	}
}

type hostportOpener func(IP string, port int, proto string, desc string) (utilnet.Closeable, error)

// openHostports opens all given hostports using the given hostportOpener
// If encounter any error, clean up and return the error
// If all ports are opened successfully, return the hostport and socket mapping
// TODO: move openHostports and closeHostports into a common struct
func openHostports(portOpener hostportOpener, podPortMapping *PodPortMapping) (map[string]utilnet.Closeable, error) {
	var retErr error
	ports := make(map[string]utilnet.Closeable)
	for _, pm := range podPortMapping.PortMappings {
		if pm.HostPort <= 0 {
			continue
		}
		socket, err := portOpener(pm.HostIP, int(pm.HostPort), strings.ToLower(string(pm.Protocol)), "")
		if err != nil {
			retErr = fmt.Errorf("cannot open hostport %d for pod %s: %v", pm.HostPort, getPodFullName(podPortMapping), err)
			break
		}
		ports[utilnet.ToLocalPortString(pm.HostIP, int(pm.HostPort), strings.ToLower(string(pm.Protocol)), "")] = socket
	}

	// If encounter any error, close all hostports that just got opened.
	if retErr != nil {
		for hp, socket := range ports {
			if err := socket.Close(); err != nil {
				glog.Errorf("Cannot clean up hostport %s for pod %s: %v", hp, getPodFullName(podPortMapping), err)
			}
		}
		return nil, retErr
	}
	return ports, nil
}

// ensureKubeHostportChains ensures the KUBE-HOSTPORTS chain is setup correctly
func ensureKubeHostportChains(iptables utiliptables.Interface, natInterfaceName string) error {
	glog.V(4).Info("Ensuring kubelet hostport chains")
	// Ensure kubeHostportChain
	if _, err := iptables.EnsureChain(utiliptables.TableNAT, kubeHostportsChain); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubeHostportsChain, err)
	}
	tableChainsNeedJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	args := []string{"-m", "comment", "--comment", "kube hostport portals",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubeHostportsChain)}
	for _, tc := range tableChainsNeedJumpServices {
		// KUBE-HOSTPORTS chain needs to be appended to the system chains.
		// This ensures KUBE-SERVICES chain gets processed first.
		// Since rules in KUBE-HOSTPORTS chain matches broader cases, allow the more specific rules to be processed first.
		if _, err := iptables.EnsureRule(utiliptables.Append, tc.table, tc.chain, args...); err != nil {
			return fmt.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubeHostportsChain, err)
		}
	}
	// Need to SNAT traffic from localhost
	args = []string{"-m", "comment", "--comment", "SNAT for localhost access to hostports", "-o", natInterfaceName, "-s", "127.0.0.0/8", "-j", "MASQUERADE"}
	if _, err := iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s jumps to MASQUERADE: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, err)
	}
	return nil
}
