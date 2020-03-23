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

package hostport

import (
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	iptablesproxy "k8s.io/kubernetes/pkg/proxy/iptables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/utils/net"
)

// HostportSyncer takes a list of PodPortMappings and implements hostport all at once
type HostportSyncer interface {
	// SyncHostports gathers all hostports on node and setup iptables rules to enable them.
	// On each invocation existing ports are synced and stale rules are deleted.
	SyncHostports(natInterfaceName string, activePodPortMappings []*PodPortMapping) error
	// OpenPodHostportsAndSync opens hostports for a new PodPortMapping, gathers all hostports on
	// node, sets up iptables rules enable them. On each invocation existing ports are synced and stale rules are deleted.
	// 'newPortMapping' must also be present in 'activePodPortMappings'.
	OpenPodHostportsAndSync(newPortMapping *PodPortMapping, natInterfaceName string, activePodPortMappings []*PodPortMapping) error
}

type hostportSyncer struct {
	hostPortMap map[hostport]closeable
	iptables    utiliptables.Interface
	portOpener  hostportOpener
}

func NewHostportSyncer(iptables utiliptables.Interface) HostportSyncer {
	return &hostportSyncer{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  openLocalPort,
	}
}

type targetPod struct {
	podFullName string
	podIP       string
}

func (hp *hostport) String() string {
	return fmt.Sprintf("%s:%d", hp.protocol, hp.port)
}

// openHostports opens all hostport for pod and returns the map of hostport and socket
func (h *hostportSyncer) openHostports(podHostportMapping *PodPortMapping) error {
	var retErr error
	ports := make(map[hostport]closeable)
	for _, port := range podHostportMapping.PortMappings {
		if port.HostPort <= 0 {
			// Assume hostport is not specified in this portmapping. So skip
			continue
		}

		// We do not open host ports for SCTP ports, as we agreed in the Support of SCTP KEP
		if port.Protocol == v1.ProtocolSCTP {
			continue
		}

		hp := hostport{
			port:     port.HostPort,
			protocol: strings.ToLower(string(port.Protocol)),
		}
		socket, err := h.portOpener(&hp)
		if err != nil {
			retErr = fmt.Errorf("cannot open hostport %d for pod %s: %v", port.HostPort, getPodFullName(podHostportMapping), err)
			break
		}
		ports[hp] = socket
	}

	// If encounter any error, close all hostports that just got opened.
	if retErr != nil {
		for hp, socket := range ports {
			if err := socket.Close(); err != nil {
				klog.Errorf("Cannot clean up hostport %d for pod %s: %v", hp.port, getPodFullName(podHostportMapping), err)
			}
		}
		return retErr
	}

	for hostPort, socket := range ports {
		h.hostPortMap[hostPort] = socket
	}

	return nil
}

func getPodFullName(pod *PodPortMapping) string {
	// Use underscore as the delimiter because it is not allowed in pod name
	// (DNS subdomain format), while allowed in the container name format.
	return pod.Name + "_" + pod.Namespace
}

// gatherAllHostports returns all hostports that should be presented on node,
// given the list of pods running on that node and ignoring host network
// pods (which don't need hostport <-> container port mapping)
// It only returns the hosports that match the IP family passed as parameter
func gatherAllHostports(activePodPortMappings []*PodPortMapping, isIPv6 bool) (map[*PortMapping]targetPod, error) {
	podHostportMap := make(map[*PortMapping]targetPod)
	for _, pm := range activePodPortMappings {
		// IP.To16() returns nil if IP is not a valid IPv4 or IPv6 address
		if pm.IP.To16() == nil {
			return nil, fmt.Errorf("Invalid or missing pod %s IP", getPodFullName(pm))
		}
		// return only entries from the same IP family
		if utilnet.IsIPv6(pm.IP) != isIPv6 {
			continue
		}
		// should not handle hostports for hostnetwork pods
		if pm.HostNetwork {
			continue
		}

		for _, port := range pm.PortMappings {
			if port.HostPort != 0 {
				podHostportMap[port] = targetPod{podFullName: getPodFullName(pm), podIP: pm.IP.String()}
			}
		}
	}
	return podHostportMap, nil
}

// Join all words with spaces, terminate with newline and write to buf.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
}

func writeBytesLine(buf *bytes.Buffer, bytes []byte) {
	buf.Write(bytes)
	buf.WriteByte('\n')
}

//hostportChainName takes containerPort for a pod and returns associated iptables chain.
// This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-SVC-".  We do
// this because IPTables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
func hostportChainName(pm *PortMapping, podFullName string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(strconv.Itoa(int(pm.HostPort)) + string(pm.Protocol) + podFullName))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain(kubeHostportChainPrefix + encoded[:16])
}

// OpenPodHostportsAndSync opens hostports for a new PodPortMapping, gathers all hostports on
// node, sets up iptables rules enable them. And finally clean up stale hostports.
// 'newPortMapping' must also be present in 'activePodPortMappings'.
func (h *hostportSyncer) OpenPodHostportsAndSync(newPortMapping *PodPortMapping, natInterfaceName string, activePodPortMappings []*PodPortMapping) error {
	// try to open pod host port if specified
	if err := h.openHostports(newPortMapping); err != nil {
		return err
	}

	// Add the new pod to active pods if it's not present.
	var found bool
	for _, pm := range activePodPortMappings {
		if pm.Namespace == newPortMapping.Namespace && pm.Name == newPortMapping.Name {
			found = true
			break
		}
	}
	if !found {
		activePodPortMappings = append(activePodPortMappings, newPortMapping)
	}

	return h.SyncHostports(natInterfaceName, activePodPortMappings)
}

// SyncHostports gathers all hostports on node and setup iptables rules enable them. And finally clean up stale hostports
func (h *hostportSyncer) SyncHostports(natInterfaceName string, activePodPortMappings []*PodPortMapping) error {
	start := time.Now()
	defer func() {
		klog.V(4).Infof("syncHostportsRules took %v", time.Since(start))
	}()

	hostportPodMap, err := gatherAllHostports(activePodPortMappings, h.iptables.IsIpv6())
	if err != nil {
		return err
	}

	// Ensure KUBE-HOSTPORTS chains
	ensureKubeHostportChains(h.iptables, natInterfaceName)

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain][]byte)
	iptablesData := bytes.NewBuffer(nil)
	err = h.iptables.SaveInto(utiliptables.TableNAT, iptablesData)
	if err != nil { // if we failed to get any rules
		klog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, iptablesData.Bytes())
	}

	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	writeLine(natChains, "*nat")
	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingNATChains[kubeHostportsChain]; ok {
		writeBytesLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeHostportsChain))
	}

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

	for port, target := range hostportPodMap {
		protocol := strings.ToLower(string(port.Protocol))
		hostportChain := hostportChainName(port, target.podFullName)

		if chain, ok := existingNATChains[hostportChain]; ok {
			writeBytesLine(natChains, chain)
		} else {
			writeLine(natChains, utiliptables.MakeChainLine(hostportChain))
		}

		activeNATChains[hostportChain] = true

		// Redirect to hostport chain
		args := []string{
			"-A", string(kubeHostportsChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, port.HostPort),
			"-m", protocol, "-p", protocol,
			"--dport", fmt.Sprintf("%d", port.HostPort),
			"-j", string(hostportChain),
		}
		writeLine(natRules, args...)

		// Assuming kubelet is syncing iptables KUBE-MARK-MASQ chain
		// If the request comes from the pod that is serving the hostport, then SNAT
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, port.HostPort),
			"-s", target.podIP, "-j", string(iptablesproxy.KubeMarkMasqChain),
		}
		writeLine(natRules, args...)

		// Create hostport chain to DNAT traffic to final destination
		// IPTables will maintained the stats for this chain
		hostPortBinding := net.JoinHostPort(target.podIP, strconv.Itoa(int(port.ContainerPort)))
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, port.HostPort),
			"-m", protocol, "-p", protocol,
			"-j", "DNAT", fmt.Sprintf("--to-destination=%s", hostPortBinding),
		}
		writeLine(natRules, args...)
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, kubeHostportChainPrefix) {
				// Ignore chains that aren't ours.
				continue
			}
			// We must (as per iptables) write a chain-line for it, which has
			// the nice effect of flushing the chain.  Then we can remove the
			// chain.
			writeBytesLine(natChains, existingNATChains[chain])
			writeLine(natRules, "-X", chainString)
		}
	}
	writeLine(natRules, "COMMIT")

	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	klog.V(3).Infof("Restoring iptables rules: %s", natLines)
	err = h.iptables.RestoreAll(natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		return fmt.Errorf("failed to execute iptables-restore: %v", err)
	}

	h.cleanupHostportMap(hostportPodMap)
	return nil
}

// cleanupHostportMap closes obsolete hostports
func (h *hostportSyncer) cleanupHostportMap(containerPortMap map[*PortMapping]targetPod) {
	// compute hostports that are supposed to be open
	currentHostports := make(map[hostport]bool)
	for containerPort := range containerPortMap {
		hp := hostport{
			port:     containerPort.HostPort,
			protocol: strings.ToLower(string(containerPort.Protocol)),
		}
		currentHostports[hp] = true
	}

	// close and delete obsolete hostports
	for hp, socket := range h.hostPortMap {
		if _, ok := currentHostports[hp]; !ok {
			socket.Close()
			klog.V(3).Infof("Closed local port %s", hp.String())
			delete(h.hostPortMap, hp)
		}
	}
}
