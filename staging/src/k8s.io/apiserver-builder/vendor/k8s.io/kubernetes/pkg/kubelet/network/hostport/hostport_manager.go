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
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"strings"
	"sync"

	"github.com/golang/glog"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	iptablesproxy "k8s.io/kubernetes/pkg/proxy/iptables"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

// HostPortManager is an interface for adding and removing hostport for a given pod sandbox.
type HostPortManager interface {
	// Add implements port mappings.
	// id should be a unique identifier for a pod, e.g. podSandboxID.
	// podPortMapping is the associated port mapping information for the pod.
	// natInterfaceName is the interface that localhost used to talk to the given pod.
	Add(id string, podPortMapping *PodPortMapping, natInterfaceName string) error
	// Remove cleans up matching port mappings
	// Remove must be able to clean up port mappings without pod IP
	Remove(id string, podPortMapping *PodPortMapping) error
}

type hostportManager struct {
	hostPortMap map[hostport]closeable
	iptables    utiliptables.Interface
	portOpener  hostportOpener
	mu          sync.Mutex
}

func NewHostportManager() HostPortManager {
	iptInterface := utiliptables.New(utilexec.New(), utildbus.New(), utiliptables.ProtocolIpv4)
	return &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptInterface,
		portOpener:  openLocalPort,
	}
}

func (hm *hostportManager) Add(id string, podPortMapping *PodPortMapping, natInterfaceName string) (err error) {
	if podPortMapping == nil || podPortMapping.HostNetwork {
		return nil
	}
	podFullName := getPodFullName(podPortMapping)

	// skip if there is no hostport needed
	hostportMappings := gatherHostportMappings(podPortMapping)
	if len(hostportMappings) == 0 {
		return nil
	}

	if podPortMapping.IP.To4() == nil {
		return fmt.Errorf("invalid or missing IP of pod %s", podFullName)
	}
	podIP := podPortMapping.IP.String()

	if err = ensureKubeHostportChains(hm.iptables, natInterfaceName); err != nil {
		return err
	}

	// Ensure atomicity for port opening and iptables operations
	hm.mu.Lock()
	defer hm.mu.Unlock()

	// try to open hostports
	ports, err := openHostports(hm.portOpener, podPortMapping)
	if err != nil {
		return err
	}
	for hostport, socket := range ports {
		hm.hostPortMap[hostport] = socket
	}

	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	writeLine(natChains, "*nat")

	existingChains, existingRules, err := getExistingHostportIPTablesRules(hm.iptables)
	if err != nil {
		// clean up opened host port if encounter any error
		return utilerrors.NewAggregate([]error{err, hm.closeHostports(hostportMappings)})
	}

	newChains := []utiliptables.Chain{}
	for _, pm := range hostportMappings {
		protocol := strings.ToLower(string(pm.Protocol))
		chain := getHostportChain(id, pm)
		newChains = append(newChains, chain)

		// Add new hostport chain
		writeLine(natChains, utiliptables.MakeChainLine(chain))

		// Prepend the new chain to KUBE-HOSTPORTS
		// This avoids any leaking iptables rule that takes up the same port
		writeLine(natRules, "-I", string(kubeHostportsChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, podFullName, pm.HostPort),
			"-m", protocol, "-p", protocol, "--dport", fmt.Sprintf("%d", pm.HostPort),
			"-j", string(chain),
		)

		// SNAT if the traffic comes from the pod itself
		writeLine(natRules, "-A", string(chain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, podFullName, pm.HostPort),
			"-s", podIP,
			"-j", string(iptablesproxy.KubeMarkMasqChain))

		// DNAT to the podIP:containerPort
		writeLine(natRules, "-A", string(chain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, podFullName, pm.HostPort),
			"-m", protocol, "-p", protocol,
			"-j", "DNAT", fmt.Sprintf("--to-destination=%s:%d", podIP, pm.ContainerPort))
	}

	// getHostportChain should be able to provide unique hostport chain name using hash
	// if there is a chain conflict or multiple Adds have been triggered for a single pod,
	// filtering should be able to avoid further problem
	filterChains(existingChains, newChains)
	existingRules = filterRules(existingRules, newChains)

	for _, chain := range existingChains {
		writeLine(natChains, chain)
	}
	for _, rule := range existingRules {
		writeLine(natRules, rule)
	}
	writeLine(natRules, "COMMIT")

	if err = hm.syncIPTables(append(natChains.Bytes(), natRules.Bytes()...)); err != nil {
		// clean up opened host port if encounter any error
		return utilerrors.NewAggregate([]error{err, hm.closeHostports(hostportMappings)})
	}
	return nil
}

func (hm *hostportManager) Remove(id string, podPortMapping *PodPortMapping) (err error) {
	if podPortMapping == nil || podPortMapping.HostNetwork {
		return nil
	}

	hostportMappings := gatherHostportMappings(podPortMapping)
	if len(hostportMappings) <= 0 {
		return nil
	}

	// Ensure atomicity for port closing and iptables operations
	hm.mu.Lock()
	defer hm.mu.Unlock()

	var existingChains map[utiliptables.Chain]string
	var existingRules []string
	existingChains, existingRules, err = getExistingHostportIPTablesRules(hm.iptables)
	if err != nil {
		return err
	}

	// Gather target hostport chains for removal
	chainsToRemove := []utiliptables.Chain{}
	for _, pm := range hostportMappings {
		chainsToRemove = append(chainsToRemove, getHostportChain(id, pm))

		// To preserve backward compatibility for k8s 1.5 or earlier.
		// Need to remove hostport chains added by hostportSyncer if there is any
		// TODO: remove this in 1.7
		chainsToRemove = append(chainsToRemove, hostportChainName(pm, getPodFullName(podPortMapping)))
	}

	// remove rules that consists of target chains
	remainingRules := filterRules(existingRules, chainsToRemove)

	// gather target hostport chains that exists in iptables-save result
	existingChainsToRemove := []utiliptables.Chain{}
	for _, chain := range chainsToRemove {
		if _, ok := existingChains[chain]; ok {
			existingChainsToRemove = append(existingChainsToRemove, chain)
		}
	}

	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	writeLine(natChains, "*nat")
	for _, chain := range existingChains {
		writeLine(natChains, chain)
	}
	for _, rule := range remainingRules {
		writeLine(natRules, rule)
	}
	for _, chain := range existingChainsToRemove {
		writeLine(natRules, "-X", string(chain))
	}
	writeLine(natRules, "COMMIT")

	if err = hm.syncIPTables(append(natChains.Bytes(), natRules.Bytes()...)); err != nil {
		return err
	}

	// clean up opened pod host ports
	return hm.closeHostports(hostportMappings)
}

// syncIPTables executes iptables-restore with given lines
func (hm *hostportManager) syncIPTables(lines []byte) error {
	glog.V(3).Infof("Restoring iptables rules: %s", lines)
	err := hm.iptables.RestoreAll(lines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		return fmt.Errorf("Failed to execute iptables-restore: %v", err)
	}
	return nil
}

// closeHostports tries to close all the listed host ports
// TODO: move closeHostports and openHostports into a common struct
func (hm *hostportManager) closeHostports(hostportMappings []*PortMapping) error {
	errList := []error{}
	for _, pm := range hostportMappings {
		hp := portMappingToHostport(pm)
		if socket, ok := hm.hostPortMap[hp]; ok {
			glog.V(2).Infof("Closing host port %s", hp.String())
			if err := socket.Close(); err != nil {
				errList = append(errList, fmt.Errorf("failed to close host port %s: %v", hp.String(), err))
				continue
			}
			delete(hm.hostPortMap, hp)
		}
	}
	return utilerrors.NewAggregate(errList)
}

// getHostportChain takes id, hostport and protocol for a pod and returns associated iptables chain.
// This is computed by hashing (sha256) then encoding to base32 and truncating with the prefix
// "KUBE-HP-". We do this because IPTables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
// WARNING: Please do not change this function. Otherwise, HostportManager may not be able to
// identify existing iptables chains.
func getHostportChain(id string, pm *PortMapping) utiliptables.Chain {
	hash := sha256.Sum256([]byte(id + string(pm.HostPort) + string(pm.Protocol)))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain(kubeHostportChainPrefix + encoded[:16])
}

// gatherHostportMappings returns all the PortMappings which has hostport for a pod
func gatherHostportMappings(podPortMapping *PodPortMapping) []*PortMapping {
	mappings := []*PortMapping{}
	for _, pm := range podPortMapping.PortMappings {
		if pm.HostPort <= 0 {
			continue
		}
		mappings = append(mappings, pm)
	}
	return mappings
}

// getExistingHostportIPTablesRules retrieves raw data from iptables-save, parse it,
// return all the hostport related chains and rules
func getExistingHostportIPTablesRules(iptables utiliptables.Interface) (map[utiliptables.Chain]string, []string, error) {
	iptablesSaveRaw, err := iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		return nil, nil, fmt.Errorf("failed to execute iptables-save: %v", err)
	}
	existingNATChains := utiliptables.GetChainLines(utiliptables.TableNAT, iptablesSaveRaw)

	existingHostportChains := make(map[utiliptables.Chain]string)
	existingHostportRules := []string{}

	for chain := range existingNATChains {
		if strings.HasPrefix(string(chain), string(kubeHostportsChain)) || strings.HasPrefix(string(chain), kubeHostportChainPrefix) {
			existingHostportChains[chain] = existingNATChains[chain]
		}
	}

	for _, line := range strings.Split(string(iptablesSaveRaw), "\n") {
		if strings.HasPrefix(line, fmt.Sprintf("-A %s", kubeHostportChainPrefix)) ||
			strings.HasPrefix(line, fmt.Sprintf("-A %s", string(kubeHostportsChain))) {
			existingHostportRules = append(existingHostportRules, line)
		}
	}
	return existingHostportChains, existingHostportRules, nil
}

// filterRules filters input rules with input chains. Rules that did not involve any filter chain will be returned.
// The order of the input rules is important and is preserved.
func filterRules(rules []string, filters []utiliptables.Chain) []string {
	filtered := []string{}
	for _, rule := range rules {
		skip := false
		for _, filter := range filters {
			if strings.Contains(rule, string(filter)) {
				skip = true
				break
			}
		}
		if !skip {
			filtered = append(filtered, rule)
		}
	}
	return filtered
}

// filterChains deletes all entries of filter chains from chain map
func filterChains(chains map[utiliptables.Chain]string, filterChains []utiliptables.Chain) {
	for _, chain := range filterChains {
		if _, ok := chains[chain]; ok {
			delete(chains, chain)
		}
	}
}
