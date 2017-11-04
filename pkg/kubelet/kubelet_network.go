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

package kubelet

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/network"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

const (
	// KubeMarkMasqChain is the mark-for-masquerade chain
	// TODO: clean up this logic in kube-proxy
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// KubeMarkDropChain is the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

	// KubePostroutingChain is kubernetes postrouting rules
	KubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// KubeFirewallChain is kubernetes firewall rules
	KubeFirewallChain utiliptables.Chain = "KUBE-FIREWALL"
)

// effectiveHairpinMode determines the effective hairpin mode given the
// configured mode, container runtime, and whether cbr0 should be configured.
func effectiveHairpinMode(hairpinMode kubeletconfig.HairpinMode, containerRuntime string, networkPlugin string) (kubeletconfig.HairpinMode, error) {
	// The hairpin mode setting doesn't matter if:
	// - We're not using a bridge network. This is hard to check because we might
	//   be using a plugin.
	// - It's set to hairpin-veth for a container runtime that doesn't know how
	//   to set the hairpin flag on the veth's of containers. Currently the
	//   docker runtime is the only one that understands this.
	// - It's set to "none".
	if hairpinMode == kubeletconfig.PromiscuousBridge || hairpinMode == kubeletconfig.HairpinVeth {
		// Only on docker.
		if containerRuntime != kubetypes.DockerContainerRuntime {
			glog.Warningf("Hairpin mode set to %q but container runtime is %q, ignoring", hairpinMode, containerRuntime)
			return kubeletconfig.HairpinNone, nil
		}
		if hairpinMode == kubeletconfig.PromiscuousBridge && networkPlugin != "kubenet" {
			// This is not a valid combination, since promiscuous-bridge only works on kubenet. Users might be using the
			// default values (from before the hairpin-mode flag existed) and we
			// should keep the old behavior.
			glog.Warningf("Hairpin mode set to %q but kubenet is not enabled, falling back to %q", hairpinMode, kubeletconfig.HairpinVeth)
			return kubeletconfig.HairpinVeth, nil
		}
	} else if hairpinMode != kubeletconfig.HairpinNone {
		return "", fmt.Errorf("unknown value: %q", hairpinMode)
	}
	return hairpinMode, nil
}

// providerRequiresNetworkingConfiguration returns whether the cloud provider
// requires special networking configuration.
func (kl *Kubelet) providerRequiresNetworkingConfiguration() bool {
	// TODO: We should have a mechanism to say whether native cloud provider
	// is used or whether we are using overlay networking. We should return
	// true for cloud providers if they implement Routes() interface and
	// we are not using overlay networking.
	if kl.cloud == nil || kl.cloud.ProviderName() != "gce" {
		return false
	}
	_, supported := kl.cloud.Routes()
	return supported
}

func omitDuplicates(kl *Kubelet, pod *v1.Pod, combinedSearch []string) []string {
	uniqueDomains := map[string]bool{}

	for _, dnsDomain := range combinedSearch {
		if _, exists := uniqueDomains[dnsDomain]; !exists {
			combinedSearch[len(uniqueDomains)] = dnsDomain
			uniqueDomains[dnsDomain] = true
		}
	}
	return combinedSearch[:len(uniqueDomains)]
}

func formDNSSearchFitsLimits(kl *Kubelet, pod *v1.Pod, composedSearch []string) []string {
	// resolver file Search line current limitations
	resolvSearchLineDNSDomainsLimit := 6
	resolvSearchLineLenLimit := 255
	limitsExceeded := false

	if len(composedSearch) > resolvSearchLineDNSDomainsLimit {
		composedSearch = composedSearch[:resolvSearchLineDNSDomainsLimit]
		limitsExceeded = true
	}

	if resolvSearchhLineStrLen := len(strings.Join(composedSearch, " ")); resolvSearchhLineStrLen > resolvSearchLineLenLimit {
		cutDomainsNum := 0
		cutDoaminsLen := 0
		for i := len(composedSearch) - 1; i >= 0; i-- {
			cutDoaminsLen += len(composedSearch[i]) + 1
			cutDomainsNum++

			if (resolvSearchhLineStrLen - cutDoaminsLen) <= resolvSearchLineLenLimit {
				break
			}
		}

		composedSearch = composedSearch[:(len(composedSearch) - cutDomainsNum)]
		limitsExceeded = true
	}

	if limitsExceeded {
		log := fmt.Sprintf("Search Line limits were exceeded, some dns names have been omitted, the applied search line is: %s", strings.Join(composedSearch, " "))
		kl.recorder.Event(pod, v1.EventTypeWarning, "DNSSearchForming", log)
		glog.Error(log)
	}
	return composedSearch
}

func (kl *Kubelet) formDNSSearchForDNSDefault(hostSearch []string, pod *v1.Pod) []string {
	return formDNSSearchFitsLimits(kl, pod, hostSearch)
}

func (kl *Kubelet) formDNSSearch(hostSearch []string, pod *v1.Pod) []string {
	if kl.clusterDomain == "" {
		formDNSSearchFitsLimits(kl, pod, hostSearch)
		return hostSearch
	}

	nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, kl.clusterDomain)
	svcDomain := fmt.Sprintf("svc.%s", kl.clusterDomain)
	dnsSearch := []string{nsSvcDomain, svcDomain, kl.clusterDomain}

	combinedSearch := append(dnsSearch, hostSearch...)

	combinedSearch = omitDuplicates(kl, pod, combinedSearch)
	return formDNSSearchFitsLimits(kl, pod, combinedSearch)
}

func (kl *Kubelet) checkLimitsForResolvConf() {
	// resolver file Search line current limitations
	resolvSearchLineDNSDomainsLimit := 6
	resolvSearchLineLenLimit := 255

	f, err := os.Open(kl.resolverConfig)
	if err != nil {
		kl.recorder.Event(kl.nodeRef, v1.EventTypeWarning, "checkLimitsForResolvConf", err.Error())
		glog.Error("checkLimitsForResolvConf: " + err.Error())
		return
	}
	defer f.Close()

	_, hostSearch, err := kl.parseResolvConf(f)
	if err != nil {
		kl.recorder.Event(kl.nodeRef, v1.EventTypeWarning, "checkLimitsForResolvConf", err.Error())
		glog.Error("checkLimitsForResolvConf: " + err.Error())
		return
	}

	domainCntLimit := resolvSearchLineDNSDomainsLimit

	if kl.clusterDomain != "" {
		domainCntLimit -= 3
	}

	if len(hostSearch) > domainCntLimit {
		log := fmt.Sprintf("Resolv.conf file '%s' contains search line consisting of more than %d domains!", kl.resolverConfig, domainCntLimit)
		kl.recorder.Event(kl.nodeRef, v1.EventTypeWarning, "checkLimitsForResolvConf", log)
		glog.Error("checkLimitsForResolvConf: " + log)
		return
	}

	if len(strings.Join(hostSearch, " ")) > resolvSearchLineLenLimit {
		log := fmt.Sprintf("Resolv.conf file '%s' contains search line which length is more than allowed %d chars!", kl.resolverConfig, resolvSearchLineLenLimit)
		kl.recorder.Event(kl.nodeRef, v1.EventTypeWarning, "checkLimitsForResolvConf", log)
		glog.Error("checkLimitsForResolvConf: " + log)
		return
	}

	return
}

// parseResolveConf reads a resolv.conf file from the given reader, and parses
// it into nameservers and searches, possibly returning an error.
// TODO: move to utility package
func (kl *Kubelet) parseResolvConf(reader io.Reader) (nameservers []string, searches []string, err error) {
	file, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, nil, err
	}

	// Lines of the form "nameserver 1.2.3.4" accumulate.
	nameservers = []string{}

	// Lines of the form "search example.com" overrule - last one wins.
	searches = []string{}

	lines := strings.Split(string(file), "\n")
	for l := range lines {
		trimmed := strings.TrimSpace(lines[l])
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) == 0 {
			continue
		}
		if fields[0] == "nameserver" && len(fields) >= 2 {
			nameservers = append(nameservers, fields[1])
		}
		if fields[0] == "search" {
			searches = fields[1:]
		}
	}

	// There used to be code here to scrub DNS for each cloud, but doesn't
	// make sense anymore since cloudproviders are being factored out.
	// contact @thockin or @wlan0 for more information

	return nameservers, searches, nil
}

// syncNetworkStatus updates the network state
func (kl *Kubelet) syncNetworkStatus() {
	// For cri integration, network state will be updated in updateRuntimeUp,
	// we'll get runtime network status through cri directly.
	// TODO: Remove this once we completely switch to cri integration.
	if kl.networkPlugin != nil {
		kl.runtimeState.setNetworkState(kl.networkPlugin.Status())
	}
}

// updatePodCIDR updates the pod CIDR in the runtime state if it is different
// from the current CIDR.
func (kl *Kubelet) updatePodCIDR(cidr string) {
	podCIDR := kl.runtimeState.podCIDR()

	if podCIDR == cidr {
		return
	}

	// kubelet -> network plugin
	// cri runtime shims are responsible for their own network plugins
	if kl.networkPlugin != nil {
		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = cidr
		kl.networkPlugin.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)
	}

	// kubelet -> generic runtime -> runtime shim -> network plugin
	// docker/rkt non-cri implementations have a passthrough UpdatePodCIDR
	if err := kl.GetRuntime().UpdatePodCIDR(cidr); err != nil {
		glog.Errorf("Failed to update pod CIDR: %v", err)
		return
	}

	glog.Infof("Setting Pod CIDR: %v -> %v", podCIDR, cidr)
	kl.runtimeState.setPodCIDR(cidr)
}

// syncNetworkUtil ensures the network utility are present on host.
// Network util includes:
// 1. 	In nat table, KUBE-MARK-DROP rule to mark connections for dropping
// 	Marked connection will be drop on INPUT/OUTPUT Chain in filter table
// 2. 	In nat table, KUBE-MARK-MASQ rule to mark connections for SNAT
// 	Marked connection will get SNAT on POSTROUTING Chain in nat table
func (kl *Kubelet) syncNetworkUtil() {
	if kl.iptablesMasqueradeBit < 0 || kl.iptablesMasqueradeBit > 31 {
		glog.Errorf("invalid iptables-masquerade-bit %v not in [0, 31]", kl.iptablesMasqueradeBit)
		return
	}

	if kl.iptablesDropBit < 0 || kl.iptablesDropBit > 31 {
		glog.Errorf("invalid iptables-drop-bit %v not in [0, 31]", kl.iptablesDropBit)
		return
	}

	if kl.iptablesDropBit == kl.iptablesMasqueradeBit {
		glog.Errorf("iptables-masquerade-bit %v and iptables-drop-bit %v must be different", kl.iptablesMasqueradeBit, kl.iptablesDropBit)
		return
	}

	// Setup KUBE-MARK-DROP rules
	dropMark := getIPTablesMark(kl.iptablesDropBit)
	if _, err := kl.iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkDropChain); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubeMarkDropChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkDropChain, "-j", "MARK", "--set-xmark", dropMark); err != nil {
		glog.Errorf("Failed to ensure marking rule for %v: %v", KubeMarkDropChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureChain(utiliptables.TableFilter, KubeFirewallChain); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableFilter, KubeFirewallChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Append, utiliptables.TableFilter, KubeFirewallChain,
		"-m", "comment", "--comment", "kubernetes firewall for dropping marked packets",
		"-m", "mark", "--mark", dropMark,
		"-j", "DROP"); err != nil {
		glog.Errorf("Failed to ensure rule to drop packet marked by %v in %v chain %v: %v", KubeMarkDropChain, utiliptables.TableFilter, KubeFirewallChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainOutput, "-j", string(KubeFirewallChain)); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableFilter, utiliptables.ChainOutput, KubeFirewallChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableFilter, utiliptables.ChainInput, "-j", string(KubeFirewallChain)); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableFilter, utiliptables.ChainInput, KubeFirewallChain, err)
		return
	}

	// Setup KUBE-MARK-MASQ rules
	masqueradeMark := getIPTablesMark(kl.iptablesMasqueradeBit)
	if _, err := kl.iptClient.EnsureChain(utiliptables.TableNAT, KubeMarkMasqChain); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubeMarkMasqChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureChain(utiliptables.TableNAT, KubePostroutingChain); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubePostroutingChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubeMarkMasqChain, "-j", "MARK", "--set-xmark", masqueradeMark); err != nil {
		glog.Errorf("Failed to ensure marking rule for %v: %v", KubeMarkMasqChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubernetes postrouting rules", "-j", string(KubePostroutingChain)); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, KubePostroutingChain, err)
		return
	}
	if _, err := kl.iptClient.EnsureRule(utiliptables.Append, utiliptables.TableNAT, KubePostroutingChain,
		"-m", "comment", "--comment", "kubernetes service traffic requiring SNAT",
		"-m", "mark", "--mark", masqueradeMark, "-j", "MASQUERADE"); err != nil {
		glog.Errorf("Failed to ensure SNAT rule for packets marked by %v in %v chain %v: %v", KubeMarkMasqChain, utiliptables.TableNAT, KubePostroutingChain, err)
		return
	}
}

// getIPTablesMark returns the fwmark given the bit
func getIPTablesMark(bit int) string {
	value := 1 << uint(bit)
	return fmt.Sprintf("%#08x/%#08x", value, value)
}
