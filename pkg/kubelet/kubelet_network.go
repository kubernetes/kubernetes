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
	"net"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	// the mark-for-masquerade chain
	// TODO: clean up this logic in kube-proxy
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

	// kubernetes postrouting rules
	KubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// kubernetes firewall rules
	KubeFirewallChain utiliptables.Chain = "KUBE-FIREWALL"
)

// effectiveHairpinMode determines the effective hairpin mode given the
// configured mode, container runtime, and whether cbr0 should be configured.
func effectiveHairpinMode(hairpinMode componentconfig.HairpinMode, containerRuntime string, configureCBR0 bool, networkPlugin string) (componentconfig.HairpinMode, error) {
	// The hairpin mode setting doesn't matter if:
	// - We're not using a bridge network. This is hard to check because we might
	//   be using a plugin. It matters if --configure-cbr0=true, and we currently
	//   don't pipe it down to any plugins.
	// - It's set to hairpin-veth for a container runtime that doesn't know how
	//   to set the hairpin flag on the veth's of containers. Currently the
	//   docker runtime is the only one that understands this.
	// - It's set to "none".
	if hairpinMode == componentconfig.PromiscuousBridge || hairpinMode == componentconfig.HairpinVeth {
		// Only on docker.
		if containerRuntime != "docker" {
			glog.Warningf("Hairpin mode set to %q but container runtime is %q, ignoring", hairpinMode, containerRuntime)
			return componentconfig.HairpinNone, nil
		}
		if hairpinMode == componentconfig.PromiscuousBridge && !configureCBR0 && networkPlugin != "kubenet" {
			// This is not a valid combination.  Users might be using the
			// default values (from before the hairpin-mode flag existed) and we
			// should keep the old behavior.
			glog.Warningf("Hairpin mode set to %q but configureCBR0 is false, falling back to %q", hairpinMode, componentconfig.HairpinVeth)
			return componentconfig.HairpinVeth, nil
		}
	} else if hairpinMode == componentconfig.HairpinNone {
		if configureCBR0 {
			glog.Warningf("Hairpin mode set to %q and configureCBR0 is true, this might result in loss of hairpin packets", hairpinMode)
		}
	} else {
		return "", fmt.Errorf("unknown value: %q", hairpinMode)
	}
	return hairpinMode, nil
}

// Validate given node IP belongs to the current host
func (kl *Kubelet) validateNodeIP() error {
	if kl.nodeIP == nil {
		return nil
	}

	// Honor IP limitations set in setNodeStatus()
	if kl.nodeIP.IsLoopback() {
		return fmt.Errorf("nodeIP can't be loopback address")
	}
	if kl.nodeIP.To4() == nil {
		return fmt.Errorf("nodeIP must be IPv4 address")
	}

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return err
	}
	for _, addr := range addrs {
		var ip net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip != nil && ip.Equal(kl.nodeIP) {
			return nil
		}
	}
	return fmt.Errorf("Node IP: %q not found in the host's network interfaces", kl.nodeIP.String())
}

// providerRequiresNetworkingConfiguration returns whether the cloud provider
// requires special networking configuration.
func (kl *Kubelet) providerRequiresNetworkingConfiguration() bool {
	// TODO: We should have a mechanism to say whether native cloud provider
	// is used or whether we are using overlay networking. We should return
	// true for cloud providers if they implement Routes() interface and
	// we are not using overlay networking.
	if kl.cloud == nil || kl.cloud.ProviderName() != "gce" || kl.flannelExperimentalOverlay {
		return false
	}
	_, supported := kl.cloud.Routes()
	return supported
}

// Returns the list of DNS servers and DNS search domains.
func (kl *Kubelet) parseResolvConf(reader io.Reader) (nameservers []string, searches []string, err error) {
	var scrubber dnsScrubber
	if kl.cloud != nil {
		scrubber = kl.cloud
	}
	return parseResolvConf(reader, scrubber)
}

// A helper for testing.
type dnsScrubber interface {
	ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string)
}

// parseResolveConf reads a resolv.conf file from the given reader, and parses
// it into nameservers and searches, possibly returning an error.  The given
// dnsScrubber allows cloud providers to post-process dns names.
// TODO: move to utility package
func parseResolvConf(reader io.Reader, dnsScrubber dnsScrubber) (nameservers []string, searches []string, err error) {
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
		if fields[0] == "nameserver" {
			nameservers = append(nameservers, fields[1:]...)
		}
		if fields[0] == "search" {
			searches = fields[1:]
		}
	}

	// Give the cloud-provider a chance to post-process DNS settings.
	if dnsScrubber != nil {
		nameservers, searches = dnsScrubber.ScrubDNS(nameservers, searches)
	}
	return nameservers, searches, nil
}

// cleanupBandwidthLimits updates the status of bandwidth-limited containers
// and ensures that only the appropriate CIDRs are active on the node.
func (kl *Kubelet) cleanupBandwidthLimits(allPods []*api.Pod) error {
	if kl.shaper == nil {
		return nil
	}
	currentCIDRs, err := kl.shaper.GetCIDRs()
	if err != nil {
		return err
	}
	possibleCIDRs := sets.String{}
	for ix := range allPods {
		pod := allPods[ix]
		ingress, egress, err := bandwidth.ExtractPodBandwidthResources(pod.Annotations)
		if err != nil {
			return err
		}
		if ingress == nil && egress == nil {
			glog.V(8).Infof("Not a bandwidth limited container...")
			continue
		}
		status, found := kl.statusManager.GetPodStatus(pod.UID)
		if !found {
			// TODO(random-liu): Cleanup status get functions. (issue #20477)
			s, err := kl.containerRuntime.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
			if err != nil {
				return err
			}
			status = kl.generateAPIPodStatus(pod, s)
		}
		if status.Phase == api.PodRunning {
			possibleCIDRs.Insert(fmt.Sprintf("%s/32", status.PodIP))
		}
	}
	for _, cidr := range currentCIDRs {
		if !possibleCIDRs.Has(cidr) {
			glog.V(2).Infof("Removing CIDR: %s (%v)", cidr, possibleCIDRs)
			if err := kl.shaper.Reset(cidr); err != nil {
				return err
			}
		}
	}
	return nil
}

// TODO: remove when kubenet plugin is ready
// NOTE!!! if you make changes here, also make them to kubenet
func (kl *Kubelet) reconcileCBR0(podCIDR string) error {
	if podCIDR == "" {
		glog.V(5).Info("PodCIDR not set. Will not configure cbr0.")
		return nil
	}
	glog.V(5).Infof("PodCIDR is set to %q", podCIDR)
	_, cidr, err := net.ParseCIDR(podCIDR)
	if err != nil {
		return err
	}
	// Set cbr0 interface address to first address in IPNet
	cidr.IP.To4()[3] += 1
	if err := ensureCbr0(cidr, kl.hairpinMode == componentconfig.PromiscuousBridge, kl.babysitDaemons); err != nil {
		return err
	}
	if kl.shapingEnabled() {
		if kl.shaper == nil {
			glog.V(5).Info("Shaper is nil, creating")
			kl.shaper = bandwidth.NewTCShaper("cbr0")
		}
		return kl.shaper.ReconcileInterface()
	}
	return nil
}

// syncNetworkStatus updates the network state, ensuring that the network is
// configured correctly if the kubelet is set to configure cbr0:
// * handshake flannel helper if the flannel experimental overlay is being used.
// * ensure that iptables masq rules are setup
// * reconcile cbr0 with the pod CIDR
func (kl *Kubelet) syncNetworkStatus() {
	var err error
	if kl.configureCBR0 {
		if kl.flannelExperimentalOverlay {
			podCIDR, err := kl.flannelHelper.Handshake()
			if err != nil {
				glog.Infof("Flannel server handshake failed %v", err)
				return
			}
			kl.updatePodCIDR(podCIDR)
		}
		if err := ensureIPTablesMasqRule(kl.iptClient, kl.nonMasqueradeCIDR); err != nil {
			err = fmt.Errorf("Error on adding ip table rules: %v", err)
			glog.Error(err)
			kl.runtimeState.setNetworkState(err)
			return
		}
		podCIDR := kl.runtimeState.podCIDR()
		if len(podCIDR) == 0 {
			err = fmt.Errorf("ConfigureCBR0 requested, but PodCIDR not set. Will not configure CBR0 right now")
			glog.Warning(err)
		} else if err = kl.reconcileCBR0(podCIDR); err != nil {
			err = fmt.Errorf("Error configuring cbr0: %v", err)
			glog.Error(err)
		}
		if err != nil {
			kl.runtimeState.setNetworkState(err)
			return
		}
	}

	kl.runtimeState.setNetworkState(kl.networkPlugin.Status())
}

// updatePodCIDR updates the pod CIDR in the runtime state if it is different
// from the current CIDR.
func (kl *Kubelet) updatePodCIDR(cidr string) {
	podCIDR := kl.runtimeState.podCIDR()

	if podCIDR == cidr {
		return
	}

	glog.Infof("Setting Pod CIDR: %v -> %v", podCIDR, cidr)
	kl.runtimeState.setPodCIDR(cidr)

	if kl.networkPlugin != nil {
		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = cidr
		kl.networkPlugin.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)
	}
}

// shapingEnabled returns whether traffic shaping is enabled.
func (kl *Kubelet) shapingEnabled() bool {
	// Disable shaping if a network plugin is defined and supports shaping
	if kl.networkPlugin != nil && kl.networkPlugin.Capabilities().Has(network.NET_PLUGIN_CAPABILITY_SHAPING) {
		return false
	}
	return true
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
