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

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
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

// This just exports required functions from kubelet proper, for use by network
// plugins.
// TODO(#35457): get rid of this backchannel to the kubelet. The scope of
// the back channel is restricted to host-ports/testing, and restricted
// to kubenet. No other network plugin wrapper needs it. Other plugins
// only require a way to access namespace information, which they can do
// directly through the methods implemented by criNetworkHost.
type networkHost struct {
	kubelet *Kubelet
}

func (nh *networkHost) GetPodByName(name, namespace string) (*v1.Pod, bool) {
	return nh.kubelet.GetPodByName(name, namespace)
}

func (nh *networkHost) GetKubeClient() clientset.Interface {
	return nh.kubelet.kubeClient
}

func (nh *networkHost) GetRuntime() kubecontainer.Runtime {
	return nh.kubelet.getRuntime()
}

func (nh *networkHost) SupportsLegacyFeatures() bool {
	return true
}

// criNetworkHost implements the part of network.Host required by the
// cri (NamespaceGetter). It leechs off networkHost for all other
// methods, because networkHost is slated for deletion.
type criNetworkHost struct {
	*networkHost
	// criNetworkHost currently support legacy features. Hence no need to support PortMappingGetter
	*network.NoopPortMappingGetter
}

// GetNetNS returns the network namespace of the given containerID.
// This method satisfies the network.NamespaceGetter interface for
// networkHost. It's only meant to be used from network plugins
// that are directly invoked by the kubelet (aka: legacy, pre-cri).
// Any network plugin invoked by a cri must implement NamespaceGetter
// to talk directly to the runtime instead.
func (c *criNetworkHost) GetNetNS(containerID string) (string, error) {
	return c.kubelet.getRuntime().GetNetNS(kubecontainer.ContainerID{Type: "", ID: containerID})
}

// NoOpLegacyHost implements the network.LegacyHost interface for the remote
// runtime shim by just returning empties. It doesn't support legacy features
// like host port and bandwidth shaping.
type NoOpLegacyHost struct{}

// GetPodByName always returns "nil, true" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetPodByName(namespace, name string) (*v1.Pod, bool) {
	return nil, true
}

// GetKubeClient always returns "nil" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetKubeClient() clientset.Interface {
	return nil
}

// getRuntime always returns "nil" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) GetRuntime() kubecontainer.Runtime {
	return nil
}

// SupportsLegacyFeatures always returns "false" for 'NoOpLegacyHost'
func (n *NoOpLegacyHost) SupportsLegacyFeatures() bool {
	return false
}

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
	if err := kl.getRuntime().UpdatePodCIDR(cidr); err != nil {
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

// GetPodDNS returns DNS settings for the pod.
// This function is defined in kubecontainer.RuntimeHelper interface so we
// have to implement it.
func (kl *Kubelet) GetPodDNS(pod *v1.Pod) (*runtimeapi.DNSConfig, error) {
	return kl.dnsConfigurer.GetPodDNS(pod)
}
