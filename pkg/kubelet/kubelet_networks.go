/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// This just exports required functions from kubelet proper, for use by network
// plugins.
type networkHost struct {
	kubelet *Kubelet
}

func (nh *networkHost) GetPodByName(name, namespace string) (*api.Pod, bool) {
	return nh.kubelet.GetPodByName(name, namespace)
}

func (nh *networkHost) GetKubeClient() clientset.Interface {
	return nh.kubelet.kubeClient
}

func (nh *networkHost) GetRuntime() kubecontainer.Runtime {
	return nh.kubelet.GetRuntime()
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

// effectiveHairpinMode determines the effective hairpin mode given the
// configured mode, container runtime, and whether cbr0 should be configured.
func effectiveHairpinMode(hairpinMode componentconfig.HairpinMode, containerRuntime string, configureCBR0 bool) (componentconfig.HairpinMode, error) {
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
		if hairpinMode == componentconfig.PromiscuousBridge && !configureCBR0 {
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

// GetClusterDNS returns a list of the DNS servers and a list of the DNS search
// domains of the cluster.
func (kl *Kubelet) GetClusterDNS(pod *api.Pod) ([]string, []string, error) {
	var hostDNS, hostSearch []string
	// Get host DNS settings
	if kl.resolverConfig != "" {
		f, err := os.Open(kl.resolverConfig)
		if err != nil {
			return nil, nil, err
		}
		defer f.Close()

		hostDNS, hostSearch, err = kl.parseResolvConf(f)
		if err != nil {
			return nil, nil, err
		}
	}
	useClusterFirstPolicy := pod.Spec.DNSPolicy == api.DNSClusterFirst
	if useClusterFirstPolicy && kl.clusterDNS == nil {
		// clusterDNS is not known.
		// pod with ClusterDNSFirst Policy cannot be created
		kl.recorder.Eventf(pod, api.EventTypeWarning, "MissingClusterDNS", "kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy)
		log := fmt.Sprintf("kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. pod: %q. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy, format.Pod(pod))
		kl.recorder.Eventf(kl.nodeRef, api.EventTypeWarning, "MissingClusterDNS", log)

		// fallback to DNSDefault
		useClusterFirstPolicy = false
	}

	if !useClusterFirstPolicy {
		// When the kubelet --resolv-conf flag is set to the empty string, use
		// DNS settings that override the docker default (which is to use
		// /etc/resolv.conf) and effectivly disable DNS lookups. According to
		// the bind documentation, the behavior of the DNS client library when
		// "nameservers" are not specified is to "use the nameserver on the
		// local machine". A nameserver setting of localhost is equivalent to
		// this documented behavior.
		if kl.resolverConfig == "" {
			hostDNS = []string{"127.0.0.1"}
			hostSearch = []string{"."}
		}
		return hostDNS, hostSearch, nil
	}

	// for a pod with DNSClusterFirst policy, the cluster DNS server is the only nameserver configured for
	// the pod. The cluster DNS server itself will forward queries to other nameservers that is configured to use,
	// in case the cluster DNS server cannot resolve the DNS query itself
	dns := []string{kl.clusterDNS.String()}

	var dnsSearch []string
	if kl.clusterDomain != "" {
		nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, kl.clusterDomain)
		svcDomain := fmt.Sprintf("svc.%s", kl.clusterDomain)
		dnsSearch = append([]string{nsSvcDomain, svcDomain, kl.clusterDomain}, hostSearch...)
	} else {
		dnsSearch = hostSearch
	}
	return dns, dnsSearch, nil
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

// returns whether the pod uses the host network namespace.
func podUsesHostNetwork(pod *api.Pod) bool {
	return pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork
}

// cleanupBandwidthLimits updates the status of bandwidth-limited containers
// and ensures that only the the appropriate CIDRs are active on the node.
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

// checkHostPortConflicts detects pods with conflicted host ports.
func hasHostPortConflicts(pods []*api.Pod) bool {
	ports := sets.String{}
	for _, pod := range pods {
		if errs := validation.AccumulateUniqueHostPorts(pod.Spec.Containers, &ports, field.NewPath("spec", "containers")); len(errs) > 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", format.Pod(pod), errs)
			return true
		}
	}
	return false
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

// Set addresses for the node.
func (kl *Kubelet) setNodeAddress(node *api.Node) error {
	// Set addresses for the node.
	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return fmt.Errorf("failed to get instances from cloud provider")
		}
		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
		nodeAddresses, err := instances.NodeAddresses(kl.nodeName)
		if err != nil {
			return fmt.Errorf("failed to get node address from cloud provider: %v", err)
		}
		node.Status.Addresses = nodeAddresses
	} else {
		if kl.nodeIP != nil {
			node.Status.Addresses = []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: kl.nodeIP.String()},
				{Type: api.NodeInternalIP, Address: kl.nodeIP.String()},
			}
		} else if addr := net.ParseIP(kl.hostname); addr != nil {
			node.Status.Addresses = []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: addr.String()},
				{Type: api.NodeInternalIP, Address: addr.String()},
			}
		} else {
			addrs, err := net.LookupIP(node.Name)
			if err != nil {
				return fmt.Errorf("can't get ip address of node %s: %v", node.Name, err)
			} else if len(addrs) == 0 {
				return fmt.Errorf("no ip address for node %v", node.Name)
			} else {
				// check all ip addresses for this node.Name and try to find the first non-loopback IPv4 address.
				// If no match is found, it uses the IP of the interface with gateway on it.
				for _, ip := range addrs {
					if ip.IsLoopback() {
						continue
					}

					if ip.To4() != nil {
						node.Status.Addresses = []api.NodeAddress{
							{Type: api.NodeLegacyHostIP, Address: ip.String()},
							{Type: api.NodeInternalIP, Address: ip.String()},
						}
						break
					}
				}

				if len(node.Status.Addresses) == 0 {
					ip, err := utilnet.ChooseHostInterface()
					if err != nil {
						return err
					}

					node.Status.Addresses = []api.NodeAddress{
						{Type: api.NodeLegacyHostIP, Address: ip.String()},
						{Type: api.NodeInternalIP, Address: ip.String()},
					}
				}
			}
		}
	}
	return nil
}

// updatePodCIDR updates the pod CIDR in the runtime state if it is different
// from the current CIDR.
func (kl *Kubelet) updatePodCIDR(cidr string) {
	if kl.runtimeState.podCIDR() == cidr {
		return
	}

	glog.Infof("Setting Pod CIDR: %v -> %v", kl.runtimeState.podCIDR(), cidr)
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
