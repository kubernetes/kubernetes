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

package dns

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"

	"github.com/golang/glog"
)

// Configurer is used for setting up DNS resolver configuration when launching pods.
type Configurer struct {
	recorder record.EventRecorder
	nodeRef  *v1.ObjectReference
	nodeIP   net.IP

	// If non-nil, use this for container DNS server.
	clusterDNS []net.IP
	// If non-empty, use this for container DNS search.
	ClusterDomain string
	// The path to the DNS resolver configuration file used as the base to generate
	// the container's DNS resolver configuration file. This can be used in
	// conjunction with clusterDomain and clusterDNS.
	ResolverConfig string
}

// NewConfigurer returns a DNS configurer for launching pods.
func NewConfigurer(recorder record.EventRecorder, nodeRef *v1.ObjectReference, nodeIP net.IP, clusterDNS []net.IP, clusterDomain, resolverConfig string) *Configurer {
	return &Configurer{
		recorder:       recorder,
		nodeRef:        nodeRef,
		nodeIP:         nodeIP,
		clusterDNS:     clusterDNS,
		ClusterDomain:  clusterDomain,
		ResolverConfig: resolverConfig,
	}
}

func omitDuplicates(pod *v1.Pod, combinedSearch []string) []string {
	uniqueDomains := map[string]bool{}

	for _, dnsDomain := range combinedSearch {
		if _, exists := uniqueDomains[dnsDomain]; !exists {
			combinedSearch[len(uniqueDomains)] = dnsDomain
			uniqueDomains[dnsDomain] = true
		}
	}
	return combinedSearch[:len(uniqueDomains)]
}

func (c *Configurer) formDNSSearchFitsLimits(pod *v1.Pod, composedSearch []string) []string {
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
		c.recorder.Event(pod, v1.EventTypeWarning, "DNSSearchForming", log)
		glog.Error(log)
	}
	return composedSearch
}

func (c *Configurer) formDNSSearchForDNSDefault(hostSearch []string, pod *v1.Pod) []string {
	return c.formDNSSearchFitsLimits(pod, hostSearch)
}

func (c *Configurer) formDNSSearch(hostSearch []string, pod *v1.Pod) []string {
	if c.ClusterDomain == "" {
		c.formDNSSearchFitsLimits(pod, hostSearch)
		return hostSearch
	}

	nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, c.ClusterDomain)
	svcDomain := fmt.Sprintf("svc.%s", c.ClusterDomain)
	dnsSearch := []string{nsSvcDomain, svcDomain, c.ClusterDomain}

	combinedSearch := append(dnsSearch, hostSearch...)

	combinedSearch = omitDuplicates(pod, combinedSearch)
	return c.formDNSSearchFitsLimits(pod, combinedSearch)
}

// CheckLimitsForResolvConf checks limits in resolv.conf.
func (c *Configurer) CheckLimitsForResolvConf() {
	// resolver file Search line current limitations
	resolvSearchLineDNSDomainsLimit := 6
	resolvSearchLineLenLimit := 255

	f, err := os.Open(c.ResolverConfig)
	if err != nil {
		c.recorder.Event(c.nodeRef, v1.EventTypeWarning, "CheckLimitsForResolvConf", err.Error())
		glog.Error("CheckLimitsForResolvConf: " + err.Error())
		return
	}
	defer f.Close()

	_, hostSearch, _, err := parseResolvConf(f)
	if err != nil {
		c.recorder.Event(c.nodeRef, v1.EventTypeWarning, "CheckLimitsForResolvConf", err.Error())
		glog.Error("CheckLimitsForResolvConf: " + err.Error())
		return
	}

	domainCntLimit := resolvSearchLineDNSDomainsLimit

	if c.ClusterDomain != "" {
		domainCntLimit -= 3
	}

	if len(hostSearch) > domainCntLimit {
		log := fmt.Sprintf("Resolv.conf file '%s' contains search line consisting of more than %d domains!", c.ResolverConfig, domainCntLimit)
		c.recorder.Event(c.nodeRef, v1.EventTypeWarning, "CheckLimitsForResolvConf", log)
		glog.Error("CheckLimitsForResolvConf: " + log)
		return
	}

	if len(strings.Join(hostSearch, " ")) > resolvSearchLineLenLimit {
		log := fmt.Sprintf("Resolv.conf file '%s' contains search line which length is more than allowed %d chars!", c.ResolverConfig, resolvSearchLineLenLimit)
		c.recorder.Event(c.nodeRef, v1.EventTypeWarning, "CheckLimitsForResolvConf", log)
		glog.Error("CheckLimitsForResolvConf: " + log)
		return
	}

	return
}

// parseResolveConf reads a resolv.conf file from the given reader, and parses
// it into nameservers, searches and options, possibly returning an error.
// TODO: move to utility package
func parseResolvConf(reader io.Reader) (nameservers []string, searches []string, options []string, err error) {
	file, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, nil, nil, err
	}

	// Lines of the form "nameserver 1.2.3.4" accumulate.
	nameservers = []string{}

	// Lines of the form "search example.com" overrule - last one wins.
	searches = []string{}

	// Lines of the form "option ndots:5 attempts:2" overrule - last one wins.
	// Each option is recorded as an element in the array.
	options = []string{}

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
		if fields[0] == "options" {
			options = fields[1:]
		}
	}

	// There used to be code here to scrub DNS for each cloud, but doesn't
	// make sense anymore since cloudproviders are being factored out.
	// contact @thockin or @wlan0 for more information

	return nameservers, searches, options, nil
}

// GetClusterDNS returns a list of the DNS servers, a list of the DNS search
// domains of the cluster, and a list of resolv.conf options.
// TODO: This should return a struct.
func (c *Configurer) GetClusterDNS(pod *v1.Pod) ([]string, []string, []string, bool, error) {
	var hostDNS, hostSearch, hostOptions []string
	// Get host DNS settings
	if c.ResolverConfig != "" {
		f, err := os.Open(c.ResolverConfig)
		if err != nil {
			return nil, nil, nil, false, err
		}
		defer f.Close()

		hostDNS, hostSearch, hostOptions, err = parseResolvConf(f)
		if err != nil {
			return nil, nil, nil, false, err
		}
	}
	useClusterFirstPolicy := ((pod.Spec.DNSPolicy == v1.DNSClusterFirst && !kubecontainer.IsHostNetworkPod(pod)) || pod.Spec.DNSPolicy == v1.DNSClusterFirstWithHostNet)
	if useClusterFirstPolicy && len(c.clusterDNS) == 0 {
		// clusterDNS is not known.
		// pod with ClusterDNSFirst Policy cannot be created
		c.recorder.Eventf(pod, v1.EventTypeWarning, "MissingClusterDNS", "kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy)
		log := fmt.Sprintf("kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. pod: %q. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy, format.Pod(pod))
		c.recorder.Eventf(c.nodeRef, v1.EventTypeWarning, "MissingClusterDNS", log)

		// fallback to DNSDefault
		useClusterFirstPolicy = false
	}

	if !useClusterFirstPolicy {
		// When the kubelet --resolv-conf flag is set to the empty string, use
		// DNS settings that override the docker default (which is to use
		// /etc/resolv.conf) and effectively disable DNS lookups. According to
		// the bind documentation, the behavior of the DNS client library when
		// "nameservers" are not specified is to "use the nameserver on the
		// local machine". A nameserver setting of localhost is equivalent to
		// this documented behavior.
		if c.ResolverConfig == "" {
			hostSearch = []string{"."}
			switch {
			case c.nodeIP == nil || c.nodeIP.To4() != nil:
				hostDNS = []string{"127.0.0.1"}
			case c.nodeIP.To16() != nil:
				hostDNS = []string{"::1"}
			}
		} else {
			hostSearch = c.formDNSSearchForDNSDefault(hostSearch, pod)
		}
		return hostDNS, hostSearch, hostOptions, useClusterFirstPolicy, nil
	}

	// for a pod with DNSClusterFirst policy, the cluster DNS server is the only nameserver configured for
	// the pod. The cluster DNS server itself will forward queries to other nameservers that is configured to use,
	// in case the cluster DNS server cannot resolve the DNS query itself
	dns := make([]string, len(c.clusterDNS))
	for i, ip := range c.clusterDNS {
		dns[i] = ip.String()
	}
	dnsSearch := c.formDNSSearch(hostSearch, pod)

	return dns, dnsSearch, hostOptions, useClusterFirstPolicy, nil
}

// SetupDNSinContainerizedMounter replaces the nameserver in containerized-mounter's rootfs/etc/resolve.conf with kubelet.ClusterDNS
func (c *Configurer) SetupDNSinContainerizedMounter(mounterPath string) {
	resolvePath := filepath.Join(strings.TrimSuffix(mounterPath, "/mounter"), "rootfs", "etc", "resolv.conf")
	dnsString := ""
	for _, dns := range c.clusterDNS {
		dnsString = dnsString + fmt.Sprintf("nameserver %s\n", dns)
	}
	if c.ResolverConfig != "" {
		f, err := os.Open(c.ResolverConfig)
		defer f.Close()
		if err != nil {
			glog.Error("Could not open resolverConf file")
		} else {
			_, hostSearch, _, err := parseResolvConf(f)
			if err != nil {
				glog.Errorf("Error for parsing the reslov.conf file: %v", err)
			} else {
				dnsString = dnsString + "search"
				for _, search := range hostSearch {
					dnsString = dnsString + fmt.Sprintf(" %s", search)
				}
				dnsString = dnsString + "\n"
			}
		}
	}
	if err := ioutil.WriteFile(resolvePath, []byte(dnsString), 0600); err != nil {
		glog.Errorf("Could not write dns nameserver in file %s, with error %v", resolvePath, err)
	}
}
