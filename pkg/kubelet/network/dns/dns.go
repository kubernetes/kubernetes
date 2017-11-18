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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/features"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"

	"github.com/golang/glog"
)

var (
	// The default dns opt strings.
	defaultDNSOptions = []string{"ndots:5"}
)

const (
	resolvNameserversLimit          = 3
	resolvSearchLineDNSDomainsLimit = 6
	resolvSearchLineLenLimit        = 256
)

type podDNSType int

const (
	podDNSCluster podDNSType = 0
	podDNSHost    podDNSType = 1
	podDNSCustom  podDNSType = 2
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

func omitDuplicates(strs []string) []string {
	uniqueStr := make(map[string]bool)

	for _, str := range strs {
		if exists := uniqueStr[str]; !exists {
			strs[len(uniqueStr)] = str
			uniqueStr[str] = true
		}
	}
	return strs[:len(uniqueStr)]
}

func (c *Configurer) formDNSSearchFitsLimits(composedSearch []string, pod *v1.Pod) []string {
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
		log := fmt.Sprintf("Search Line limits were exceeded, some search paths have been omitted, the applied search line is: %s", strings.Join(composedSearch, " "))
		c.recorder.Event(pod, v1.EventTypeWarning, "DNSConfigForming", log)
		glog.Error(log)
	}
	return composedSearch
}

func (c *Configurer) formDNSNameserversFitsLimits(nameservers []string, pod *v1.Pod) []string {
	if len(nameservers) > resolvNameserversLimit {
		nameservers = nameservers[0:resolvNameserversLimit]
		log := fmt.Sprintf("Nameserver limits were exceeded, some nameservers have been omitted, the applied nameserver line is: %s", strings.Join(nameservers, " "))
		c.recorder.Event(pod, v1.EventTypeWarning, "DNSConfigForming", log)
		glog.Error(log)
	}
	return nameservers
}

func (c *Configurer) formDNSConfigFitsLimits(dnsConfig *runtimeapi.DNSConfig, pod *v1.Pod) *runtimeapi.DNSConfig {
	dnsConfig.Servers = c.formDNSNameserversFitsLimits(dnsConfig.Servers, pod)
	dnsConfig.Searches = c.formDNSSearchFitsLimits(dnsConfig.Searches, pod)
	return dnsConfig
}

func (c *Configurer) generateSearchesForDNSClusterFirst(hostSearch []string, pod *v1.Pod) []string {
	if c.ClusterDomain == "" {
		return hostSearch
	}

	nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, c.ClusterDomain)
	svcDomain := fmt.Sprintf("svc.%s", c.ClusterDomain)
	clusterSearch := []string{nsSvcDomain, svcDomain, c.ClusterDomain}

	return omitDuplicates(append(clusterSearch, hostSearch...))
}

// CheckLimitsForResolvConf checks limits in resolv.conf.
func (c *Configurer) CheckLimitsForResolvConf() {
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

	return nameservers, searches, options, nil
}

func (c *Configurer) getHostDNSConfig(pod *v1.Pod) (*runtimeapi.DNSConfig, error) {
	var hostDNS, hostSearch, hostOptions []string
	// Get host DNS settings
	if c.ResolverConfig != "" {
		f, err := os.Open(c.ResolverConfig)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		hostDNS, hostSearch, hostOptions, err = parseResolvConf(f)
		if err != nil {
			return nil, err
		}
	}
	return &runtimeapi.DNSConfig{
		Servers:  hostDNS,
		Searches: hostSearch,
		Options:  hostOptions,
	}, nil
}

func (c *Configurer) getPodDNSType(pod *v1.Pod) podDNSType {
	dnsPolicy := pod.Spec.DNSPolicy
	if dnsPolicy == v1.DNSCustom {
		if utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) {
			return podDNSCustom
		}
		// Fallback to DNSDefault. This should not happen as kube-apiserver should
		// have rejected setting dnsPolicy to DNSCustom when feature gate is disabled.
		return podDNSHost
	}

	// Only attempt to use ClusterFirst Policy when:
	// 1. dnsPolicy set to DNSClusterFirst on non-hostnetwork pod, or
	// 2. dnsPolicy set to DNSClusterFirstWithHostNet.
	// This behavior is defined by API.
	useClusterFirstPolicy := ((dnsPolicy == v1.DNSClusterFirst && !kubecontainer.IsHostNetworkPod(pod)) || dnsPolicy == v1.DNSClusterFirstWithHostNet)
	if useClusterFirstPolicy {
		if len(c.clusterDNS) != 0 {
			return podDNSCluster
		}
		// clusterDNS is not known.
		// Pod with ClusterDNSFirst Policy cannot be created.
		c.recorder.Eventf(pod, v1.EventTypeWarning, "MissingClusterDNS", "kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. Falling back to DNSDefault policy.", dnsPolicy)
		log := fmt.Sprintf("kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. pod: %q. Falling back to DNSDefault policy.", dnsPolicy, format.Pod(pod))
		c.recorder.Eventf(c.nodeRef, v1.EventTypeWarning, "MissingClusterDNS", log)
		// Fallback to DNSDefault.
	}
	return podDNSHost
}

// Merge DNS options. If duplicated, entries given by DNSParams will overwrite
// the existing ones.
func mergeDNSOptions(dnsConfigOptions []string, dnsParamsOptions []v1.PodDNSParamsOption) []string {
	optionsMap := make(map[string]string)
	for _, op := range dnsConfigOptions {
		var opName, opValue string
		if index := strings.Index(op, ":"); index != -1 {
			opName = op[:index]
			opValue = op[index+1:]
		} else {
			opName = op
		}
		optionsMap[opName] = opValue
	}
	for _, op := range dnsParamsOptions {
		if op.Value != nil {
			optionsMap[op.Name] = *op.Value
		} else {
			optionsMap[op.Name] = ""
		}
	}
	// Reconvert DNS options into a string array.
	options := []string{}
	for opName, opValue := range optionsMap {
		op := opName
		if len(opValue) > 0 {
			op = op + ":" + opValue
		}
		options = append(options, op)
	}
	return options
}

// appendDNSParams appends DNS servers, search paths and options given by DNSParams
// to the existing DNS config. Duplicated entries will be merged.
// This assume dnsConfig and dnsParams are not nil.
func appendDNSParams(dnsConfig *runtimeapi.DNSConfig, dnsParams *v1.PodDNSParams) *runtimeapi.DNSConfig {
	dnsConfig.Servers = omitDuplicates(append(dnsConfig.Servers, dnsParams.Nameservers...))
	dnsConfig.Searches = omitDuplicates(append(dnsConfig.Searches, dnsParams.Searches...))
	dnsConfig.Options = mergeDNSOptions(dnsConfig.Options, dnsParams.Options)
	return dnsConfig
}

// GetPodDNS returns DNS setttings for the pod.
func (c *Configurer) GetPodDNS(pod *v1.Pod) (*runtimeapi.DNSConfig, error) {
	dnsConfig, err := c.getHostDNSConfig(pod)
	if err != nil {
		return nil, err
	}

	dnsType := c.getPodDNSType(pod)
	switch dnsType {
	case podDNSHost:
		// When the kubelet --resolv-conf flag is set to the empty string, use
		// DNS settings that override the docker default (which is to use
		// /etc/resolv.conf) and effectively disable DNS lookups. According to
		// the bind documentation, the behavior of the DNS client library when
		// "nameservers" are not specified is to "use the nameserver on the
		// local machine". A nameserver setting of localhost is equivalent to
		// this documented behavior.
		if c.ResolverConfig == "" {
			switch {
			case c.nodeIP == nil || c.nodeIP.To4() != nil:
				dnsConfig.Servers = []string{"127.0.0.1"}
			case c.nodeIP.To16() != nil:
				dnsConfig.Servers = []string{"::1"}
			}
			dnsConfig.Searches = []string{"."}
		}
	case podDNSCluster:
		// for a pod with DNSClusterFirst policy, the cluster DNS server is the only nameserver configured for
		// the pod. The cluster DNS server itself will forward queries to other nameservers that is configured to use,
		// in case the cluster DNS server cannot resolve the DNS query itself
		dnsConfig.Servers = []string{}
		for _, ip := range c.clusterDNS {
			dnsConfig.Servers = append(dnsConfig.Servers, ip.String())
		}
		dnsConfig.Searches = c.generateSearchesForDNSClusterFirst(dnsConfig.Searches, pod)
		dnsConfig.Options = defaultDNSOptions
	case podDNSCustom:
		// DNSCustom should use an empty DNS settings as the base.
		dnsConfig = &runtimeapi.DNSConfig{}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CustomPodDNS) && pod.Spec.DNSParams != nil {
		dnsConfig = appendDNSParams(dnsConfig, pod.Spec.DNSParams)
	}
	return c.formDNSConfigFitsLimits(dnsConfig, pod), nil
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
