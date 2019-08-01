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

package ipvs

import (
	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	"net"

	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/apimachinery/pkg/runtime"
	"fmt"
	"k8s.io/klog"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	"k8s.io/utils/exec"
	utilnode "k8s.io/kubernetes/pkg/util/node"

)

type MetaProxier struct {
	ipv4Proxier *Proxier
	ipv6Proxier *Proxier
}

type ipFamily int
const (
	familyUnknown = iota + 1
	familyIpv4
	familyIpv6
)

// NewMetaProxier returns a "meta-proxier" which will create two ipvs
// proxier instances, one for ipv4 and one for ipv6. Proxier API calls
// will be dispatched to ipvs-proxier instances according to address
// family.
func NewMetaProxier(
	mainProxier *Proxier,
	config *proxyconfigapi.KubeProxyConfiguration,
	cleanupAndExit bool,
	scheme *runtime.Scheme,
	master string,
) (*MetaProxier, error) {
	ipv6Proxier, err := newOtherProxyServer(config, cleanupAndExit, scheme, master)
	if err != nil {
		return nil, err
	}
	return &MetaProxier{
		ipv4Proxier: mainProxier,
		ipv6Proxier: ipv6Proxier,
	}, nil
}

func newOtherProxyServer(
	config *proxyconfigapi.KubeProxyConfiguration,
	cleanupAndExit bool,
	scheme *runtime.Scheme,
	master string) (*Proxier, error) {

	// TODO: Remove hard-coded ipv6
	nodeIP := net.ParseIP("::1")
	protocol := utiliptables.ProtocolIpv6
	clusterCIDR := "1100::/16"

	var iptInterface utiliptables.Interface
	var ipvsInterface utilipvs.Interface
	var ipsetInterface utilipset.Interface
	var dbus utildbus.Interface

	// Create a iptables utils.
	execer := exec.New()

	dbus = utildbus.New()
	iptInterface = utiliptables.New(execer, dbus, protocol)
	ipsetInterface = utilipset.New(execer)
	ipvsInterface = utilipvs.New(execer)

	// Create event recorder
	hostname, err := utilnode.GetHostname(config.HostnameOverride)
	if err != nil {
		return nil, err
	}
	eventBroadcaster := record.NewBroadcaster()
	_ = eventBroadcaster.NewRecorder(scheme, v1.EventSource{Component: "kube-proxy-6", Host: hostname})


	klog.V(0).Info("Using ipvs MetaProxier.")
	proxier, err := NewProxier(
		iptInterface,
		ipvsInterface,
		ipsetInterface,
		utilsysctl.New(),
		execer,
		config.IPVS.SyncPeriod.Duration,
		config.IPVS.MinSyncPeriod.Duration,
		config.IPVS.ExcludeCIDRs,
		config.IPVS.StrictARP,
		config.IPTables.MasqueradeAll,
		int(*config.IPTables.MasqueradeBit),
		clusterCIDR,
		hostname,
		nodeIP,
		nil, // recorder
		nil, // healthzServer
		config.IPVS.Scheduler,
		config.NodePortAddresses,
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create ipv6 proxier: %v", err)
	}
	return proxier, nil
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules.
func (proxier *MetaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
	proxier.ipv6Proxier.Sync()
}

// SyncLoop runs periodic work.  This is expected to run as a
// goroutine or as the main loop of the app.  It does not return.
func (proxier *MetaProxier) SyncLoop() {
	go proxier.ipv6Proxier.SyncLoop() // Use go-routine here!
	proxier.ipv4Proxier.SyncLoop()    // never returns
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *MetaProxier) OnServiceAdd(service *v1.Service) {
	family := serviceFamily(service)
	klog.Infof("ServiceAdd. family = %d", family)
	switch family {
	case familyIpv4:
		proxier.ipv4Proxier.OnServiceAdd(service)
	case familyIpv6:
		proxier.ipv6Proxier.OnServiceAdd(service)
	}
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *MetaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	family := serviceFamily(service)
	if family == familyUnknown {
		family = serviceFamily(oldService)
	}
	klog.Infof("ServiceUpdate. family = %d", family)
	switch family {
	case familyIpv4:
		proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
	case familyIpv6:
		proxier.ipv6Proxier.OnServiceUpdate(oldService, service)
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *MetaProxier) OnServiceDelete(service *v1.Service) {
	switch serviceFamily(service) {
	case familyIpv4:
		proxier.ipv4Proxier.OnServiceDelete(service)
	case familyIpv6:
		proxier.ipv6Proxier.OnServiceDelete(service)
	}
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
	proxier.ipv6Proxier.OnServiceSynced()
}

func serviceFamily(service *v1.Service) ipFamily {
	if service.Spec.ClusterIP == "" {
		return familyUnknown
	}
	if net.ParseIP(service.Spec.ClusterIP).To4() == nil {
		return familyIpv6
	}
	return familyIpv4
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *MetaProxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	switch endpointsFamily(endpoints) {
	case familyIpv4:
		proxier.ipv4Proxier.OnEndpointsAdd(endpoints)
	case familyIpv6:
		proxier.ipv6Proxier.OnEndpointsAdd(endpoints)
	}
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	family := endpointsFamily(endpoints)
	if family == familyUnknown {
		family = endpointsFamily(oldEndpoints)
	}
	switch family {
	case familyIpv4:
		proxier.ipv4Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
	case familyIpv6:
		proxier.ipv6Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
	}
}

// OnEndpointsDelete is called whenever deletion of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	switch endpointsFamily(endpoints) {
	case familyIpv4:
		proxier.ipv4Proxier.OnEndpointsDelete(endpoints)
	case familyIpv6:
		proxier.ipv6Proxier.OnEndpointsDelete(endpoints)
	}
}

// OnEndpointsSynced is called once all the initial event handlers
// were called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnEndpointsSynced() {
	proxier.ipv4Proxier.OnEndpointsSynced()
	proxier.ipv6Proxier.OnEndpointsSynced()
}

func endpointsFamily(endpoints *v1.Endpoints) ipFamily {
	if endpoints.Subsets == nil || len(endpoints.Subsets) == 0 {
		klog.Warningf("Endpoints familyUnknown. No subsets")
		return familyUnknown
	}
	for _, subset := range endpoints.Subsets {
		if family := endpointsSubsetFamily(subset); family != familyUnknown {
			return family
		}
	}
	klog.Warningf("Endpoints familyUnknown. No address match")
	return familyUnknown
}
func endpointsSubsetFamily(subsets v1.EndpointSubset) ipFamily {
	if family := endpointsAddressesFamily(subsets.Addresses); family != familyUnknown {
		return family
	}
	klog.Infof("Endpoints testing NotReadyAddresses")
	return endpointsAddressesFamily(subsets.NotReadyAddresses)
}
func endpointsAddressesFamily(addresses []v1.EndpointAddress) ipFamily {
	if addresses == nil || len(addresses) == 0 {
		return familyUnknown
	}
	for _, adr := range addresses {
		if adr.IP != "" {
			if net.ParseIP(adr.IP).To4() == nil {
				return familyIpv6
			} else {
				return familyIpv4
			}
		}
	}
	return familyUnknown
}
