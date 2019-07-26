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
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
	"net"
	"time"
)

type MetaProxier struct {
	ipv4Proxier *Proxier
	ipv6Proxier *Proxier
}

// NewMetaProxier returns a "meta-proxier" which will create two ipvs
// proxier instances, one for ipv4 and one for ipv6. Proxier API calls
// will be dispatched to ipvs-proxier instances according to address
// family.
func NewMetaProxier(
	ipt utiliptables.Interface,
	ipvs utilipvs.Interface,
	ipset utilipset.Interface,
	sysctl utilsysctl.Interface,
	exec utilexec.Interface,
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	excludeCIDRs []string,
	strictARP bool,
	masqueradeAll bool,
	masqueradeBit int,
	clusterCIDR string,
	hostname string,
	nodeIP net.IP,
	recorder record.EventRecorder,
	healthzServer healthcheck.HealthzUpdater,
	scheduler string,
	nodePortAddresses []string,
) (*MetaProxier, error) {

	ipv4Proxier, err := NewProxier(
		ipt, ipvs, ipset, sysctl, exec, syncPeriod, minSyncPeriod,
		excludeCIDRs, strictARP, masqueradeAll, masqueradeBit,
		clusterCIDR, hostname, nodeIP, recorder, healthzServer,
		scheduler, nodePortAddresses)
	if err != nil {
		return nil, err
	}

	return &MetaProxier{
		ipv4Proxier: ipv4Proxier,
	}, nil
}

// Sync immediately synchronizes the ProxyProvider's current state to
// proxy rules.
func (proxier *MetaProxier) Sync() {
	proxier.ipv4Proxier.Sync()
}

// SyncLoop runs periodic work.  This is expected to run as a
// goroutine or as the main loop of the app.  It does not return.
func (proxier *MetaProxier) SyncLoop() {
	proxier.ipv4Proxier.SyncLoop()
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *MetaProxier) OnServiceAdd(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceAdd(service)
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *MetaProxier) OnServiceUpdate(oldService, service *v1.Service) {
	proxier.ipv4Proxier.OnServiceUpdate(oldService, service)
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *MetaProxier) OnServiceDelete(service *v1.Service) {
	proxier.ipv4Proxier.OnServiceDelete(service)
}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnServiceSynced() {
	proxier.ipv4Proxier.OnServiceSynced()
}

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *MetaProxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	proxier.ipv4Proxier.OnEndpointsAdd(endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	proxier.ipv4Proxier.OnEndpointsUpdate(oldEndpoints, endpoints)
}

// OnEndpointsDelete is called whenever deletion of an existing
// endpoints object is observed.
func (proxier *MetaProxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	proxier.ipv4Proxier.OnEndpointsDelete(endpoints)
}

// OnEndpointsSynced is called once all the initial event handlers
// were called and the state is fully propagated to local cache.
func (proxier *MetaProxier) OnEndpointsSynced() {
	proxier.ipv4Proxier.OnEndpointsSynced()
}
