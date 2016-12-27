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

package dns

/*
Quinton: Pseudocode....
See https://github.com/kubernetes/kubernetes/pull/25107#issuecomment-218026648
For each service we need the following DNS names:
mysvc.myns.myfed.svc.z1.r1.mydomain.com  (for zone z1 in region r1)
 - an A record to IP address of specific shard in that zone (if that shard exists and has healthy endpoints)
 - OR a CNAME record to the next level up, i.e. mysvc.myns.myfed.svc.r1.mydomain.com  (if a healthy shard does not
 exist in zone z1)
mysvc.myns.myfed.svc.r1.federation
 - a set of A records to IP addresses of all healthy shards in region r1, if one or more of these exist
 - OR a CNAME record to the next level up, i.e. mysvc.myns.myfed.svc.mydomain.com (if no healthy shards exist in
 region r1)
mysvc.myns.myfed.svc.federation
 - a set of A records to IP addresses of all healthy shards in all regions, if one or more of these exist.
 - no record (NXRECORD response) if no healthy shards exist in any regions)

For each cached service, cachedService.lastState tracks the current known state of the service, while
cachedService.appliedState contains
the state of the service when we last successfully sync'd it's DNS records.
So this time around we only need to patch that (add new records, remove deleted records, and update changed records.
*/

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/golang/glog"

	fed "k8s.io/kubernetes/federation/apis/federation"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
	servicecontroller "k8s.io/kubernetes/federation/pkg/federation-controller/service"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	ControllerName = "dns"

	// minDnsTtl is the minimum safe DNS TTL value to use (in seconds).  We use this as the TTL for all DNS records.
	minDNSTTL = 180
)

var (
	RequiredResources = []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource("services")}
)

type DNSController struct {
	dns            dnsprovider.Interface
	federationName string
	// serviceDNSSuffix is the DNS suffix we use when publishing service DNS names
	serviceDNSSuffix string
	// zoneName and zoneID are used to identify the zone in which to put records
	zoneName string
	zoneID   string
	// Each federation should be configured with a single zone (e.g. "mycompany.com")
	dnsZones dnsprovider.Zones
	dnsZone  dnsprovider.Zone
	// Informer Store for federated services.
	serviceInformerStore cache.StoreToReplicaSetLister
	// Informer controller for federated services.
	serviceInformerController cache.ControllerInterface
	// Client to federated api server.
	federationClient fedclientset.Interface
	workQueue        workqueue.Interface
}

// NewDNSController returns a new dns controller
func NewDNSController(client fedclientset.Interface, dnsProvider, dnsProviderConfig, federationName,
	serviceDNSSuffix, zoneName, zoneID string) *DNSController {
	dns, err := dnsprovider.InitDnsProvider(dnsProvider, dnsProviderConfig)
	if err != nil {
		glog.Fatalf("DNS provider could not be initialized: %v", err)
		return nil
	}

	dc := &DNSController{
		federationClient: client,
		dns:              dns,
		federationName:   federationName,
		serviceDNSSuffix: serviceDNSSuffix,
		zoneName:         zoneName,
		zoneID:           zoneID,
		workQueue:        workqueue.New(),
	}

	// Start informer in federated API servers on services that should be federated.
	dc.serviceInformerStore.Indexer, dc.serviceInformerController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
				return client.Core().Services(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return client.Core().Services(apiv1.NamespaceAll).Watch(options)
			},
		},
		&apiv1.Service{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj runtime.Object) { dc.workQueue.Add(obj) }),
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	// Check configuration
	if err := dc.init(); err != nil {
		glog.Fatalf("DNS provider failed to initialize: %v", err)
		return nil
	}

	return dc
}

// Check and initialize configuration
func (dc *DNSController) init() error {
	if dc.federationName == "" {
		return fmt.Errorf("DnsController should not be run without federationName.")
	}
	if dc.zoneName == "" && dc.zoneID == "" {
		return fmt.Errorf("DnsController must be run with either zoneName or zoneID.")
	}
	if dc.serviceDNSSuffix == "" {
		if dc.zoneName == "" {
			return fmt.Errorf("DnsController must be run with zoneName, if serviceDnsSuffix is not set.")
		}
		dc.serviceDNSSuffix = dc.zoneName
	}
	if dc.dns == nil {
		return fmt.Errorf("DnsController should not be run without a dnsprovider.")
	}
	zones, ok := dc.dns.Zones()
	if !ok {
		return fmt.Errorf("the dns provider does not support zone enumeration, " +
			"which is required for creating dns records")
	}
	dc.dnsZones = zones
	matchingZones, err := getDNSZones(dc.zoneName, dc.zoneID, dc.dnsZones)
	if err != nil {
		return fmt.Errorf("error querying for DNS zones: %v", err)
	}
	if len(matchingZones) == 0 {
		if dc.zoneName == "" {
			return fmt.Errorf("DnsController must be run with zoneName to create zone automatically.")
		}
		glog.Infof("DNS zone %q not found.  Creating DNS zone %q.", dc.zoneName, dc.zoneName)
		managedZone, err := dc.dnsZones.New(dc.zoneName)
		if err != nil {
			return err
		}
		zone, err := dc.dnsZones.Add(managedZone)
		if err != nil {
			return err
		}
		glog.Infof("DNS zone %q successfully created.  Note that DNS resolution will not work until you have "+
			"registered this name with a DNS registrar and they have changed the authoritative name "+
			"servers for your domain to point to your DNS provider.", zone.Name())
	}
	if len(matchingZones) > 1 {
		return fmt.Errorf("Multiple matching DNS zones found for %q; please specify zoneID", dc.zoneName)
	}
	dc.dnsZone, err = getDNSZone(dc.zoneName, dc.zoneID, dc.dnsZones)
	if err != nil {
		return fmt.Errorf("Get DNS Zone failed: %v", err)
	}
	return nil
}

func (dc *DNSController) Run(workers int, stopChan <-chan struct{}) {
	for i := 0; i < workers; i++ {
		go wait.Until(dc.worker, time.Second, stopChan)
	}
	go dc.serviceInformerController.Run(stopChan)
}

func (sc *DNSController) worker() {
	for {
		item, quit := sc.workQueue.Get()
		if quit {
			return
		}
		service := item.(*apiv1.Service)
		sc.reconcileDNS(service)
		sc.workQueue.Done(item)
	}
}

// getUplevelCName returns one level higher CNAME for a dnsName
// if dnsName is of zone level, then returns region CNAME
// if dnsName is of region level, then returns global CNAME
// if dnsName is of global level, then returns "" empty string
func (dc *DNSController) getUplevelCName(dnsName, prefix string) (uplevelCname string) {
	prefixSlice := strings.Split(prefix, ".")
	dnsNameSlice := strings.Split(dnsName, ".")
	itemToDelete := len(prefixSlice)
	dnsNameSlice = append(dnsNameSlice[:itemToDelete], dnsNameSlice[itemToDelete+1:]...)

	uplevelCname = strings.Join(dnsNameSlice, ".")
	if !strings.HasSuffix(uplevelCname, dc.serviceDNSSuffix) {
		uplevelCname = ""
	}
	return uplevelCname
}

// getDNSZones returns the DNS zones matching dnsZoneName and dnsZoneID (if specified)
func getDNSZones(dnsZoneName string, dnsZoneID string, dnsZonesInterface dnsprovider.Zones) (
	[]dnsprovider.Zone, error) {
	// TODO: We need query-by-name and query-by-id functions
	dnsZones, err := dnsZonesInterface.List()
	if err != nil {
		return nil, err
	}

	var matches []dnsprovider.Zone
	findName := strings.TrimSuffix(dnsZoneName, ".")
	for _, dnsZone := range dnsZones {
		if dnsZoneID != "" {
			if dnsZoneID != dnsZone.ID() {
				continue
			}
		}
		if findName != "" {
			if strings.TrimSuffix(dnsZone.Name(), ".") != findName {
				continue
			}
		}
		matches = append(matches, dnsZone)
	}

	return matches, nil
}

// getDNSZone returns the DNS zone, as identified by dnsZoneName and dnsZoneID
// This is similar to getDnsZones, but returns an error if there are zero or multiple matching zones.
func getDNSZone(dnsZoneName string, dnsZoneID string, dnsZonesInterface dnsprovider.Zones) (dnsprovider.Zone, error) {
	dnsZones, err := getDNSZones(dnsZoneName, dnsZoneID, dnsZonesInterface)
	if err != nil {
		return nil, err
	}

	if len(dnsZones) == 1 {
		return dnsZones[0], nil
	}

	name := dnsZoneName
	if dnsZoneID != "" {
		name += "/" + dnsZoneID
	}

	if len(dnsZones) == 0 {
		return nil, fmt.Errorf("DNS zone %s not found", name)
	}

	return nil, fmt.Errorf("DNS zone %s is ambiguous (please specify zoneID)", name)
}

// getResolvedEndpoints performs DNS resolution on the provided slice of endpoints (which might be DNS names or
// IPv4 addresses) and returns a list of IPv4 addresses.  If any of the endpoints are neither valid IPv4 addresses nor
// resolvable DNS names, non-nil error is also returned (possibly along with a partially complete list of resolved
// endpoints).
func getResolvedEndpoints(endpoints []string) ([]string, error) {
	resolvedEndpoints := make([]string, 0, len(endpoints))
	for _, endpoint := range endpoints {
		if net.ParseIP(endpoint) == nil {
			// It's not a valid IP address, so assume it's a DNS name, and try to resolve it,
			// replacing its DNS name with its IP addresses in expandedEndpoints
			ipAddrs, err := net.LookupHost(endpoint)
			if err != nil {
				return resolvedEndpoints, err
			}
			resolvedEndpoints = append(resolvedEndpoints, ipAddrs...)

		} else {
			resolvedEndpoints = append(resolvedEndpoints, endpoint)
		}
	}
	return resolvedEndpoints, nil
}

// ensureDNSRrsets ensures (idempotently, and with minimum mutations) that all of the DNS resource record sets for
// dnsName are consistent with endpoints. if endpoints is nil or empty, a CNAME record to uplevelCname is ensured.
func (dc *DNSController) ensureDNSRrsets(dnsZone dnsprovider.Zone, service *apiv1.Service, addRecords bool) (
	err error) {
	rrsets, supported := dnsZone.ResourceRecordSets()
	if !supported {
		return fmt.Errorf("DNS provider does not support the ResourceRecordSets interface")
	}

	changeSet := rrsets.StartChangeset()

	prefix := service.Name + "." + service.Namespace + "." + dc.federationName + ".svc"

	ingress := &fed.FederatedServiceIngress{}
	if addRecords {
		ingress, err = servicecontroller.ParseFederationServiceIngresses(service)
		if err != nil {
			glog.Errorf("Error in parsing lb endpoints for service %s/%s: %v",
				service.Namespace, service.Name, err)
			return err
		}
	}

	allEndpoints := make(map[string][]string)
	globalEndpoints := []string{}
	for regionName, region := range ingress.Endpoints {
		regionEndpoints := []string{}
		for zoneName, zone := range region {
			zoneEndpoints := []string{}
			for _, clusterIngEps := range zone {
				if clusterIngEps.Healthy {
					zoneEndpoints = append(zoneEndpoints, clusterIngEps.Endpoints...)
				}
			}
			allEndpoints[prefix+"."+zoneName+"."+regionName+"."+dc.serviceDNSSuffix] = zoneEndpoints
			regionEndpoints = append(regionEndpoints, zoneEndpoints...)
		}
		allEndpoints[prefix+"."+regionName+"."+dc.serviceDNSSuffix] = regionEndpoints
		globalEndpoints = append(globalEndpoints, regionEndpoints...)
	}
	allEndpoints[prefix+"."+dc.serviceDNSSuffix] = globalEndpoints

	for dnsName, endpoints := range allEndpoints {
		uplevelCname := dc.getUplevelCName(dnsName, prefix)

		rrset, err := rrsets.Get(dnsName)
		if err != nil {
			return err
		}

		if len(endpoints) > 0 {
			resolvedEndpoints, err := getResolvedEndpoints(endpoints)
			if err != nil {
				return err
			}
			newRrset := rrsets.New(dnsName, resolvedEndpoints, minDNSTTL, rrstype.A)
			if rrset != nil {
				if !dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
					glog.V(4).Infof("Replacing recordset %v with %v for %s",
						rrset, newRrset, dnsName)
					changeSet.Remove(rrset).Add(newRrset)
				}
			} else {
				glog.V(4).Infof("Adding recordset %v for %s", newRrset, dnsName)
				changeSet.Add(newRrset)
			}
		} else {
			glog.V(4).Infof("No healthy endpoints for %s.", dnsName)
			newRrset := rrsets.New(dnsName, []string{uplevelCname}, minDNSTTL, rrstype.CNAME)
			if rrset != nil {
				if !dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
					glog.V(4).Infof("Replacing recordset %v with %v for %s",
						rrset, newRrset, dnsName)
					changeSet.Remove(rrset).Add(newRrset)
				}
			} else {
				if uplevelCname != "" {
					glog.V(4).Infof("Adding recordset %v for %s", newRrset, dnsName)
					changeSet.Add(newRrset)
				}
			}
		}
	}

	if err := changeSet.Apply(); err != nil {
		return err
	}

	return nil
}

// reconcileDNS updates the dns records for federated service in the configured dns provider.
func (dc *DNSController) reconcileDNS(service *apiv1.Service) {
	key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}.String()

	addRecords := true
	if service.DeletionTimestamp != nil {
		glog.V(2).Infof("Deleting federation dns records for %s service", key)
		addRecords = false
	}

	if err := dc.ensureDNSRrsets(dc.dnsZone, service, addRecords); err != nil {
		glog.Errorf("Error in ensuring dns rrsets for service %s: %v", key, err)
		return
	}
}
