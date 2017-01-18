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

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/golang/glog"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	fed "k8s.io/kubernetes/federation/apis/federation"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
	si "k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/legacylisters"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

const (
	ControllerName = "service-dns"

	// minDnsTtl is the minimum safe DNS TTL value to use (in seconds).  We use this as the TTL for all DNS records.
	minDnsTtl = 180
)

var (
	RequiredResources = []schema.GroupVersionResource{v1.SchemeGroupVersion.WithResource("services")}
)

type ServiceDNSController struct {
	dns            dnsprovider.Interface
	federationName string
	// serviceDnsSuffix is the DNS suffix we use when publishing service DNS names
	serviceDnsSuffix string
	// zoneName and zoneID are used to identify the zone in which to put records
	zoneName string
	zoneID   string
	// Each federation should be configured with a single zone (e.g. "mycompany.com")
	dnsZones dnsprovider.Zones
	dnsZone  dnsprovider.Zone
	// Informer Store for federated services.
	serviceInformerStore listers.StoreToReplicaSetLister
	// Informer controller for federated services.
	serviceInformerController cache.Controller
	// Client to federated api server.
	federationClient fedclientset.Interface
	workQueue        workqueue.Interface
}

// NewServiceDNSController returns a new service dns controller
func NewServiceDNSController(client fedclientset.Interface, dnsProvider, dnsProviderConfig, federationName,
	serviceDNSSuffix, zoneName, zoneID string) (*ServiceDNSController, error) {
	dns, err := dnsprovider.InitDnsProvider(dnsProvider, dnsProviderConfig)
	if err != nil {
		glog.Fatalf("DNS provider could not be initialized: %v", err)
		return nil, err
	}

	d := &ServiceDNSController{
		federationClient: client,
		dns:              dns,
		federationName:   federationName,
		serviceDnsSuffix: serviceDNSSuffix,
		zoneName:         zoneName,
		zoneID:           zoneID,
		workQueue:        workqueue.New(),
	}

	// Start informer in federated API servers on services that should be federated.
	d.serviceInformerStore.Indexer, d.serviceInformerController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (pkgruntime.Object, error) {
				return client.Core().Services(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return client.Core().Services(v1.NamespaceAll).Watch(options)
			},
		},
		&v1.Service{},
		controller.NoResyncPeriodFunc(),
		util.NewTriggerOnAllChanges(func(obj pkgruntime.Object) { d.workQueue.Add(obj) }),
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	// Check configuration
	if err := d.init(); err != nil {
		glog.Fatalf("DNS provider failed to initialize: %v", err)
		return nil, err
	}

	return d, nil
}

// Check and initialize configuration
func (d *ServiceDNSController) init() error {
	if d.federationName == "" {
		return fmt.Errorf("ServiceDNSController should not be run without federationName.")
	}
	if d.zoneName == "" && d.zoneID == "" {
		return fmt.Errorf("ServiceDNSController must be run with either zoneName or zoneID.")
	}
	if d.serviceDnsSuffix == "" {
		// TODO: Is this the right place to do defaulting?
		if d.zoneName == "" {
			return fmt.Errorf("ServiceDNSController must be run with zoneName, if serviceDnsSuffix is not set.")
		}
		d.serviceDnsSuffix = d.zoneName
	}
	if d.dns == nil {
		return fmt.Errorf("ServiceDNSController should not be run without a dnsprovider.")
	}
	zones, ok := d.dns.Zones()
	if !ok {
		return fmt.Errorf("the dns provider does not support zone enumeration, which is required for creating dns records.")
	}
	d.dnsZones = zones
	matchingZones, err := getDnsZones(d.zoneName, d.zoneID, d.dnsZones)
	if err != nil {
		return fmt.Errorf("error querying for DNS zones: %v", err)
	}
	if len(matchingZones) == 0 {
		if d.zoneName == "" {
			return fmt.Errorf("ServiceDNSController must be run with zoneName to create zone automatically.")
		}
		glog.Infof("DNS zone %q not found.  Creating DNS zone %q.", d.zoneName, d.zoneName)
		managedZone, err := d.dnsZones.New(d.zoneName)
		if err != nil {
			return err
		}
		zone, err := d.dnsZones.Add(managedZone)
		if err != nil {
			return err
		}
		glog.Infof("DNS zone %q successfully created.  Note that DNS resolution will not work until you have registered this name with "+
			"a DNS registrar and they have changed the authoritative name servers for your domain to point to your DNS provider.", zone.Name())
	}
	if len(matchingZones) > 1 {
		return fmt.Errorf("Multiple matching DNS zones found for %q; please specify zoneID", d.zoneName)
	}
	d.dnsZone, err = getDnsZone(d.zoneName, d.zoneID, d.dnsZones)
	if err != nil {
		return fmt.Errorf("Get DNS Zone failed: %v", err)
	}
	return nil
}

// getDnsZones returns the DNS zones matching dnsZoneName and dnsZoneID (if specified)
func getDnsZones(dnsZoneName string, dnsZoneID string, dnsZonesInterface dnsprovider.Zones) ([]dnsprovider.Zone, error) {
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

// getDnsZone returns the DNS zone, as identified by dnsZoneName and dnsZoneID
// This is similar to getDnsZones, but returns an error if there are zero or multiple matching zones.
func getDnsZone(dnsZoneName string, dnsZoneID string, dnsZonesInterface dnsprovider.Zones) (dnsprovider.Zone, error) {
	dnsZones, err := getDnsZones(dnsZoneName, dnsZoneID, dnsZonesInterface)
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
		return nil, fmt.Errorf("DNS zone %s not found.", name)
	} else {
		return nil, fmt.Errorf("DNS zone %s is ambiguous (please specify zoneID).", name)
	}
}

func (d *ServiceDNSController) Run(workers int, stopChan <-chan struct{}) {
	for i := 0; i < workers; i++ {
		go wait.Until(d.worker, time.Second, stopChan)
	}
	go d.serviceInformerController.Run(stopChan)
}

func (d *ServiceDNSController) worker() {
	for {
		item, quit := d.workQueue.Get()
		if quit {
			return
		}
		service := item.(*v1.Service)
		d.ensureDnsRecords(service)
		d.workQueue.Done(item)
	}
}

/* getResolvedEndpoints performs DNS resolution on the provided slice of endpoints (which might be DNS names or IPv4 addresses)
   and returns a list of IPv4 addresses.  If any of the endpoints are neither valid IPv4 addresses nor resolvable DNS names,
   non-nil error is also returned (possibly along with a partially complete list of resolved endpoints.
*/
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

// getUplevelCName returns one level higher name for a dnsName, dnsNamePrefix is the common prefix after which zone/region string exist before the serviceDNSSuffix in a FQDN
// if dnsName is of zone level, then return region CNAME, for e.g. if dnsName = "servicename.ns.federationname.svc.z1.r1.example.com", then return "servicename.ns.federationname.svc.r1.example.com"
// if dnsName is of region level, then return global CNAME, for e.g. if dnsName = "servicename.ns.federationname.svc.r1.example.com", then return "servicename.ns.federationname.svc.example.com"
// if dnsName is of global level, then return "" empty string, for e.g. if dnsName = "servicename.ns.federationname.svc.example.com", then return ""
func (d *ServiceDNSController) getUplevelCName(dnsName, dnsNamePrefix string) (uplevelCname string) {
	// split the dnsName with . as separator to get slice of strings
	dnsNameSlice := strings.Split(dnsName, ".")
	// find the index of the string which needs to be removed, so that we get uplevel CNAME
	indexToDelete := len(strings.Split(dnsNamePrefix, "."))
	// remove the zone/region from dnsName
	dnsNameSlice = append(dnsNameSlice[:indexToDelete], dnsNameSlice[indexToDelete+1:]...)

	uplevelCname = strings.Join(dnsNameSlice, ".")

	// uplevelCname should have serviceDNSSuffix, otherwise the input dnsName was a global level name, so return empty string
	if !strings.HasSuffix(uplevelCname, d.serviceDnsSuffix) {
		uplevelCname = ""
	}
	return uplevelCname
}

/* ensureDnsRrsets ensures (idempotently, and with minimum mutations) that all of the DNS resource record sets are consistent with endpoints.
 */
func (d *ServiceDNSController) ensureDnsRrsets(dnsZone dnsprovider.Zone, endpoints map[string][]string, dnsNamePrefix string) error {
	rrsets, supported := dnsZone.ResourceRecordSets()
	if !supported {
		return fmt.Errorf("DNS provider does not support the ResourceRecordSets interface")
	}

	changeSet := rrsets.StartChangeset()

	for dnsName, endpoints := range endpoints {
		rrset, err := rrsets.Get(dnsName)
		if err != nil {
			return err
		}

		if len(endpoints) < 1 {
			// There are no healthy endpoints for the dnsName, So need to add an appropriate CNAME record.
			uplevelCname := d.getUplevelCName(dnsName, dnsNamePrefix)
			glog.V(4).Infof("There are no healthy endpoint addresses at level %q, so CNAME to %q, if provided", dnsName, uplevelCname)
			if uplevelCname != "" {
				// dnsName is for zone or region level
				// should ensure a CNAME pointing to one level above
				newRrset := rrsets.New(dnsName, []string{uplevelCname}, minDnsTtl, rrstype.CNAME)
				if rrset == nil {
					// rrset does not exist, so need to create new CNAME Record
					glog.V(4).Infof("Creating CNAME to %q for %q", uplevelCname, dnsName)
					changeSet.Add(newRrset) // the rrset already exists, ensure that it is right.

				} else {
					if !dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
						glog.V(4).Infof("Have recordset %v. Need recordset %v", rrset, newRrset)
						changeSet.Remove(rrset).Add(newRrset)
					} else {
						glog.V(4).Infof("Existing recordset %v is equivalent to needed recordset %v, our work is done here.", rrset, newRrset)
					}
				}
			} else {
				// dnsName is for global level
				if rrset == nil {
					glog.V(4).Infof("We want no record for %q, and we have no record, so we're all good.", dnsName)
				} else {
					// the rrset exists, so should remove it as no healthy endpoints exist now.
					glog.V(4).Infof("Successfully removed existing recordset %v", rrset)
					changeSet.Remove(rrset)
				}
			}
		} else {
			// We have valid endpoint addresses, so just add them as A records.
			// But first resolve DNS names, as some cloud providers (like AWS) expose
			// load balancers behind DNS names, not IP addresses.
			glog.V(4).Infof("We have valid endpoint addresses %v at level %q, so add them as A records, after resolving DNS names", endpoints, dnsName)
			resolvedEndpoints, err := getResolvedEndpoints(endpoints)
			if err != nil {
				return err // TODO: We could potentially add the ones we did get back, even if some of them failed to resolve.
			}
			newRrset := rrsets.New(dnsName, resolvedEndpoints, minDnsTtl, rrstype.A)
			if rrset == nil {
				// rrset does not exist, so need to create new A Record
				glog.V(4).Infof("Adding recordset %v for %s", newRrset, dnsName)
				changeSet.Add(newRrset)
			} else {
				// the rrset already exists, ensure that it is right.
				if !dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
					// TODO: We could be more thorough about checking for equivalence to avoid unnecessary updates, but in the
					//       worst case we'll just replace what's there with an equivalent, if not exactly identical record set.
					glog.V(4).Infof("Have recordset %v. Need recordset %v", rrset, newRrset)
					changeSet.Remove(rrset).Add(newRrset)
				} else {
					glog.V(4).Infof("Existing recordset %v is equivalent to needed recordset %v, our work is done here.", rrset, newRrset)
				}
			}
		}
	}

	if err := changeSet.Apply(); err != nil {
		return err
	}

	return nil
}

/* ensureDnsRecords ensures (idempotently, and with minimum mutations) that all of the DNS records for a service are correct.
This should be called every time the state of a service might have changed
(either w.r.t. it's loadbalancer address, or if the number of healthy backend endpoints for that service transitioned from zero to non-zero
(or vice verse).  Only shards of the service which have both a loadbalancer ingress IP address or hostname AND at least one healthy backend endpoint
are included in DNS records for that service (at all of zone, region and global levels). All other addresses are removed.  Also, if no shards exist
in the zone or region of the cluster, a CNAME reference to the next higher level is ensured to exist. */
func (d *ServiceDNSController) ensureDnsRecords(service *v1.Service) {
	// Quinton: Pseudocode....
	// See https://github.com/kubernetes/kubernetes/pull/25107#issuecomment-218026648
	// For each service we need the following DNS names:
	// mysvc.myns.myfed.svc.z1.r1.mydomain.com  (for zone z1 in region r1)
	//         - an A record to IP address of specific shard in that zone (if that shard exists and has healthy endpoints)
	//         - OR a CNAME record to the next level up, i.e. mysvc.myns.myfed.svc.r1.mydomain.com  (if a healthy shard does not exist in zone z1)
	// mysvc.myns.myfed.svc.r1.federation
	//         - a set of A records to IP addresses of all healthy shards in region r1, if one or more of these exist
	//         - OR a CNAME record to the next level up, i.e. mysvc.myns.myfed.svc.mydomain.com (if no healthy shards exist in region r1)
	// mysvc.myns.myfed.svc.federation
	//         - a set of A records to IP addresses of all healthy shards in all regions, if one or more of these exist.
	//         - no record (NXRECORD response) if no healthy shards exist in any regions)

	key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}.String()

	dnsNamePrefix := service.Name + "." + service.Namespace + "." + d.federationName + ".svc"

	ingress := &fed.FederatedServiceIngress{}
	if service.DeletionTimestamp != nil {
		// if service is about to  be deleted, remove all ingress endpoints associated with the service
		glog.V(2).Infof("Deleting federation dns records for %s service", key)
	} else {
		// otherwise, parse the ingress from service annotations.
		var err error
		ingress, err = si.ParseFederationServiceIngresses(service)
		if err != nil {
			glog.Errorf("Error in parsing lb ingress for service %s/%s: %v", service.Namespace, service.Name, err)
			return
		}
	}

	endpointsMap := make(map[string][]string)
	globalEndpoints := []string{}
	for regionName, region := range ingress.Endpoints {
		regionEndpoints := []string{}
		for zoneName, zone := range region {
			zoneEndpoints := []string{}
			for _, clusterIngEps := range zone {
				// add service lb ingress endpoints only if they are backed by reachable service endpoints in underlying cluster
				if clusterIngEps.Healthy {
					zoneEndpoints = append(zoneEndpoints, clusterIngEps.Endpoints...)
				}
			}
			endpointsMap[dnsNamePrefix+"."+zoneName+"."+regionName+"."+d.serviceDnsSuffix] = zoneEndpoints
			regionEndpoints = append(regionEndpoints, zoneEndpoints...)
		}
		endpointsMap[dnsNamePrefix+"."+regionName+"."+d.serviceDnsSuffix] = regionEndpoints
		globalEndpoints = append(globalEndpoints, regionEndpoints...)
	}
	endpointsMap[dnsNamePrefix+"."+d.serviceDnsSuffix] = globalEndpoints

	if err := d.ensureDnsRrsets(d.dnsZone, endpointsMap, dnsNamePrefix); err != nil {
		glog.Errorf("Error in ensuring dns rrsets for service %s: %v", key, err)
		return
	}
}
