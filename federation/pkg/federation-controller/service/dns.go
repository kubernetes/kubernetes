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

package service

import (
	"fmt"
	"net"

	"github.com/golang/glog"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
	"strings"
)

const (
	// minDnsTtl is the minimum safe DNS TTL value to use (in seconds).  We use this as the TTL for all DNS records.
	minDnsTtl = 180
)

// getHealthyEndpoints returns the hostnames and/or IP addresses of healthy endpoints for the service, at a zone, region and global level (or an error)
func (s *ServiceController) getHealthyEndpoints(clusterName string, cachedService *cachedService) (zoneEndpoints, regionEndpoints, globalEndpoints []string, err error) {
	var (
		zoneNames  []string
		regionName string
	)
	if zoneNames, regionName, err = s.getClusterZoneNames(clusterName); err != nil {
		return nil, nil, nil, err
	}
	for lbClusterName, lbStatus := range cachedService.serviceStatusMap {
		lbZoneNames, lbRegionName, err := s.getClusterZoneNames(lbClusterName)
		if err != nil {
			return nil, nil, nil, err
		}
		for _, ingress := range lbStatus.Ingress {
			readyEndpoints, ok := cachedService.endpointMap[lbClusterName]
			if !ok || readyEndpoints == 0 {
				continue
			}
			var address string
			// We should get either an IP address or a hostname - use whichever one we get
			if ingress.IP != "" {
				address = ingress.IP
			} else if ingress.Hostname != "" {
				address = ingress.Hostname
			}
			if len(address) <= 0 {
				return nil, nil, nil, fmt.Errorf("Service %s/%s in cluster %s has neither LoadBalancerStatus.ingress.ip nor LoadBalancerStatus.ingress.hostname. Cannot use it as endpoint for federated service.",
					cachedService.lastState.Name, cachedService.lastState.Namespace, clusterName)
			}
			for _, lbZoneName := range lbZoneNames {
				for _, zoneName := range zoneNames {
					if lbZoneName == zoneName {
						zoneEndpoints = append(zoneEndpoints, address)
					}
				}
			}
			if lbRegionName == regionName {
				regionEndpoints = append(regionEndpoints, address)
			}
			globalEndpoints = append(globalEndpoints, address)
		}
	}
	return zoneEndpoints, regionEndpoints, globalEndpoints, nil
}

// getClusterZoneNames returns the name of the zones (and the region) where the specified cluster exists (e.g. zones "us-east1-c" on GCE, or "us-east-1b" on AWS)
func (s *ServiceController) getClusterZoneNames(clusterName string) (zones []string, region string, err error) {
	client, ok := s.clusterCache.clientMap[clusterName]
	if !ok {
		return nil, "", fmt.Errorf("Cluster cache does not contain entry for cluster %s", clusterName)
	}
	if client.cluster == nil {
		return nil, "", fmt.Errorf("Cluster cache entry for cluster %s is nil", clusterName)
	}
	return client.cluster.Status.Zones, client.cluster.Status.Region, nil
}

// getServiceDnsSuffix returns the DNS suffix to use when creating federated-service DNS records
func (s *ServiceController) getServiceDnsSuffix() (string, error) {
	return s.serviceDnsSuffix, nil
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

/* getRrset is a hack around the fact that dnsprovider.ResourceRecordSets interface does not yet include a Get() method, only a List() method.  TODO:  Fix that.
   Note that if the named resource record set does not exist, but no error occurred, the returned set, and error, are both nil
*/
func getRrset(dnsName string, rrsetsInterface dnsprovider.ResourceRecordSets) (dnsprovider.ResourceRecordSet, error) {
	var returnVal dnsprovider.ResourceRecordSet
	rrsets, err := rrsetsInterface.List()
	if err != nil {
		return nil, err
	}
	for _, rrset := range rrsets {
		if rrset.Name() == dnsName {
			returnVal = rrset
			break
		}
	}
	return returnVal, nil
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

/* ensureDnsRrsets ensures (idempotently, and with minimum mutations) that all of the DNS resource record sets for dnsName are consistent with endpoints.
   if endpoints is nil or empty, a CNAME record to uplevelCname is ensured.
*/
func (s *ServiceController) ensureDnsRrsets(dnsZone dnsprovider.Zone, dnsName string, endpoints []string, uplevelCname string) error {
	rrsets, supported := dnsZone.ResourceRecordSets()
	if !supported {
		return fmt.Errorf("Failed to ensure DNS records for %s. DNS provider does not support the ResourceRecordSets interface.", dnsName)
	}
	rrset, err := getRrset(dnsName, rrsets) // TODO: rrsets.Get(dnsName)
	if err != nil {
		return err
	}
	if rrset == nil {
		glog.V(4).Infof("No recordsets found for DNS name %q.  Need to add either A records (if we have healthy endpoints), or a CNAME record to %q", dnsName, uplevelCname)
		if len(endpoints) < 1 {
			glog.V(4).Infof("There are no healthy endpoint addresses at level %q, so CNAME to %q, if provided", dnsName, uplevelCname)
			if uplevelCname != "" {
				glog.V(4).Infof("Creating CNAME to %q for %q", uplevelCname, dnsName)
				newRrset := rrsets.New(dnsName, []string{uplevelCname}, minDnsTtl, rrstype.CNAME)
				glog.V(4).Infof("Adding recordset %v", newRrset)
				err = rrsets.StartChangeset().Add(newRrset).Apply()
				if err != nil {
					return err
				}
				glog.V(4).Infof("Successfully created CNAME to %q for %q", uplevelCname, dnsName)
			} else {
				glog.V(4).Infof("We want no record for %q, and we have no record, so we're all good.", dnsName)
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
			glog.V(4).Infof("Adding recordset %v", newRrset)
			err = rrsets.StartChangeset().Add(newRrset).Apply()
			if err != nil {
				return err
			}
			glog.V(4).Infof("Successfully added recordset %v", newRrset)
		}
	} else {
		// the rrset already exists, so make it right.
		glog.V(4).Infof("Recordset %v already exists.  Ensuring that it is correct.", rrset)
		if len(endpoints) < 1 {
			// Need an appropriate CNAME record.  Check that we have it.
			newRrset := rrsets.New(dnsName, []string{uplevelCname}, minDnsTtl, rrstype.CNAME)
			glog.V(4).Infof("No healthy endpoints for %s.  Have recordset %v. Need recordset %v", dnsName, rrset, newRrset)
			if dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
				// The existing rrset is equivalent to the required one - our work is done here
				glog.V(4).Infof("Existing recordset %v is equivalent to needed recordset %v, our work is done here.", rrset, newRrset)
				return nil
			} else {
				// Need to replace the existing one with a better one (or just remove it if we have no healthy endpoints).
				glog.V(4).Infof("Existing recordset %v not equivalent to needed recordset %v removing existing and adding needed.", rrset, newRrset)
				changeSet := rrsets.StartChangeset()
				changeSet.Remove(rrset)
				if uplevelCname != "" {
					changeSet.Add(newRrset)
					if err := changeSet.Apply(); err != nil {
						return err
					}
					glog.V(4).Infof("Successfully replaced needed recordset %v -> %v", rrset, newRrset)
				} else {
					if err := changeSet.Apply(); err != nil {
						return err
					}
					glog.V(4).Infof("Successfully removed existing recordset %v", rrset)
					glog.V(4).Infof("Uplevel CNAME is empty string. Not adding recordset %v", newRrset)
				}
			}
		} else {
			// We have an rrset in DNS, possibly with some missing addresses and some unwanted addresses.
			// And we have healthy endpoints.  Just replace what's there with the healthy endpoints, if it's not already correct.
			glog.V(4).Infof("%s: Healthy endpoints %v exist.  Recordset %v exists.  Reconciling.", dnsName, endpoints, rrset)
			resolvedEndpoints, err := getResolvedEndpoints(endpoints)
			if err != nil { // Some invalid addresses or otherwise unresolvable DNS names.
				return err // TODO: We could potentially add the ones we did get back, even if some of them failed to resolve.
			}
			newRrset := rrsets.New(dnsName, resolvedEndpoints, minDnsTtl, rrstype.A)
			glog.V(4).Infof("Have recordset %v. Need recordset %v", rrset, newRrset)
			if dnsprovider.ResourceRecordSetsEquivalent(rrset, newRrset) {
				glog.V(4).Infof("Existing recordset %v is equivalent to needed recordset %v, our work is done here.", rrset, newRrset)
				// TODO: We could be more thorough about checking for equivalence to avoid unnecessary updates, but in the
				//       worst case we'll just replace what's there with an equivalent, if not exactly identical record set.
				return nil
			} else {
				// Need to replace the existing one with a better one
				glog.V(4).Infof("Existing recordset %v is not equivalent to needed recordset %v, removing existing and adding needed.", rrset, newRrset)
				if err = rrsets.StartChangeset().Remove(rrset).Add(newRrset).Apply(); err != nil {
					return err
				}
				glog.V(4).Infof("Successfully replaced recordset %v -> %v", rrset, newRrset)
			}
		}
	}
	return nil
}

/* ensureDnsRecords ensures (idempotently, and with minimum mutations) that all of the DNS records for a service in a given cluster are correct,
given the current state of that service in that cluster.  This should be called every time the state of a service might have changed
(either w.r.t. it's loadbalancer address, or if the number of healthy backend endpoints for that service transitioned from zero to non-zero
(or vice verse).  Only shards of the service which have both a loadbalancer ingress IP address or hostname AND at least one healthy backend endpoint
are included in DNS records for that service (at all of zone, region and global levels). All other addresses are removed.  Also, if no shards exist
in the zone or region of the cluster, a CNAME reference to the next higher level is ensured to exist. */
func (s *ServiceController) ensureDnsRecords(clusterName string, cachedService *cachedService) error {
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
	//
	// For each cached service, cachedService.lastState tracks the current known state of the service, while cachedService.appliedState contains
	// the state of the service when we last successfully sync'd it's DNS records.
	// So this time around we only need to patch that (add new records, remove deleted records, and update changed records.
	//
	if s == nil {
		return fmt.Errorf("nil ServiceController passed to ServiceController.ensureDnsRecords(clusterName: %s, cachedService: %v)", clusterName, cachedService)
	}
	if s.dns == nil {
		return nil
	}
	if cachedService == nil {
		return fmt.Errorf("nil cachedService passed to ServiceController.ensureDnsRecords(clusterName: %s, cachedService: %v)", clusterName, cachedService)
	}
	serviceName := cachedService.lastState.Name
	namespaceName := cachedService.lastState.Namespace
	zoneNames, regionName, err := s.getClusterZoneNames(clusterName)
	if err != nil {
		return err
	}
	if zoneNames == nil {
		return fmt.Errorf("failed to get cluster zone names")
	}
	serviceDnsSuffix, err := s.getServiceDnsSuffix()
	if err != nil {
		return err
	}
	zoneEndpoints, regionEndpoints, globalEndpoints, err := s.getHealthyEndpoints(clusterName, cachedService)
	if err != nil {
		return err
	}
	commonPrefix := serviceName + "." + namespaceName + "." + s.federationName + ".svc"
	// dnsNames is the path up the DNS search tree, starting at the leaf
	dnsNames := []string{
		commonPrefix + "." + zoneNames[0] + "." + regionName + "." + serviceDnsSuffix, // zone level - TODO might need other zone names for multi-zone clusters
		commonPrefix + "." + regionName + "." + serviceDnsSuffix,                      // region level, one up from zone level
		commonPrefix + "." + serviceDnsSuffix,                                         // global level, one up from region level
		"", // nowhere to go up from global level
	}

	endpoints := [][]string{zoneEndpoints, regionEndpoints, globalEndpoints}

	dnsZone, err := getDnsZone(s.zoneName, s.zoneID, s.dnsZones)
	if err != nil {
		return err
	}

	for i, endpoint := range endpoints {
		if err = s.ensureDnsRrsets(dnsZone, dnsNames[i], endpoint, dnsNames[i+1]); err != nil {
			return err
		}
	}
	return nil
}
