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

package ipvs

import (
	"fmt"
	"net"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/slice"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
	"github.com/google/seesaw/ipvs"
)

const defaultScheduler = "rr"

type portal struct {
	ip         net.IP
	port       int
	isExternal bool
}

// internal struct for string service information
type serviceInfo struct {
	portal             portal
	protocol           api.Protocol
	nodePort           int
	externalIPs        []string
	loadBalancerStatus api.LoadBalancerStatus

	sessionAffinityType api.ServiceAffinity
	stickyMaxAgeSeconds int
}

// returns a new serviceInfo struct
func newServiceInfo() *serviceInfo {
	return &serviceInfo{
		sessionAffinityType: api.ServiceAffinityNone, // default
		// TODO: paramaterize this in the API. By now ipvs will use the default value(300s)
		//stickyMaxAgeSeconds: 300,
	}
}

type Proxier struct {
	mu sync.Mutex

	iptables   iptables.Interface
	syncPeriod time.Duration

	serviceMap                  map[proxy.ServicePortName]serviceInfo
	endpointsMap                map[proxy.ServicePortName][]string
	haveReceivedServiceUpdate   bool // true once we've seen an OnServiceUpdate event
	haveReceivedEndpointsUpdate bool // true once we've seen an OnEndpointsUpdate event
}

// assert Proxier is a ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

func NewProxier(iptables iptables.Interface, syncPeriod time.Duration) (*Proxier, error) {
	p := &Proxier{
		iptables:     iptables,
		syncPeriod:   syncPeriod,
		serviceMap:   map[proxy.ServicePortName]serviceInfo{},
		endpointsMap: map[proxy.ServicePortName][]string{},
	}

	// Set up the iptables foundations we need.
	if err := iptablesInit(iptables); err != nil {
		return nil, fmt.Errorf("failed to initialize iptables: %v", err)
	}
	// Flush old iptables rules (since the bound ports will be invalid after a restart).
	// When OnUpdate() is first called, the rules will be recreated.
	if err := iptablesFlush(iptables); err != nil {
		return nil, fmt.Errorf("failed to flush iptables: %v", err)
	}

	if err := ipvs.Init(); err != nil {
		return nil, err
	}
	return p, nil
}

func (proxier *Proxier) OnServiceUpdate(services []api.Service) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedServiceUpdate = true

	activeServices := make(map[proxy.ServicePortName]bool) // use a map as a set

	for i := range services {
		service := &services[i]
		svcName := types.NamespacedName{
			Namespace: service.Namespace,
			Name:      service.Name,
		}

		// if ClusterIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			serviceName := proxy.ServicePortName{
				NamespacedName: svcName,
				Port:           servicePort.Name,
			}
			activeServices[serviceName] = true
			info, exists := proxier.serviceMap[serviceName]
			if exists && proxier.sameConfig(&info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				// Something changed.
				glog.V(3).Infof("Something changed for service %q: removing it", serviceName)
				err := proxier.deleteIptablesRule(serviceName, &info)
				if err != nil {
					glog.Errorf("Failed to delete iptables rule for %q: %v", serviceName, err)
				}
				delete(proxier.serviceMap, serviceName)
			}
			serviceIP := net.ParseIP(service.Spec.ClusterIP)
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)

			info = *newServiceInfo()
			info.portal.ip = serviceIP
			info.portal.port = int(servicePort.Port)
			info.protocol = servicePort.Protocol
			info.nodePort = int(servicePort.NodePort)
			info.externalIPs = service.Spec.ExternalIPs
			// Deep-copy in case the service instance changes
			info.loadBalancerStatus = *api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
			info.sessionAffinityType = service.Spec.SessionAffinity
			proxier.serviceMap[serviceName] = info

			err := proxier.AddIptablesRule(serviceName, &info)
			if err != nil {
				glog.Errorf("Failed to add iptables rule for %q: %v", serviceName, err)
			}

			glog.V(4).Infof("added serviceInfo(%s): %s", serviceName, spew.Sdump(info))
		}
	}

	// Remove services missing from the update.
	for name, info := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Removing service %q", name)
			err := proxier.deleteIptablesRule(name, &info)
			if err != nil {
				glog.Errorf("Failed to delete iptables rule for %q: %v", name, err)
			}
			delete(proxier.serviceMap, name)
		}
	}

	proxier.syncIpvsRules()
}

// OnEndpointsUpdate takes in a slice of updated endpoints.
func (proxier *Proxier) OnEndpointsUpdate(allEndpoints []api.Endpoints) {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("OnEndpointsUpdate took %v for %d endpoints", time.Since(start), len(allEndpoints))
	}()

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedEndpointsUpdate = true

	activeEndpoints := make(map[proxy.ServicePortName]bool) // use a map as a set

	// Update endpoints for services.
	for i := range allEndpoints {
		svcEndpoints := &allEndpoints[i]

		// We need to build a map of portname -> all ip:ports for that
		// portname.  Explode Endpoints.Subsets[*] into this structure.
		portsToEndpoints := map[string][]hostPortPair{}
		for i := range svcEndpoints.Subsets {
			ss := &svcEndpoints.Subsets[i]
			for i := range ss.Ports {
				port := &ss.Ports[i]
				for i := range ss.Addresses {
					addr := &ss.Addresses[i]
					portsToEndpoints[port.Name] = append(portsToEndpoints[port.Name], hostPortPair{addr.IP, int(port.Port)})
				}
			}
		}

		for portname := range portsToEndpoints {
			svcPort := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: svcEndpoints.Namespace, Name: svcEndpoints.Name}, Port: portname}
			curEndpoints := proxier.endpointsMap[svcPort]
			newEndpoints := flattenValidEndpoints(portsToEndpoints[portname])

			if len(curEndpoints) != len(newEndpoints) || !slicesEquiv(slice.CopyStrings(curEndpoints), newEndpoints) {
				glog.V(1).Infof("Setting endpoints for %q to %+v", svcPort, newEndpoints)
				proxier.endpointsMap[svcPort] = newEndpoints
			}
			activeEndpoints[svcPort] = true
		}
	}

	// Remove endpoints missing from the update.
	for name := range proxier.endpointsMap {
		if !activeEndpoints[name] {
			glog.V(2).Infof("Removing endpoints for %q", name)
			delete(proxier.endpointsMap, name)
		}
	}

	proxier.syncIpvsRules()
}

func (proxier *Proxier) syncIpvsRules() {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.haveReceivedEndpointsUpdate || !proxier.haveReceivedServiceUpdate {
		glog.V(2).Info("Not syncing iptables until Services and Endpoints have been received from master")
		return
	}
	glog.V(3).Infof("Syncing ipvs rules")

	// sync ipvs rules
	type serviceKey struct {
		ip       string
		port     int
		protocol api.Protocol
	}
	activeServices := make(map[serviceKey]bool)

	for svcName, svcInfo := range proxier.serviceMap {
		svcKey := serviceKey{
			ip:       svcInfo.portal.ip.String(),
			port:     svcInfo.portal.port,
			protocol: svcInfo.protocol,
		}
		activeServices[svcKey] = true

		// sync service
		{
			applied, err := ipvs.GetService(&ipvs.Service{
				Address:  svcInfo.portal.ip,
				Port:     uint16(svcInfo.portal.port),
				Protocol: toProtoNum(svcInfo.protocol),
			})
			if err != nil || applied.Scheduler != defaultScheduler || (applied.Flags&ipvs.SFPersistent) == 0 {
				if err == nil {
					glog.V(4).Infof("Something changed for service %v: stopping it", svcName)
					err = ipvs.DeleteService(ipvs.Service{
						Address:  svcInfo.portal.ip,
						Port:     uint16(svcInfo.portal.port),
						Protocol: toProtoNum(svcInfo.protocol),
					})
					if err != nil {
						glog.Errorf("Failed to remove Service for %v: %v", svcName, err)
					}
				}

				glog.V(1).Infof("Adding new service %q at %s:%d/%s", svcName, svcInfo.portal.ip, svcInfo.portal.port, svcInfo.protocol)

				var flags ipvs.ServiceFlags
				if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
					flags |= ipvs.SFPersistent
				}

				err = ipvs.AddService(ipvs.Service{
					Address:  svcInfo.portal.ip,
					Port:     uint16(svcInfo.portal.port),
					Protocol: toProtoNum(svcInfo.protocol),
					Flags:    flags,
				})
				if err != nil {
					glog.Errorf("Failed to add Service for %q: %v", svcName, err)
				}
			}
		}
		// sync endpoints
		{
			applied, err := ipvs.GetService(&ipvs.Service{
				Address:  svcInfo.portal.ip,
				Port:     uint16(svcInfo.portal.port),
				Protocol: toProtoNum(svcInfo.protocol),
			})
			if err != nil {
				continue
			}

			curEndpoints := sets.NewString()
			newEndpoints := sets.NewString()
			for _, des := range applied.Destinations {
				curEndpoints.Insert([]string{net.JoinHostPort(des.Address.String(), fmt.Sprintf("%s", des.Port))}...)
			}
			for _, eps := range proxier.endpointsMap[svcName] {
				newEndpoints.Insert(eps)
			}
			if !curEndpoints.Equal(newEndpoints) {
				for _, ep := range newEndpoints.Difference(curEndpoints).List() {
					ip, port := parseHostPort(ep)
					ipvs.AddDestination(*applied, ipvs.Destination{
						Address: net.ParseIP(ip),
						Port:    uint16(port),
					})
				}
				for _, ep := range curEndpoints.Difference(newEndpoints).List() {
					ip, port := parseHostPort(ep)
					ipvs.DeleteDestination(*applied, ipvs.Destination{
						Address: net.ParseIP(ip),
						Port:    uint16(port),
					})
				}
			}
		}
	}

	// reads all applied ipvs rules from the host
	curServices, err := ipvs.GetServices()
	if err != nil {
		glog.Errorf("Failed to list all ipvs Service from host: %v", err)
		return
	}
	for _, srv := range curServices {
		svcKey := serviceKey{
			ip:       srv.Address.String(),
			port:     int(srv.Port),
			protocol: toProtoType(srv.Protocol),
		}

		if !activeServices[svcKey] {
			glog.V(1).Infof("Stopping service %q", srv)
			err := ipvs.DeleteService(*srv)
			if err != nil {
				glog.Errorf("Failed to remove Service for %q: %v", srv, err)
			}
		}
	}
}

func CleanupIpvsLeftover() error {
	return ipvs.Flush()
}

// Sync is called to immediately synchronize the proxier state to iptables
func (proxier *Proxier) Sync() {
	if err := iptablesInit(proxier.iptables); err != nil {
		glog.Errorf("Failed to ensure iptables: %v", err)
	}
	proxier.ensureIptablesRules()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		proxier.Sync()
	}
}

// Ensure that portals exist for all services.
func (proxier *Proxier) ensureIptablesRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	// NB: This does not remove rules that should not be present.
	for name, info := range proxier.serviceMap {
		err := proxier.AddIptablesRule(name, &info)
		if err != nil {
			glog.Errorf("Failed to ensure portal for %q: %v", name, err)
		}
	}
}

func (proxier *Proxier) sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.portal.port != int(port.Port) || info.nodePort != int(port.NodePort) {
		return false
	}
	if !info.portal.ip.Equal(net.ParseIP(service.Spec.ClusterIP)) {
		return false
	}
	if !ipsEqual(info.externalIPs, service.Spec.ExternalIPs) {
		return false
	}
	if !api.LoadBalancerStatusEqual(&info.loadBalancerStatus, &service.Status.LoadBalancer) {
		return false
	}
	if info.sessionAffinityType != service.Spec.SessionAffinity {
		return false
	}
	return true
}

func ipsEqual(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if lhs[i] != rhs[i] {
			return false
		}
	}
	return true
}

type endpointServicePair struct {
	endpoint        string
	servicePortName proxy.ServicePortName
}

// used in OnEndpointsUpdate
type hostPortPair struct {
	host string
	port int
}

func isValidEndpoint(hpp *hostPortPair) bool {
	return hpp.host != "" && hpp.port > 0
}

// Tests whether two slices are equivalent.  This sorts both slices in-place.
func slicesEquiv(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	if reflect.DeepEqual(slice.SortStrings(lhs), slice.SortStrings(rhs)) {
		return true
	}
	return false
}

func flattenValidEndpoints(endpoints []hostPortPair) []string {
	// Convert Endpoint objects into strings for easier use later.
	var result []string
	for i := range endpoints {
		hpp := &endpoints[i]
		if isValidEndpoint(hpp) {
			result = append(result, net.JoinHostPort(hpp.host, strconv.Itoa(hpp.port)))
		} else {
			glog.Warningf("got invalid endpoint: %+v", *hpp)
		}
	}
	return result
}

func toProtoNum(proto api.Protocol) ipvs.IPProto {
	p := string(proto)
	switch strings.ToLower(p) {
	case "tcp":
		return ipvs.IPProto(syscall.IPPROTO_TCP)
	case "udp":
		return ipvs.IPProto(syscall.IPPROTO_UDP)
	}
	return ipvs.IPProto(0)
}

func toProtoType(proto ipvs.IPProto) api.Protocol {
	switch proto {
	case syscall.IPPROTO_TCP:
		return api.ProtocolTCP
	case syscall.IPPROTO_UDP:
		return api.ProtocolUDP
	}
	return api.ProtocolTCP
}

func parseHostPort(hostPort string) (string, int) {
	host, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		return hostPort, 0
	}
	intPort, err := strconv.Atoi(port)
	if err != nil {
		return hostPort, 0
	}
	return host, intPort
}
