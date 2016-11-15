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
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/miekg/dns"
	skymsg "github.com/skynetservices/skydns/msg"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/dns/config"
	"k8s.io/kubernetes/pkg/dns/treecache"
	"k8s.io/kubernetes/pkg/dns/util"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// A subdomain added to the user specified domain for all services.
	serviceSubdomain = "svc"

	// A subdomain added to the user specified dmoain for all pods.
	podSubdomain = "pod"

	// Resync period for the kube controller loop.
	resyncPeriod = 5 * time.Minute

	// Duration for which the TTL cache should hold the node resource to retrieve the zone
	// annotation from it so that it could be added to federation CNAMEs. There is ideally
	// no need to expire this cache, but we don't want to assume that node annotations
	// never change. So we expire the cache and retrieve a node once every 180 seconds.
	// The value is chosen to be neither too long nor too short.
	nodeCacheTTL = 180 * time.Second
)

type KubeDNS struct {
	// kubeClient makes calls to API Server and registers calls with API Server
	// to get Endpoints and Service objects.
	kubeClient clientset.Interface

	// domain for which this DNS Server is authoritative.
	domain string
	// configMap where kube-dns dynamic configuration is store. If this
	// is empty then getting configuration from a configMap will be
	// disabled.
	configMap string

	// endpointsStore that contains all the endpoints in the system.
	endpointsStore kcache.Store
	// servicesStore that contains all the services in the system.
	servicesStore kcache.Store
	// nodesStore contains some subset of nodes in the system so that we
	// can retrieve the cluster zone annotation from the cached node
	// instead of getting it from the API server every time.
	nodesStore kcache.Store

	// cache stores DNS records for the domain.  A Records and SRV Records for
	// (regular) services and headless Services.  CNAME Records for
	// ExternalName Services.
	cache treecache.TreeCache
	// TODO(nikhiljindal): Remove this. It can be recreated using
	// clusterIPServiceMap.
	reverseRecordMap map[string]*skymsg.Service
	// clusterIPServiceMap to service object. Headless services are not
	// part of this map. Used to get a service when given its cluster
	// IP.  Access to this is coordinated using cacheLock. We use the
	// same lock for cache and this map to ensure that they don't get
	// out of sync.
	clusterIPServiceMap map[string]*kapi.Service
	// cacheLock protecting the cache. caller is responsible for using
	// the cacheLock before invoking methods on cache the cache is not
	// thread-safe, and the caller can guarantee thread safety by using
	// the cacheLock
	cacheLock sync.RWMutex

	// The domain for which this DNS Server is authoritative, in array
	// format and reversed.  e.g. if domain is "cluster.local",
	// domainPath is []string{"local", "cluster"}
	domainPath []string

	// endpointsController  invokes registered callbacks when endpoints change.
	endpointsController *kcache.Controller
	// serviceController invokes registered callbacks when services change.
	serviceController *kcache.Controller

	// config set from the dynamic configuration source.
	config *config.Config
	// configLock protects the config below.
	configLock sync.RWMutex
	// configSync manages synchronization of the config map
	configSync config.Sync
}

func NewKubeDNS(client clientset.Interface, clusterDomain string, configSync config.Sync) *KubeDNS {
	kd := &KubeDNS{
		kubeClient:          client,
		domain:              clusterDomain,
		cache:               treecache.NewTreeCache(),
		cacheLock:           sync.RWMutex{},
		nodesStore:          kcache.NewStore(kcache.MetaNamespaceKeyFunc),
		reverseRecordMap:    make(map[string]*skymsg.Service),
		clusterIPServiceMap: make(map[string]*kapi.Service),
		domainPath:          util.ReverseArray(strings.Split(strings.TrimRight(clusterDomain, "."), ".")),

		configLock: sync.RWMutex{},
		configSync: configSync,
	}

	kd.setEndpointsStore()
	kd.setServicesStore()

	return kd
}

func (kd *KubeDNS) Start() {
	glog.V(2).Infof("Starting endpointsController")
	go kd.endpointsController.Run(wait.NeverStop)

	glog.V(2).Infof("Starting serviceController")
	go kd.serviceController.Run(wait.NeverStop)

	kd.startConfigMapSync()

	// Wait synchronously for the Kubernetes service. This ensures that
	// the Start function returns only after having received Service
	// objects from APIServer.
	//
	// TODO: we might not have to wait for kubernetes service
	// specifically. We should just wait for a list operation to be
	// complete from APIServer.
	kd.waitForKubernetesService()
}

func (kd *KubeDNS) waitForKubernetesService() {
	glog.V(2).Infof("Waiting for Kubernetes service")

	const kubernetesSvcName = "kubernetes"
	const servicePollInterval = 1 * time.Second

	name := fmt.Sprintf("%v/%v", kapi.NamespaceDefault, kubernetesSvcName)
	glog.V(2).Infof("Waiting for service: %v", name)

	for {
		svc, err := kd.kubeClient.Core().Services(kapi.NamespaceDefault).Get(kubernetesSvcName)
		if err != nil || svc == nil {
			glog.V(3).Infof(
				"Ignoring error while waiting for service %v: %v. Sleeping %v before retrying.",
				name, err, servicePollInterval)
			time.Sleep(servicePollInterval)
			continue
		}
		break
	}

	return
}

func (kd *KubeDNS) startConfigMapSync() {
	initialConfig, err := kd.configSync.Once()
	if err != nil {
		glog.Errorf(
			"Error getting initial ConfigMap: %v, starting with default values", err)
		kd.config = config.NewDefaultConfig()
	} else {
		kd.config = initialConfig
	}

	go kd.syncConfigMap(kd.configSync.Periodic())
}

func (kd *KubeDNS) syncConfigMap(syncChan <-chan *config.Config) {
	for {
		nextConfig := <-syncChan

		kd.configLock.Lock()
		kd.config = nextConfig
		glog.V(2).Infof("Configuration updated: %+v", *kd.config)
		kd.configLock.Unlock()
	}
}

func (kd *KubeDNS) GetCacheAsJSON() (string, error) {
	kd.cacheLock.RLock()
	defer kd.cacheLock.RUnlock()
	json, err := kd.cache.Serialize()
	return json, err
}

func (kd *KubeDNS) setServicesStore() {
	// Returns a cache.ListWatch that gets all changes to services.
	kd.servicesStore, kd.serviceController = kcache.NewInformer(
		&kcache.ListWatch{
			ListFunc: func(options kapi.ListOptions) (runtime.Object, error) {
				return kd.kubeClient.Core().Services(kapi.NamespaceAll).List(options)
			},
			WatchFunc: func(options kapi.ListOptions) (watch.Interface, error) {
				return kd.kubeClient.Core().Services(kapi.NamespaceAll).Watch(options)
			},
		},
		&kapi.Service{},
		resyncPeriod,
		kcache.ResourceEventHandlerFuncs{
			AddFunc:    kd.newService,
			DeleteFunc: kd.removeService,
			UpdateFunc: kd.updateService,
		},
	)
}

func (kd *KubeDNS) setEndpointsStore() {
	// Returns a cache.ListWatch that gets all changes to endpoints.
	kd.endpointsStore, kd.endpointsController = kcache.NewInformer(
		&kcache.ListWatch{
			ListFunc: func(options kapi.ListOptions) (runtime.Object, error) {
				return kd.kubeClient.Core().Endpoints(kapi.NamespaceAll).List(options)
			},
			WatchFunc: func(options kapi.ListOptions) (watch.Interface, error) {
				return kd.kubeClient.Core().Endpoints(kapi.NamespaceAll).Watch(options)
			},
		},
		&kapi.Endpoints{},
		resyncPeriod,
		kcache.ResourceEventHandlerFuncs{
			AddFunc: kd.handleEndpointAdd,
			UpdateFunc: func(oldObj, newObj interface{}) {
				// TODO: Avoid unwanted updates.
				kd.handleEndpointAdd(newObj)
			},
			// No DeleteFunc for EndpointsStore because endpoint object will be deleted
			// when corresponding service is deleted.
		},
	)
}

func assertIsService(obj interface{}) (*kapi.Service, bool) {
	if service, ok := obj.(*kapi.Service); ok {
		return service, ok
	} else {
		glog.Errorf("Type assertion failed! Expected 'Service', got %T", service)
		return nil, ok
	}
}

func (kd *KubeDNS) newService(obj interface{}) {
	if service, ok := assertIsService(obj); ok {
		glog.V(2).Infof("New service: %v", service.Name)
		glog.V(4).Infof("Service details: %v", service)

		// ExternalName services are a special kind that return CNAME records
		if service.Spec.Type == kapi.ServiceTypeExternalName {
			kd.newExternalNameService(service)
			return
		}
		// if ClusterIP is not set, a DNS entry should not be created
		if !kapi.IsServiceIPSet(service) {
			kd.newHeadlessService(service)
			return
		}
		if len(service.Spec.Ports) == 0 {
			glog.Warningf("Service with no ports, this should not have happened: %v",
				service)
		}
		kd.newPortalService(service)
	}
}

func (kd *KubeDNS) removeService(obj interface{}) {
	if s, ok := assertIsService(obj); ok {
		subCachePath := append(kd.domainPath, serviceSubdomain, s.Namespace, s.Name)
		kd.cacheLock.Lock()
		defer kd.cacheLock.Unlock()

		success := kd.cache.DeletePath(subCachePath...)
		glog.V(2).Infof("removeService %v at path %v. Success: %v",
			s.Name, subCachePath, success)

		// ExternalName services have no IP
		if kapi.IsServiceIPSet(s) {
			delete(kd.reverseRecordMap, s.Spec.ClusterIP)
			delete(kd.clusterIPServiceMap, s.Spec.ClusterIP)
		}
	}
}

func (kd *KubeDNS) updateService(oldObj, newObj interface{}) {
	if new, ok := assertIsService(newObj); ok {
		if old, ok := assertIsService(oldObj); ok {
			// Remove old cache path only if changing type to/from ExternalName.
			// In all other cases, we'll update records in place.
			if (new.Spec.Type == kapi.ServiceTypeExternalName) !=
				(old.Spec.Type == kapi.ServiceTypeExternalName) {
				kd.removeService(oldObj)
			}
			kd.newService(newObj)
		}
	}
}

func (kd *KubeDNS) handleEndpointAdd(obj interface{}) {
	if e, ok := obj.(*kapi.Endpoints); ok {
		kd.addDNSUsingEndpoints(e)
	}
}

func (kd *KubeDNS) addDNSUsingEndpoints(e *kapi.Endpoints) error {
	svc, err := kd.getServiceFromEndpoints(e)
	if err != nil {
		return err
	}
	if svc == nil || kapi.IsServiceIPSet(svc) {
		// No headless service found corresponding to endpoints object.
		return nil
	}
	return kd.generateRecordsForHeadlessService(e, svc)
}

func (kd *KubeDNS) getServiceFromEndpoints(e *kapi.Endpoints) (*kapi.Service, error) {
	key, err := kcache.MetaNamespaceKeyFunc(e)
	if err != nil {
		return nil, err
	}
	obj, exists, err := kd.servicesStore.GetByKey(key)
	if err != nil {
		return nil, fmt.Errorf("failed to get service object from services store - %v", err)
	}
	if !exists {
		glog.V(3).Infof("No service for endpoint %q in namespace %q",
			e.Name, e.Namespace)
		return nil, nil
	}
	if svc, ok := assertIsService(obj); ok {
		return svc, nil
	}
	return nil, fmt.Errorf("got a non service object in services store %v", obj)
}

// fqdn constructs the fqdn for the given service. subpaths is a list of path
// elements rooted at the given service, ending at a service record.
func (kd *KubeDNS) fqdn(service *kapi.Service, subpaths ...string) string {
	domainLabels := append(append(kd.domainPath, serviceSubdomain, service.Namespace, service.Name), subpaths...)
	return dns.Fqdn(strings.Join(util.ReverseArray(domainLabels), "."))
}

func (kd *KubeDNS) newPortalService(service *kapi.Service) {
	subCache := treecache.NewTreeCache()
	recordValue, recordLabel := util.GetSkyMsg(service.Spec.ClusterIP, 0)
	subCache.SetEntry(recordLabel, recordValue, kd.fqdn(service, recordLabel))

	// Generate SRV Records
	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		if port.Name != "" && port.Protocol != "" {
			srvValue := kd.generateSRVRecordValue(service, int(port.Port))

			l := []string{"_" + strings.ToLower(string(port.Protocol)), "_" + port.Name}
			glog.V(2).Infof("Added SRV record %+v", srvValue)

			subCache.SetEntry(recordLabel, srvValue, kd.fqdn(service, append(l, recordLabel)...), l...)
		}
	}
	subCachePath := append(kd.domainPath, serviceSubdomain, service.Namespace)
	host := getServiceFQDN(kd.domain, service)
	reverseRecord, _ := util.GetSkyMsg(host, 0)

	kd.cacheLock.Lock()
	defer kd.cacheLock.Unlock()
	kd.cache.SetSubCache(service.Name, subCache, subCachePath...)
	kd.reverseRecordMap[service.Spec.ClusterIP] = reverseRecord
	kd.clusterIPServiceMap[service.Spec.ClusterIP] = service
}

func (kd *KubeDNS) generateRecordsForHeadlessService(e *kapi.Endpoints, svc *kapi.Service) error {
	// TODO: remove this after v1.4 is released and the old annotations are EOL
	podHostnames, err := getPodHostnamesFromAnnotation(e.Annotations)
	if err != nil {
		return err
	}
	subCache := treecache.NewTreeCache()
	glog.V(4).Infof("Endpoints Annotations: %v", e.Annotations)
	for idx := range e.Subsets {
		for subIdx := range e.Subsets[idx].Addresses {
			address := &e.Subsets[idx].Addresses[subIdx]
			endpointIP := address.IP
			recordValue, endpointName := util.GetSkyMsg(endpointIP, 0)
			if hostLabel, exists := getHostname(address, podHostnames); exists {
				endpointName = hostLabel
			}
			subCache.SetEntry(endpointName, recordValue, kd.fqdn(svc, endpointName))
			for portIdx := range e.Subsets[idx].Ports {
				endpointPort := &e.Subsets[idx].Ports[portIdx]
				if endpointPort.Name != "" && endpointPort.Protocol != "" {
					srvValue := kd.generateSRVRecordValue(svc, int(endpointPort.Port), endpointName)
					glog.V(2).Infof("Added SRV record %+v", srvValue)

					l := []string{"_" + strings.ToLower(string(endpointPort.Protocol)), "_" + endpointPort.Name}
					subCache.SetEntry(endpointName, srvValue, kd.fqdn(svc, append(l, endpointName)...), l...)
				}
			}
		}
	}
	subCachePath := append(kd.domainPath, serviceSubdomain, svc.Namespace)
	kd.cacheLock.Lock()
	defer kd.cacheLock.Unlock()
	kd.cache.SetSubCache(svc.Name, subCache, subCachePath...)
	return nil
}

func getHostname(address *kapi.EndpointAddress, podHostnames map[string]endpoints.HostRecord) (string, bool) {
	if len(address.Hostname) > 0 {
		return address.Hostname, true
	}
	if hostRecord, exists := podHostnames[address.IP]; exists && len(validation.IsDNS1123Label(hostRecord.HostName)) == 0 {
		return hostRecord.HostName, true
	}
	return "", false
}

func getPodHostnamesFromAnnotation(annotations map[string]string) (map[string]endpoints.HostRecord, error) {
	hostnames := map[string]endpoints.HostRecord{}

	if annotations != nil {
		if serializedHostnames, exists := annotations[endpoints.PodHostnamesAnnotation]; exists && len(serializedHostnames) > 0 {
			err := json.Unmarshal([]byte(serializedHostnames), &hostnames)
			if err != nil {
				return nil, err
			}
		}
	}
	return hostnames, nil
}

func (kd *KubeDNS) generateSRVRecordValue(svc *kapi.Service, portNumber int, labels ...string) *skymsg.Service {
	host := strings.Join([]string{svc.Name, svc.Namespace, serviceSubdomain, kd.domain}, ".")
	for _, cNameLabel := range labels {
		host = cNameLabel + "." + host
	}
	recordValue, _ := util.GetSkyMsg(host, portNumber)
	return recordValue
}

// Generates skydns records for a headless service.
func (kd *KubeDNS) newHeadlessService(service *kapi.Service) error {
	// Create an A record for every pod in the service.
	// This record must be periodically updated.
	// Format is as follows:
	// For a service x, with pods a and b create DNS records,
	// a.x.ns.domain. and, b.x.ns.domain.
	key, err := kcache.MetaNamespaceKeyFunc(service)
	if err != nil {
		return err
	}
	e, exists, err := kd.endpointsStore.GetByKey(key)
	if err != nil {
		return fmt.Errorf("failed to get endpoints object from endpoints store - %v", err)
	}
	if !exists {
		glog.V(1).Infof("Could not find endpoints for service %q in namespace %q. DNS records will be created once endpoints show up.",
			service.Name, service.Namespace)
		return nil
	}
	if e, ok := e.(*kapi.Endpoints); ok {
		return kd.generateRecordsForHeadlessService(e, service)
	}
	return nil
}

// Generates skydns records for an ExternalName service.
func (kd *KubeDNS) newExternalNameService(service *kapi.Service) {
	// Create a CNAME record for the service's ExternalName.
	// TODO: TTL?
	recordValue, _ := util.GetSkyMsg(service.Spec.ExternalName, 0)
	cachePath := append(kd.domainPath, serviceSubdomain, service.Namespace)
	fqdn := kd.fqdn(service)
	glog.V(2).Infof("newExternalNameService: storing key %s with value %v as %s under %v",
		service.Name, recordValue, fqdn, cachePath)
	kd.cacheLock.Lock()
	defer kd.cacheLock.Unlock()
	// Store the service name directly as the leaf key
	kd.cache.SetEntry(service.Name, recordValue, fqdn, cachePath...)
}

// Records responds with DNS records that match the given name, in a format
// understood by the skydns server. If "exact" is true, a single record
// matching the given name is returned, otherwise all records stored under
// the subtree matching the name are returned.
func (kd *KubeDNS) Records(name string, exact bool) (retval []skymsg.Service, err error) {
	glog.V(3).Infof("Query for %q, exact: %v", name, exact)

	trimmed := strings.TrimRight(name, ".")
	segments := strings.Split(trimmed, ".")
	isFederationQuery := false
	federationSegments := []string{}

	if !exact && kd.isFederationQuery(segments) {
		glog.V(3).Infof("Received federation query, trying local service first")
		// Try querying the non-federation (local) service first. Will try
		// the federation one later, if this fails.
		isFederationQuery = true
		federationSegments = append(federationSegments, segments...)
		// To try local service, remove federation name from segments.
		// Federation name is 3rd in the segment (after service name and
		// namespace).
		segments = append(segments[:2], segments[3:]...)
	}

	path := util.ReverseArray(segments)
	records, err := kd.getRecordsForPath(path, exact)

	if err != nil {
		return nil, err
	}

	if isFederationQuery {
		return kd.recordsForFederation(records, path, exact, federationSegments)
	} else if len(records) > 0 {
		glog.V(4).Infof("Records for %v: %v", name, records)
		return records, nil
	}

	glog.V(3).Infof("No record found for %v", name)
	return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
}

func (kd *KubeDNS) recordsForFederation(records []skymsg.Service, path []string, exact bool, federationSegments []string) (retval []skymsg.Service, err error) {
	// For federation query, verify that the local service has endpoints.
	validRecord := false
	for _, val := range records {
		// We know that a headless service has endpoints for sure if a
		// record was returned for it. The record contains endpoint
		// IPs. So nothing to check for headless services.
		//
		// TODO: this access to the cluster IP map does not seem to be
		// threadsafe.
		if !kd.isHeadlessServiceRecord(&val) {
			ok, err := kd.serviceWithClusterIPHasEndpoints(&val)
			if err != nil {
				glog.V(2).Infof(
					"Federation: error finding if service has endpoint: %v", err)
				continue
			}
			if !ok {
				glog.V(2).Infof("Federation: skipping record since service has no endpoint: %v", val)
				continue
			}
		}
		validRecord = true
		break
	}

	if validRecord {
		// There is a local service with valid endpoints, return its CNAME.
		name := strings.Join(util.ReverseArray(path), ".")
		// Ensure that this name that we are returning as a CNAME response
		// is a fully qualified domain name so that the client's resolver
		// library doesn't have to go through its search list all over
		// again.
		if !strings.HasSuffix(name, ".") {
			name = name + "."
		}
		glog.V(3).Infof(
			"Federation: Returning CNAME for local service: %v", name)
		return []skymsg.Service{{Host: name}}, nil
	}

	// If the name query is not an exact query and does not match any
	// records in the local store, attempt to send a federation redirect
	// (CNAME) response.
	if !exact {
		glog.V(3).Infof(
			"Federation: Did not find a local service. Trying federation redirect (CNAME)")
		return kd.federationRecords(util.ReverseArray(federationSegments))
	}

	return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
}

func (kd *KubeDNS) getRecordsForPath(path []string, exact bool) ([]skymsg.Service, error) {
	if kd.isPodRecord(path) {
		ip, err := kd.getPodIP(path)
		if err == nil {
			skyMsg, _ := util.GetSkyMsg(ip, 0)
			return []skymsg.Service{*skyMsg}, nil
		}
		return nil, err
	}

	if exact {
		key := path[len(path)-1]
		if key == "" {
			return []skymsg.Service{}, nil
		}
		kd.cacheLock.RLock()
		defer kd.cacheLock.RUnlock()
		if record, ok := kd.cache.GetEntry(key, path[:len(path)-1]...); ok {
			glog.V(3).Infof("Exact match %v for %v received from cache", record, path[:len(path)-1])
			return []skymsg.Service{*(record.(*skymsg.Service))}, nil
		}

		glog.V(3).Infof("Exact match for %v not found in cache", path)
		return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
	}

	kd.cacheLock.RLock()
	defer kd.cacheLock.RUnlock()
	records := kd.cache.GetValuesForPathWithWildcards(path...)
	glog.V(3).Infof("Found %d records for %v in the cache", len(records), path)

	retval := []skymsg.Service{}
	for _, val := range records {
		retval = append(retval, *val)
	}

	glog.V(4).Infof("getRecordsForPath retval=%+v, path=%v", retval, path)

	return retval, nil
}

// Returns true if the given record corresponds to a headless service.
// Important: Assumes that we already have the cacheLock. Callers responsibility to acquire it.
// This is because the code will panic, if we try to acquire it again if we already have it.
func (kd *KubeDNS) isHeadlessServiceRecord(msg *skymsg.Service) bool {
	// If it is not a headless service, then msg.Host will be the cluster IP.
	// So we can check if msg.host exists in our clusterIPServiceMap.
	_, ok := kd.clusterIPServiceMap[msg.Host]
	// It is headless service if no record was found.
	return !ok
}

// Returns true if the service corresponding to the given message has endpoints.
// Note: Works only for services with ClusterIP. Will return an error for headless service (service without a clusterIP).
// Important: Assumes that we already have the cacheLock. Callers responsibility to acquire it.
// This is because the code will panic, if we try to acquire it again if we already have it.
func (kd *KubeDNS) serviceWithClusterIPHasEndpoints(msg *skymsg.Service) (bool, error) {
	svc, ok := kd.clusterIPServiceMap[msg.Host]
	if !ok {
		// It is a headless service.
		return false, fmt.Errorf("method not expected to be called for headless service")
	}
	key, err := kcache.MetaNamespaceKeyFunc(svc)
	if err != nil {
		return false, err
	}
	e, exists, err := kd.endpointsStore.GetByKey(key)
	if err != nil {
		return false, fmt.Errorf("failed to get endpoints object from endpoints store - %v", err)
	}
	if !exists {
		return false, nil
	}
	if e, ok := e.(*kapi.Endpoints); ok {
		return len(e.Subsets) > 0, nil
	}
	return false, fmt.Errorf("unexpected: found non-endpoint object in endpoint store: %v", e)
}

// ReverseRecords performs a reverse lookup for the given name.
func (kd *KubeDNS) ReverseRecord(name string) (*skymsg.Service, error) {
	glog.V(3).Infof("Query for ReverseRecord %q", name)

	// if portalIP is not a valid IP, the reverseRecordMap lookup will fail
	portalIP, ok := util.ExtractIP(name)
	if !ok {
		return nil, fmt.Errorf("does not support reverse lookup for %s", name)
	}

	kd.cacheLock.RLock()
	defer kd.cacheLock.RUnlock()
	if reverseRecord, ok := kd.reverseRecordMap[portalIP]; ok {
		return reverseRecord, nil
	}

	return nil, fmt.Errorf("must be exactly one service record")
}

// e.g {"local", "cluster", "pod", "default", "10-0-0-1"}
func (kd *KubeDNS) isPodRecord(path []string) bool {
	if len(path) != len(kd.domainPath)+3 {
		return false
	}
	if path[len(kd.domainPath)] != "pod" {
		return false
	}
	for _, segment := range path {
		if segment == "*" {
			return false
		}
	}
	return true
}

func (kd *KubeDNS) getPodIP(path []string) (string, error) {
	ipStr := path[len(path)-1]
	ip := strings.Replace(ipStr, "-", ".", -1)
	if parsed := net.ParseIP(ip); parsed != nil {
		return ip, nil
	}
	return "", fmt.Errorf("Invalid IP Address %v", ip)
}

// isFederationQuery checks if the given query `path` matches the federated service query pattern.
// The conjunction of the following conditions forms the test for the federated service query
// pattern:
//   1. `path` has exactly 4+len(domainPath) segments: mysvc.myns.myfederation.svc.domain.path.
//   2. Service name component must be a valid RFC 1035 name.
//   3. Namespace component must be a valid RFC 1123 name.
//   4. Federation component must also be a valid RFC 1123 name.
//   5. Fourth segment is exactly "svc"
//   6. The remaining segments match kd.domainPath.
//   7. And federation must be one of the listed federations in the config.
//   Note: Because of the above conditions, this method will treat wildcard queries such as
//   *.mysvc.myns.myfederation.svc.domain.path as non-federation queries.
//   We can add support for wildcard queries later, if needed.
func (kd *KubeDNS) isFederationQuery(path []string) bool {
	if len(path) != 4+len(kd.domainPath) {
		glog.V(4).Infof("Not a federation query: len(%q) != 4+len(%q)", path, kd.domainPath)
		return false
	}
	if errs := validation.IsDNS1035Label(path[0]); len(errs) != 0 {
		glog.V(4).Infof("Not a federation query: %q is not an RFC 1035 label: %q",
			path[0], errs)
		return false
	}
	if errs := validation.IsDNS1123Label(path[1]); len(errs) != 0 {
		glog.V(4).Infof("Not a federation query: %q is not an RFC 1123 label: %q",
			path[1], errs)
		return false
	}
	if errs := validation.IsDNS1123Label(path[2]); len(errs) != 0 {
		glog.V(4).Infof("Not a federation query: %q is not an RFC 1123 label: %q",
			path[2], errs)
		return false
	}
	if path[3] != serviceSubdomain {
		glog.V(4).Infof("Not a federation query: %q != %q (serviceSubdomain)",
			path[3], serviceSubdomain)
		return false
	}
	for i, domComp := range kd.domainPath {
		// kd.domainPath is reversed, so we need to look in the `path` in the reverse order.
		if domComp != path[len(path)-i-1] {
			glog.V(4).Infof("Not a federation query: kd.domainPath[%d] != path[%d] (%q != %q)",
				i, len(path)-i-1, domComp, path[len(path)-i-1])
			return false
		}
	}

	kd.configLock.RLock()
	defer kd.configLock.RUnlock()

	if _, ok := kd.config.Federations[path[2]]; !ok {
		glog.V(4).Infof("Not a federation query: label %q not found", path[2])
		return false
	}

	return true
}

// federationRecords checks if the given `queryPath` is for a federated service and if it is,
// it returns a CNAME response containing the cluster zone name and federation domain name
// suffix.
func (kd *KubeDNS) federationRecords(queryPath []string) ([]skymsg.Service, error) {
	// `queryPath` is a reversed-array of the queried name, reverse it back to make it easy
	// to follow through this code and reduce confusion. There is no reason for it to be
	// reversed here.
	path := util.ReverseArray(queryPath)

	// Check if the name query matches the federation query pattern.
	if !kd.isFederationQuery(path) {
		return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
	}

	// Now that we have already established that the query is a federation query, remove the local
	// domain path components, i.e. kd.domainPath, from the query.
	path = path[:len(path)-len(kd.domainPath)]

	// Append the zone name (zone in the cloud provider terminology, not a DNS
	// zone) and the region name.
	zone, region, err := kd.getClusterZoneAndRegion()
	if err != nil {
		return nil, fmt.Errorf("failed to obtain the cluster zone and region: %v", err)
	}
	path = append(path, zone, region)

	// We have already established that the map entry exists for the given federation,
	// we just need to retrieve the domain name, validate it and append it to the path.
	kd.configLock.RLock()
	domain := kd.config.Federations[path[2]]
	kd.configLock.RUnlock()

	// We accept valid subdomains as well, so just let all the valid subdomains.
	if len(validation.IsDNS1123Subdomain(domain)) != 0 {
		return nil, fmt.Errorf("%s is not a valid domain name for federation %s", domain, path[2])
	}
	name := strings.Join(append(path, domain), ".")

	// Ensure that this name that we are returning as a CNAME response is a fully qualified
	// domain name so that the client's resolver library doesn't have to go through its
	// search list all over again.
	if !strings.HasSuffix(name, ".") {
		name = name + "."
	}
	return []skymsg.Service{{Host: name}}, nil
}

// getClusterZoneAndRegion returns the name of the zone and the region the
// cluster is running in. It arbitrarily selects a node and reads the failure
// domain label on the node. An alternative is to obtain this pod's
// (i.e. kube-dns pod's) name using the downward API, get the pod, get the
// node the pod is bound to and retrieve that node's labels. But even just by
// reading those steps, it looks complex and it is not entirely clear what
// that complexity is going to buy us. So taking a simpler approach here.
// Also note that zone here means the zone in cloud provider terminology, not
// the DNS zone.
func (kd *KubeDNS) getClusterZoneAndRegion() (string, string, error) {
	var node *kapi.Node

	objs := kd.nodesStore.List()
	if len(objs) > 0 {
		var ok bool
		if node, ok = objs[0].(*kapi.Node); !ok {
			return "", "", fmt.Errorf("expected node object, got: %T", objs[0])
		}
	} else {
		// An alternative to listing nodes each time is to set a watch, but that is totally
		// wasteful in case of non-federated independent Kubernetes clusters. So carefully
		// proceeding here.
		// TODO(madhusudancs): Move this to external/v1 API.
		nodeList, err := kd.kubeClient.Core().Nodes().List(kapi.ListOptions{})
		if err != nil || len(nodeList.Items) == 0 {
			return "", "", fmt.Errorf("failed to retrieve the cluster nodes: %v", err)
		}

		// Select a node (arbitrarily the first node) that has
		// `LabelZoneFailureDomain` and `LabelZoneRegion` set.
		for _, nodeItem := range nodeList.Items {
			_, zfound := nodeItem.Labels[unversioned.LabelZoneFailureDomain]
			_, rfound := nodeItem.Labels[unversioned.LabelZoneRegion]
			if !zfound || !rfound {
				continue
			}
			// Make a copy of the node, don't rely on the loop variable.
			node = &(*(&nodeItem))
			if err := kd.nodesStore.Add(node); err != nil {
				return "", "", fmt.Errorf("couldn't add the retrieved node to the cache: %v", err)
			}
			// Node is found, break out of the loop.
			break
		}
	}

	if node == nil {
		return "", "", fmt.Errorf("Could not find any nodes")
	}

	zone, ok := node.Labels[unversioned.LabelZoneFailureDomain]
	if !ok || zone == "" {
		return "", "", fmt.Errorf("unknown cluster zone")
	}
	region, ok := node.Labels[unversioned.LabelZoneRegion]
	if !ok || region == "" {
		return "", "", fmt.Errorf("unknown cluster region")
	}
	return zone, region, nil
}

func getServiceFQDN(domain string, service *kapi.Service) string {
	return strings.Join(
		[]string{service.Name, service.Namespace, serviceSubdomain, domain}, ".")
}
