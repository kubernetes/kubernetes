/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"hash/fnv"
	"net"
	"strings"
	"sync"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	skymsg "github.com/skynetservices/skydns/msg"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kframework "k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	kubernetesSvcName = "kubernetes"

	// A subdomain added to the user specified domain for all services.
	serviceSubdomain = "svc"

	// A subdomain added to the user specified dmoain for all pods.
	podSubdomain = "pod"

	// arpaSuffix is the standard suffix for PTR IP reverse lookups.
	arpaSuffix = ".in-addr.arpa."

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

	// The domain for which this DNS Server is authoritative.
	domain string

	// A cache that contains all the endpoints in the system.
	endpointsStore kcache.Store

	// A cache that contains all the services in the system.
	servicesStore kcache.Store

	// stores DNS records for the domain.
	// A Records and SRV Records for (regular) services and headless Services.
	cache *TreeCache

	reverseRecordMap map[string]*skymsg.Service

	// caller is responsible for using the cacheLock before invoking methods on cache
	// the cache is not thread-safe, and the caller can guarantee thread safety by using
	// the cacheLock
	cacheLock sync.RWMutex

	// The domain for which this DNS Server is authoritative, in array format and reversed.
	// e.g. if domain is "cluster.local", domainPath is []string{"local", "cluster"}
	domainPath []string

	// endpointsController  invokes registered callbacks when endpoints change.
	endpointsController *kframework.Controller

	// serviceController invokes registered callbacks when services change.
	serviceController *kframework.Controller

	// Map of federation names that the cluster in which this kube-dns is running belongs to, to
	// the corresponding domain names.
	federations map[string]string

	// A TTL cache that contains some subset of nodes in the system so that we can retrieve the
	// cluster zone annotation from the cached node instead of getting it from the API server
	// every time.
	nodesStore kcache.Store
}

func NewKubeDNS(client clientset.Interface, domain string, federations map[string]string) *KubeDNS {
	kd := &KubeDNS{
		kubeClient:       client,
		domain:           domain,
		cache:            NewTreeCache(),
		cacheLock:        sync.RWMutex{},
		reverseRecordMap: make(map[string]*skymsg.Service),
		domainPath:       reverseArray(strings.Split(strings.TrimRight(domain, "."), ".")),
		federations:      federations,
	}
	kd.setEndpointsStore()
	kd.setServicesStore()
	return kd
}

func (kd *KubeDNS) Start() {
	go kd.endpointsController.Run(wait.NeverStop)
	go kd.serviceController.Run(wait.NeverStop)
	// Wait synchronously for the Kubernetes service and add a DNS record for it.
	// This ensures that the Start function returns only after having received Service objects
	// from APIServer.
	// TODO: we might not have to wait for kubernetes service specifically. We should just wait
	// for a list operation to be complete from APIServer.
	kd.waitForKubernetesService()
}

func (kd *KubeDNS) waitForKubernetesService() (svc *kapi.Service) {
	name := fmt.Sprintf("%v/%v", kapi.NamespaceDefault, kubernetesSvcName)
	glog.Infof("Waiting for service: %v", name)
	var err error
	servicePollInterval := 1 * time.Second
	for {
		svc, err = kd.kubeClient.Core().Services(kapi.NamespaceDefault).Get(kubernetesSvcName)
		if err != nil || svc == nil {
			glog.Infof("Ignoring error while waiting for service %v: %v. Sleeping %v before retrying.", name, err, servicePollInterval)
			time.Sleep(servicePollInterval)
			continue
		}
		break
	}
	return
}

func (kd *KubeDNS) GetCacheAsJSON() (string, error) {
	kd.cacheLock.RLock()
	defer kd.cacheLock.RUnlock()
	json, err := kd.cache.Serialize()
	return json, err
}

func (kd *KubeDNS) setServicesStore() {
	// Returns a cache.ListWatch that gets all changes to services.
	kd.servicesStore, kd.serviceController = kframework.NewInformer(
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
		kframework.ResourceEventHandlerFuncs{
			AddFunc:    kd.newService,
			DeleteFunc: kd.removeService,
			UpdateFunc: kd.updateService,
		},
	)
}

func (kd *KubeDNS) setEndpointsStore() {
	// Returns a cache.ListWatch that gets all changes to endpoints.
	kd.endpointsStore, kd.endpointsController = kframework.NewInformer(
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
		kframework.ResourceEventHandlerFuncs{
			AddFunc: kd.handleEndpointAdd,
			UpdateFunc: func(oldObj, newObj interface{}) {
				// TODO: Avoid unwanted updates.
				kd.handleEndpointAdd(newObj)
			},
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
		// if ClusterIP is not set, a DNS entry should not be created
		if !kapi.IsServiceIPSet(service) {
			kd.newHeadlessService(service)
			return
		}
		if len(service.Spec.Ports) == 0 {
			glog.Warning("Unexpected service with no ports, this should not have happend: %v", service)
		}
		kd.newPortalService(service)
	}
}

func (kd *KubeDNS) removeService(obj interface{}) {
	if s, ok := assertIsService(obj); ok {
		subCachePath := append(kd.domainPath, serviceSubdomain, s.Namespace, s.Name)
		kd.cacheLock.Lock()
		defer kd.cacheLock.Unlock()
		kd.cache.deletePath(subCachePath...)
		delete(kd.reverseRecordMap, s.Spec.ClusterIP)
	}
}

func (kd *KubeDNS) updateService(oldObj, newObj interface{}) {
	kd.newService(newObj)
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
		glog.V(1).Infof("could not find service for endpoint %q in namespace %q", e.Name, e.Namespace)
		return nil, nil
	}
	if svc, ok := assertIsService(obj); ok {
		return svc, nil
	}
	return nil, fmt.Errorf("got a non service object in services store %v", obj)
}

func (kd *KubeDNS) newPortalService(service *kapi.Service) {
	subCache := NewTreeCache()
	recordValue, recordLabel := getSkyMsg(service.Spec.ClusterIP, 0)
	subCache.setEntry(recordLabel, recordValue)

	// Generate SRV Records
	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		if port.Name != "" && port.Protocol != "" {
			srvValue := kd.generateSRVRecordValue(service, int(port.Port))
			subCache.setEntry(recordLabel, srvValue, "_"+strings.ToLower(string(port.Protocol)), "_"+port.Name)
		}
	}
	subCachePath := append(kd.domainPath, serviceSubdomain, service.Namespace)
	host := kd.getServiceFQDN(service)
	reverseRecord, _ := getSkyMsg(host, 0)

	kd.cacheLock.Lock()
	defer kd.cacheLock.Unlock()
	kd.cache.setSubCache(service.Name, subCache, subCachePath...)
	kd.reverseRecordMap[service.Spec.ClusterIP] = reverseRecord
}

func (kd *KubeDNS) generateRecordsForHeadlessService(e *kapi.Endpoints, svc *kapi.Service) error {
	// TODO: remove this after v1.4 is released and the old annotations are EOL
	podHostnames, err := getPodHostnamesFromAnnotation(e.Annotations)
	if err != nil {
		return err
	}
	subCache := NewTreeCache()
	glog.V(4).Infof("Endpoints Annotations: %v", e.Annotations)
	for idx := range e.Subsets {
		for subIdx := range e.Subsets[idx].Addresses {
			address := &e.Subsets[idx].Addresses[subIdx]
			endpointIP := address.IP
			recordValue, endpointName := getSkyMsg(endpointIP, 0)
			if hostLabel, exists := getHostname(address, podHostnames); exists {
				endpointName = hostLabel
			}
			subCache.setEntry(endpointName, recordValue)
			for portIdx := range e.Subsets[idx].Ports {
				endpointPort := &e.Subsets[idx].Ports[portIdx]
				if endpointPort.Name != "" && endpointPort.Protocol != "" {
					srvValue := kd.generateSRVRecordValue(svc, int(endpointPort.Port), endpointName)
					subCache.setEntry(endpointName, srvValue, "_"+strings.ToLower(string(endpointPort.Protocol)), "_"+endpointPort.Name)
				}
			}
		}
	}
	subCachePath := append(kd.domainPath, serviceSubdomain, svc.Namespace)
	kd.cacheLock.Lock()
	defer kd.cacheLock.Unlock()
	kd.cache.setSubCache(svc.Name, subCache, subCachePath...)
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
	recordValue, _ := getSkyMsg(host, portNumber)
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
		glog.V(1).Infof("Could not find endpoints for service %q in namespace %q. DNS records will be created once endpoints show up.", service.Name, service.Namespace)
		return nil
	}
	if e, ok := e.(*kapi.Endpoints); ok {
		return kd.generateRecordsForHeadlessService(e, service)
	}
	return nil
}

func (kd *KubeDNS) Records(name string, exact bool) ([]skymsg.Service, error) {
	glog.Infof("Received DNS Request:%s, exact:%v", name, exact)
	trimmed := strings.TrimRight(name, ".")
	segments := strings.Split(trimmed, ".")
	path := reverseArray(segments)
	if kd.isPodRecord(path) {
		ip, err := kd.getPodIP(path)
		if err == nil {
			skyMsg, _ := getSkyMsg(ip, 0)
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
		if record, ok := kd.cache.getEntry(key, path[:len(path)-1]...); ok {
			return []skymsg.Service{*(record.(*skymsg.Service))}, nil
		}
		return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
	}

	kd.cacheLock.RLock()
	defer kd.cacheLock.RUnlock()
	records := kd.cache.getValuesForPathWithWildcards(path...)
	retval := []skymsg.Service{}
	for _, val := range records {
		retval = append(retval, *(val.(*skymsg.Service)))
	}
	glog.Infof("records:%v, retval:%v, path:%v", records, retval, path)
	if len(retval) > 0 {
		return retval, nil
	}

	// If the name query is not an exact query and does not match any records in the local store,
	// attempt to send a federation redirect (CNAME) response.
	if !exact {
		return kd.federationRecords(path)
	}

	return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
}

func (kd *KubeDNS) ReverseRecord(name string) (*skymsg.Service, error) {
	glog.Infof("Received ReverseRecord Request:%s", name)

	// if portalIP is not a valid IP, the reverseRecordMap lookup will fail
	portalIP, ok := extractIP(name)
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

// extractIP turns a standard PTR reverse record lookup name
// into an IP address
func extractIP(reverseName string) (string, bool) {
	if !strings.HasSuffix(reverseName, arpaSuffix) {
		return "", false
	}
	search := strings.TrimSuffix(reverseName, arpaSuffix)

	// reverse the segments and then combine them
	segments := reverseArray(strings.Split(search, "."))
	return strings.Join(segments, "."), true
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

// Returns record in a format that SkyDNS understands.
// Also return the hash of the record.
func getSkyMsg(ip string, port int) (*skymsg.Service, string) {
	msg := &skymsg.Service{
		Host:     ip,
		Port:     port,
		Priority: 10,
		Weight:   10,
		Ttl:      30,
	}
	s := fmt.Sprintf("%v", msg)
	h := fnv.New32a()
	h.Write([]byte(s))
	hash := fmt.Sprintf("%x", h.Sum32())
	glog.Infof("DNS Record:%s, hash:%s", s, hash)
	return msg, fmt.Sprintf("%x", hash)
}

// isFederationQuery checks if the given query `path` matches the federated service query pattern.
// The conjunction of the following conditions forms the test for the federated service query
// pattern:
//   1. `path` has exactly 4+len(domainPath) segments: mysvc.myns.myfederation.svc.domain.path.
//   2. Service name component must be a valid RFC 952 name.
//   3. Namespace component must be a valid RFC 1123 name.
//   4. Federation component must also be a valid RFC 1123 name.
//   5. Fourth segment is exactly "svc"
//   6. The remaining segments match kd.domainPath.
//   7. And federation must be one of the listed federations in the config.
func (kd *KubeDNS) isFederationQuery(path []string) bool {
	if len(path) == 4+len(kd.domainPath) &&
		len(validation.IsDNS952Label(path[0])) == 0 &&
		len(validation.IsDNS1123Label(path[1])) == 0 &&
		len(validation.IsDNS1123Label(path[2])) == 0 &&
		path[3] == serviceSubdomain {
		for i, domComp := range kd.domainPath {
			// kd.domainPath is reversed, so we need to look in the `path` in the reverse order.
			if domComp != path[len(path)-i-1] {
				return false
			}
		}
		if _, ok := kd.federations[path[2]]; ok {
			return true
		}
	}
	return false
}

// federationRecords checks if the given `queryPath` is for a federated service and if it is,
// it returns a CNAME response containing the cluster zone name and federation domain name
// suffix.
func (kd *KubeDNS) federationRecords(queryPath []string) ([]skymsg.Service, error) {
	// `queryPath` is a reversed-array of the queried name, reverse it back to make it easy
	// to follow through this code and reduce confusion. There is no reason for it to be
	// reversed here.
	path := reverseArray(queryPath)

	// Check if the name query matches the federation query pattern.
	if !kd.isFederationQuery(path) {
		return nil, etcd.Error{Code: etcd.ErrorCodeKeyNotFound}
	}

	// Now that we have already established that the query is a federation query, remove the local
	// domain path components, i.e. kd.domainPath, from the query.
	path = path[:len(path)-len(kd.domainPath)]

	// Append the zone name (zone in the cloud provider terminology, not a DNS zone)
	zone, err := kd.getClusterZone()
	if err != nil {
		return nil, fmt.Errorf("failed to obtain the cluster zone: %v", err)
	}
	path = append(path, zone)

	// We have already established that the map entry exists for the given federation,
	// we just need to retrieve the domain name, validate it and append it to the path.
	domain := kd.federations[path[2]]
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

// getClusterZone returns the name of the zone the cluster is running in. It arbitrarily selects
// a node and reads the failure domain annotation on the node. An alternative is to obtain this
// pod's (i.e. kube-dns pod's) name using the downward API, get the pod, get the node the pod is
// bound to and retrieve that node's annotations. But even just by reading those steps, it looks
// complex and it is not entirely clear what that complexity is going to buy us. So taking a
// simpler approach here.
// Also note that zone here means the zone in cloud provider terminology, not the DNS zone.
func (kd *KubeDNS) getClusterZone() (string, error) {
	var node *kapi.Node

	objs := kd.nodesStore.List()
	if len(objs) > 0 {
		var ok bool
		if node, ok = objs[0].(*kapi.Node); !ok {
			return "", fmt.Errorf("expected node object, got: %T", objs[0])
		}
	} else {
		// An alternative to listing nodes each time is to set a watch, but that is totally
		// wasteful in case of non-federated independent Kubernetes clusters. So carefully
		// proceeding here.
		// TODO(madhusudancs): Move this to external/v1 API.
		nodeList, err := kd.kubeClient.Core().Nodes().List(kapi.ListOptions{})
		if err != nil || len(nodeList.Items) == 0 {
			return "", fmt.Errorf("failed to retrieve the cluster nodes: %v", err)
		}

		// Select a node (arbitrarily the first node) that has `LabelZoneFailureDomain` set.
		for _, nodeItem := range nodeList.Items {
			if _, ok := nodeItem.Annotations[unversioned.LabelZoneFailureDomain]; !ok {
				continue
			}
			// Make a copy of the node, don't rely on the loop variable.
			node = &(*(&nodeItem))
			if err := kd.nodesStore.Add(node); err != nil {
				return "", fmt.Errorf("couldn't add the retrieved node to the cache: %v", err)
			}
		}
	}

	zone, ok := node.Annotations[unversioned.LabelZoneFailureDomain]
	if !ok || zone == "" {
		return "", fmt.Errorf("unknown cluster zone")
	}
	return zone, nil
}

func (kd *KubeDNS) getServiceFQDN(service *kapi.Service) string {
	return strings.Join([]string{service.Name, service.Namespace, serviceSubdomain, kd.domain}, ".")
}

func reverseArray(arr []string) []string {
	for i := 0; i < len(arr)/2; i++ {
		j := len(arr) - i - 1
		arr[i], arr[j] = arr[j], arr[i]
	}
	return arr
}
