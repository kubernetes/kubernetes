package dns

import (
	"fmt"
	"hash/fnv"
	"strings"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"

	skymsg "github.com/skynetservices/skydns/msg"
)

type dnsStore struct {
	cache.BulkIndexer
	endpoints  client.EndpointsNamespacer
	headlessMu sync.Mutex
	isHeadless map[string]bool
}

type dnsEntry struct {
	Namespace string
	Name      string
	skymsg.Service
}

var _ cache.Store = &dnsStore{}

func makeSkyDNSService(host string, port int, key string, group string) skymsg.Service {
	return skymsg.Service{
		Host: host,
		Port: port,

		Priority: 10,
		Weight:   10,
		Ttl:      30,

		Text: "",
		Key:  key,

		Group: group,
	}
}

func makeEntriesForService(service *api.Service) []interface{} {
	entries := make([]interface{}, len(service.Spec.Ports)+1)
	srvPath := fmt.Sprintf(srvPathFormat, service.Name, service.Namespace)
	for i, port := range service.Spec.Ports {
		portPath := fmt.Sprintf(srvPortPathFormat, port.Name, strings.ToLower(string(port.Protocol)), service.Name, service.Namespace)
		entries[i] = &dnsEntry{
			Name:      service.Name,
			Namespace: service.Namespace,
			Service:   makeSkyDNSService(srvPath, port.Port, skymsg.Path(portPath), "srv"),
		}
	}

	entries[len(entries)-1] = &dnsEntry{
		Name:      service.Name,
		Namespace: service.Namespace,
		Service:   makeSkyDNSService(service.Spec.ClusterIP, 0, skymsg.Path(srvPath), "a"),
	}

	return entries
}

func getHash(text string) string {
	h := fnv.New32a()
	h.Write([]byte(text))
	return fmt.Sprintf("%x", h.Sum32())
}

func makeEntriesForEndpoints(endpoints *api.Endpoints) []interface{} {
	if len(endpoints.Subsets) == 0 {
		return make([]interface{}, 0)
	}

	entries := make([]interface{}, 0, len(endpoints.Subsets)*(len(endpoints.Subsets[0].Addresses)*len(endpoints.Subsets[0].Ports)+1))

	for _, subset := range endpoints.Subsets {
		for _, addr := range subset.Addresses {
			hsh := getHash(addr.IP)
			srvPath := fmt.Sprintf(headlessSrvPathFormat, hsh, endpoints.Name, endpoints.Namespace)
			for _, port := range subset.Ports {
				portPath := fmt.Sprintf(srvPortPathFormat, port.Name, strings.ToLower(string(port.Protocol)), endpoints.Name, endpoints.Namespace)

				entries = append(entries, &dnsEntry{
					Name:      endpoints.Name,
					Namespace: endpoints.Namespace,
					Service:   makeSkyDNSService(srvPath, port.Port, skymsg.Path(portPath), "srv"),
				})
			}

			entries = append(entries, &dnsEntry{
				Name:      endpoints.Name,
				Namespace: endpoints.Namespace,
				Service:   makeSkyDNSService(addr.IP, 0, skymsg.Path(srvPath), "a"),
			})
		}
	}

	return entries
}

func (store *dnsStore) Add(obj interface{}) error {
	service := obj.(*api.Service)

	store.headlessMu.Lock()
	defer store.headlessMu.Unlock()
	if api.IsServiceIPSet(service) {
		store.isHeadless[service.Namespace+"/"+service.Name] = false
		return store.BulkAdd(makeEntriesForService(service))
	} else {
		store.isHeadless[service.Namespace+"/"+service.Name] = true
		endpoints, err := store.endpoints.Endpoints(service.Namespace).Get(service.Name)
		if err != nil {
			return err
		}
		return store.BulkIndexer.BulkAdd(makeEntriesForEndpoints(endpoints))
	}
}

func (store *dnsStore) Replace(list []interface{}) error {
	// fetch the endpoints, too
	endpointSets, err := store.endpoints.Endpoints(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		return err
	}
	endpointsMap := make(map[string]*api.Endpoints)
	for i, endpoints := range endpointSets.Items {
		endpointsMap[endpoints.Namespace+"/"+endpoints.Name] = &endpointSets.Items[i]
	}

	store.headlessMu.Lock()
	defer store.headlessMu.Unlock()
	store.isHeadless = map[string]bool{}

	items := make([]interface{}, 0)
	for _, rawService := range list {
		service := rawService.(*api.Service)
		if api.IsServiceIPSet(service) {
			items = append(items, makeEntriesForService(service)...)
		} else {
			store.isHeadless[service.Namespace+"/"+service.Name] = true
			if endpoints, ok := endpointsMap[service.Namespace+"/"+service.Name]; ok {
				items = append(items, makeEntriesForEndpoints(endpoints)...)
			} else {
				glog.V(2).Infof("No endpoints for service %s/%s", service.Name, service.Namespace)
			}
		}
	}

	return store.BulkIndexer.Replace(items)
}

func (store *dnsStore) Update(obj interface{}) error {
	service := obj.(*api.Service)

	store.headlessMu.Lock()
	defer store.headlessMu.Unlock()

	if api.IsServiceIPSet(service) {
		return store.ReplaceByIndex("service", service, makeEntriesForService(service))
	} else {
		store.isHeadless[service.Namespace+"/"+service.Name] = true
		endpoints, err := store.endpoints.Endpoints(service.Namespace).Get(service.Name)
		if err != nil {
			return err
		}
		return store.ReplaceByIndex("service", service, makeEntriesForEndpoints(endpoints))
	}
}

func (store *dnsStore) Delete(obj interface{}) error {
	defer store.headlessMu.Unlock()
	store.isHeadless = map[string]bool{}
	service := obj.(*api.Service)
	delete(store.isHeadless, service.Namespace+"/"+service.Name)
	return store.DeleteByIndex("service", service)
}

type endpointsListener struct {
	entries *dnsStore
}

var _ cache.Store = &endpointsListener{}

// we only listen for specific changes in between updates to the service list

func (listener *endpointsListener) Add(obj interface{}) error {
	listener.entries.headlessMu.Lock()
	defer listener.entries.headlessMu.Unlock()
	endpoints := obj.(*api.Endpoints)
	if listener.entries.isHeadless[endpoints.Namespace+"/"+endpoints.Name] {
		return listener.entries.BulkAdd(makeEntriesForEndpoints(obj.(*api.Endpoints)))
	} else {
		return nil
	}
}

func (listener *endpointsListener) Update(obj interface{}) error {
	listener.entries.headlessMu.Lock()
	defer listener.entries.headlessMu.Unlock()
	endpoints := obj.(*api.Endpoints)
	if listener.entries.isHeadless[endpoints.Namespace+"/"+endpoints.Name] {
		//return listener.entries.ReplaceByIndex("service", endpoints, makeEntriesForEndpoints(endpoints))
		entries := makeEntriesForEndpoints(endpoints)
		return listener.entries.ReplaceByIndex("service", endpoints, entries)
	} else {
		return nil
	}
}

func (listener *endpointsListener) Delete(obj interface{}) error {
	listener.entries.headlessMu.Lock()
	defer listener.entries.headlessMu.Unlock()
	endpoints := obj.(*api.Endpoints)
	if listener.entries.isHeadless[endpoints.Namespace+"/"+endpoints.Name] {
		return listener.entries.DeleteByIndex("service", endpoints)
	} else {
		return nil
	}
}

// Not implemented (we don't need these

func (listener *endpointsListener) List() []interface{} { return nil }
func (listener *endpointsListener) ListKeys() []string  { return nil }
func (listener *endpointsListener) Get(obj interface{}) (item interface{}, exists bool, err error) {
	return nil, false, fmt.Errorf("not implemented")
}
func (listener *endpointsListener) GetByKey(key string) (item interface{}, exists bool, err error) {
	return nil, false, fmt.Errorf("not implemented")
}
func (listener *endpointsListener) Replace([]interface{}) error {
	// we can't return "not implemented" or else it will keep trying
	return nil
}
