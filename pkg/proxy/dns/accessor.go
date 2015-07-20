package dns

import (
	"fmt"
	"net"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"

	skymsg "github.com/skynetservices/skydns/msg"
)

type DNSAccessor interface {
	GetByService(string, string) ([]skymsg.Service, error)
	GetByPath(string) ([]skymsg.Service, error)
	GetByIP(string) (*skymsg.Service, error)
	GetByNamespace(string) ([]skymsg.Service, error)
	GetAll() ([]skymsg.Service, error)
}

type dnsStoreAccessor struct {
	*dnsStore
}

// dns accessor
var _ DNSAccessor = &dnsStoreAccessor{}

func NewDNSAccessor(client *client.Client, stopCh <-chan struct{}) DNSAccessor {
	serviceWatcher := cache.NewListWatchFromClient(client, "services", api.NamespaceAll, fields.Everything())
	endpointsWatcher := cache.NewListWatchFromClient(client, "endpoints", api.NamespaceAll, fields.Everything())

	// TODO: do we want to index by name as well (for wildcard lookups)?
	entryStore := cache.NewBulkIndexer(dnsEntryKeyFunc, cache.Indexers{
		"clusterIP": indexServiceByClusterIP, // for reverse lookups
		"namespace": namespaceIndexFunc,
		"service":   serviceIndexFunc,
		"path":      pathIndexFunc,
	})

	mainStore := &dnsStore{
		BulkIndexer: entryStore,
		endpoints:   client,
		isHeadless:  make(map[string]bool),
	}
	endpointsStore := &endpointsListener{mainStore}

	// 30 seconds is the same amount used by the main part of the proxy
	serviceReflector := cache.NewReflector(serviceWatcher, &api.Service{}, mainStore, 30*time.Second)

	// don't get full bulk updates, just incremental onces
	endpointsReflector := cache.NewReflector(endpointsWatcher, &api.Endpoints{}, endpointsStore, 0)

	if stopCh != nil {
		serviceReflector.RunUntil(stopCh)
		endpointsReflector.RunUntil(stopCh)
	} else {
		serviceReflector.Run()
		endpointsReflector.Run()
	}

	return &dnsStoreAccessor{mainStore}
}

func dnsEntryKeyFunc(obj interface{}) (string, error) {
	entry, ok := obj.(*dnsEntry)
	if !ok {
		return "", fmt.Errorf("The key function only accepts dnsEntry structs")
	}
	return fmt.Sprintf("%s:%d", entry.Host, entry.Port), nil
}

func pathIndexFunc(obj interface{}) (string, error) {
	entry, ok := obj.(*dnsEntry)
	if !ok {
		return "", fmt.Errorf("The path index function only accepts dnsEntry structs")
	}
	return entry.Key, nil
}

func namespaceIndexFunc(obj interface{}) (string, error) {
	entry, ok := obj.(*dnsEntry)
	if !ok {
		return "", fmt.Errorf("The namespace index function only accepts dnsEntry structs")
	}
	return entry.Namespace, nil
}

func serviceIndexFunc(obj interface{}) (string, error) {
	if entry, ok := obj.(*dnsEntry); ok {
		return entry.Namespace + "/" + entry.Name, nil
	} else {
		return cache.MetaNamespaceKeyFunc(obj)
	}
}

func indexServiceByClusterIP(obj interface{}) (string, error) {
	entry, ok := obj.(*dnsEntry)
	if !ok {
		return "", fmt.Errorf("The IP index function only accepts dnsEntry structs")
	}
	if ip := net.ParseIP(entry.Host); ip != nil {
		return entry.Host, nil
	} else {
		return "", nil
	}
}

func entriesToServices(entries []interface{}) []skymsg.Service {
	if len(entries) == 0 {
		return nil
	}
	services := make([]skymsg.Service, len(entries))
	for i, entry := range entries {
		services[i] = entry.(*dnsEntry).Service
	}

	return services
}

func (store *dnsStoreAccessor) GetByService(namespace string, name string) ([]skymsg.Service, error) {
	entries, err := store.Index("service", &dnsEntry{Namespace: namespace, Name: name})
	if err != nil {
		return nil, err
	}
	return entriesToServices(entries), nil
}

func (store *dnsStoreAccessor) GetByIP(ip string) (*skymsg.Service, error) {
	entries, err := store.Index("IP", &dnsEntry{Service: skymsg.Service{Host: ip}})
	if err != nil {
		return nil, err
	}

	if len(entries) == 0 {
		return nil, nil
	}

	return &entries[0].(*dnsEntry).Service, nil
}

func (store *dnsStoreAccessor) GetByNamespace(namespace string) ([]skymsg.Service, error) {
	entries, err := store.Index("namespace", &dnsEntry{Namespace: namespace})
	if err != nil {
		return nil, err
	}
	return entriesToServices(entries), nil
}

func (store *dnsStoreAccessor) GetAll() ([]skymsg.Service, error) {
	entries := store.List()
	return entriesToServices(entries), nil
}

func (store *dnsStoreAccessor) GetByPath(path string) ([]skymsg.Service, error) {
	entries, err := store.Index("path", &dnsEntry{Service: skymsg.Service{Key: path}})
	if err != nil {
		return nil, err
	}
	return entriesToServices(entries), nil
}
