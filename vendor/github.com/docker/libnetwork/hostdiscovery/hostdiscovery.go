package hostdiscovery

import (
	"net"
	"sync"

	"github.com/sirupsen/logrus"

	mapset "github.com/deckarep/golang-set"
	"github.com/docker/docker/pkg/discovery"
	// Including KV
	_ "github.com/docker/docker/pkg/discovery/kv"
	"github.com/docker/libkv/store/consul"
	"github.com/docker/libkv/store/etcd"
	"github.com/docker/libkv/store/zookeeper"
	"github.com/docker/libnetwork/types"
)

type hostDiscovery struct {
	watcher  discovery.Watcher
	nodes    mapset.Set
	stopChan chan struct{}
	sync.Mutex
}

func init() {
	consul.Register()
	etcd.Register()
	zookeeper.Register()
}

// NewHostDiscovery function creates a host discovery object
func NewHostDiscovery(watcher discovery.Watcher) HostDiscovery {
	return &hostDiscovery{watcher: watcher, nodes: mapset.NewSet(), stopChan: make(chan struct{})}
}

func (h *hostDiscovery) Watch(activeCallback ActiveCallback, joinCallback JoinCallback, leaveCallback LeaveCallback) error {
	h.Lock()
	d := h.watcher
	h.Unlock()
	if d == nil {
		return types.BadRequestErrorf("invalid discovery watcher")
	}
	discoveryCh, errCh := d.Watch(h.stopChan)
	go h.monitorDiscovery(discoveryCh, errCh, activeCallback, joinCallback, leaveCallback)
	return nil
}

func (h *hostDiscovery) monitorDiscovery(ch <-chan discovery.Entries, errCh <-chan error,
	activeCallback ActiveCallback, joinCallback JoinCallback, leaveCallback LeaveCallback) {
	for {
		select {
		case entries := <-ch:
			h.processCallback(entries, activeCallback, joinCallback, leaveCallback)
		case err := <-errCh:
			if err != nil {
				logrus.Errorf("discovery error: %v", err)
			}
		case <-h.stopChan:
			return
		}
	}
}

func (h *hostDiscovery) StopDiscovery() error {
	h.Lock()
	stopChan := h.stopChan
	h.watcher = nil
	h.Unlock()

	close(stopChan)
	return nil
}

func (h *hostDiscovery) processCallback(entries discovery.Entries,
	activeCallback ActiveCallback, joinCallback JoinCallback, leaveCallback LeaveCallback) {
	updated := hosts(entries)
	h.Lock()
	existing := h.nodes
	added, removed := diff(existing, updated)
	h.nodes = updated
	h.Unlock()

	activeCallback()
	if len(added) > 0 {
		joinCallback(added)
	}
	if len(removed) > 0 {
		leaveCallback(removed)
	}
}

func diff(existing mapset.Set, updated mapset.Set) (added []net.IP, removed []net.IP) {
	addSlice := updated.Difference(existing).ToSlice()
	removeSlice := existing.Difference(updated).ToSlice()
	for _, ip := range addSlice {
		added = append(added, net.ParseIP(ip.(string)))
	}
	for _, ip := range removeSlice {
		removed = append(removed, net.ParseIP(ip.(string)))
	}
	return
}

func (h *hostDiscovery) Fetch() []net.IP {
	h.Lock()
	defer h.Unlock()
	ips := []net.IP{}
	for _, ipstr := range h.nodes.ToSlice() {
		ips = append(ips, net.ParseIP(ipstr.(string)))
	}
	return ips
}

func hosts(entries discovery.Entries) mapset.Set {
	hosts := mapset.NewSet()
	for _, entry := range entries {
		hosts.Add(entry.Host)
	}
	return hosts
}
