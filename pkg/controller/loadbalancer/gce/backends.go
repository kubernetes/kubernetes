/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package lb

import (
	"fmt"
	"strconv"

	compute "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

type Backends struct {
	*ClusterManager
	pool *poolStore
}

func portKey(port int64) string {
	return fmt.Sprintf("%v", port)
}

func bgName(port int64) string {
	return fmt.Sprintf("%v-%v", backendPrefix, port)
}

func NewBackendPool(c *ClusterManager) *Backends {
	return &Backends{c, newPoolStore()}
}

// GetBackend returns a single backend
func (b *Backends) Get(port int64) (*compute.BackendService, error) {
	be, err := b.cloud.GetBackend(bgName(port))
	if err != nil {
		return nil, err
	}
	b.pool.Add(portKey(port), be)
	return be, nil
}

func (b *Backends) create(ig *compute.InstanceGroup, namedPort *compute.NamedPort, name string) (*compute.BackendService, error) {
	// Get the default health check
	hc, err := b.cloud.GetHttpHealthCheck(defaultHttpHealthCheck)
	if err != nil {
		return nil, err
	}
	// Create a new backend
	backend := &compute.BackendService{
		Name:     name,
		Protocol: "HTTP",
		Backends: []*compute.Backend{
			&compute.Backend{
				Group: ig.SelfLink,
			},
		},
		// Api expects one, means little to kubernetes.
		HealthChecks: []string{hc.SelfLink},
		Port:         namedPort.Port,
		PortName:     namedPort.Name,
	}
	if err := b.cloud.CreateBackend(backend); err != nil {
		return nil, err
	}
	return b.cloud.GetBackend(name)
}

// Backend will return a backend for the given port.
// If a backend already exists, it performs an edgehop.
// If one doesn't already exist, it will create it.
// If the port isn't one of the named ports in the instance group,
// it will add it. It returns a backend ready for insertion into a
// urlmap.
func (b *Backends) Add(port int64) error {
	namedPort, err := b.cloud.AddPortToInstanceGroup(
		b.defaultIg, port)
	if err != nil {
		return err
	}
	bg, _ := b.Get(port)
	if bg == nil {
		glog.Infof("Creating backend for instance group %v and port %v",
			b.defaultIg.Name, port)
		_, err = b.create(b.defaultIg, namedPort, bgName(port))
		if err != nil {
			return err
		}
	} else {
		glog.Infof("Backend %v already exists", bg.Name)
		if err := b.edgeHop(bg); err != nil {
			return err
		}
	}
	_, err = b.Get(port)
	return err
}

func (b *Backends) Delete(port int64) error {
	name := bgName(port)
	glog.Infof("Deleting backend %v", name)
	if err := b.cloud.DeleteBackend(name); err != nil {
		return err
	}
	b.pool.Delete(portKey(port))
	return nil
}

// edgeHop checks the links of the given backend by executing an edge hop.
// It fixes broken links.
func (b *Backends) edgeHop(be *compute.BackendService) error {
	if len(be.Backends) == 1 &&
		compareLinks(be.Backends[0].Group, b.defaultIg.SelfLink) {
		return nil
	}
	glog.Infof("Backend %v has a broken edge, adding link to %v",
		be.Name, b.defaultIg.Name)
	be.Backends = []*compute.Backend{
		&compute.Backend{Group: b.defaultIg.SelfLink},
	}
	if err := b.cloud.UpdateBackend(be); err != nil {
		return err
	}
	return nil
}

// Sync Backends for the given nodeports.
func (b *Backends) Sync(svcNodePorts []int64) error {
	glog.Infof("Syncing backends %v", svcNodePorts)

	knownPorts := util.NewStringSet()
	for _, port := range svcNodePorts {
		knownPorts.Insert(portKey(port))
	}
	pool := b.pool.snapshot()

	// gc unknown ports
	for port, _ := range pool {
		p, err := strconv.Atoi(port)
		if err != nil {
			return err
		}
		if knownPorts.Has(portKey(int64(p))) {
			continue
		}
		if err := b.Delete(int64(p)); err != nil {
			return err
		}
	}

	// create backends for new ports, perform an edge hop for existing ports
	for _, port := range svcNodePorts {
		if err := b.Add(port); err != nil {
			return err
		}
	}

	// The default backend isn't a part of the backed pool because it doesn't
	// consume a port on the instance group.
	return b.edgeHop(b.defaultBackend)
}
