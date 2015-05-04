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

package master

import (
	"net"
	"reflect"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"

	"github.com/golang/glog"
)

func (m *Master) serviceWriterLoop(stop chan struct{}) {
	t := time.NewTicker(10 * time.Second)
	defer t.Stop()
	for {
		// Update service & endpoint records.
		// TODO: when it becomes possible to change this stuff,
		// stop polling and start watching.
		// TODO: add endpoints of all replicas, not just the elected master.
		if err := m.createMasterNamespaceIfNeeded(api.NamespaceDefault); err != nil {
			glog.Errorf("Can't create master namespace: %v", err)
		}
		if m.serviceReadWriteIP != nil {
			if err := m.createMasterServiceIfNeeded("kubernetes", m.serviceReadWriteIP, m.serviceReadWritePort); err != nil {
				glog.Errorf("Can't create rw service: %v", err)
			}
			if err := m.setEndpoints("kubernetes", m.clusterIP, m.publicReadWritePort); err != nil {
				glog.Errorf("Can't create rw endpoints: %v", err)
			}
		}

		select {
		case <-stop:
			return
		case <-t.C:
		}
	}
}

func (m *Master) roServiceWriterLoop(stop chan struct{}) {
	t := time.NewTicker(10 * time.Second)
	defer t.Stop()
	for {
		// Update service & endpoint records.
		// TODO: when it becomes possible to change this stuff,
		// stop polling and start watching.
		if err := m.createMasterNamespaceIfNeeded(api.NamespaceDefault); err != nil {
			glog.Errorf("Can't create master namespace: %v", err)
		}
		if m.serviceReadOnlyIP != nil {
			if err := m.createMasterServiceIfNeeded("kubernetes-ro", m.serviceReadOnlyIP, m.serviceReadOnlyPort); err != nil {
				glog.Errorf("Can't create ro service: %v", err)
			}
			if err := m.setEndpoints("kubernetes-ro", m.clusterIP, m.publicReadOnlyPort); err != nil {
				glog.Errorf("Can't create ro endpoints: %v", err)
			}
		}

		select {
		case <-stop:
			return
		case <-t.C:
		}
	}
}

// createMasterNamespaceIfNeeded will create the namespace that contains the master services if it doesn't already exist
func (m *Master) createMasterNamespaceIfNeeded(ns string) error {
	ctx := api.NewContext()
	if _, err := m.namespaceRegistry.GetNamespace(ctx, api.NamespaceDefault); err == nil {
		// the namespace already exists
		return nil
	}
	namespace := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      ns,
			Namespace: "",
		},
	}
	_, err := m.storage["namespaces"].(rest.Creater).Create(ctx, namespace)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// createMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (m *Master) createMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePort int) error {
	ctx := api.NewDefaultContext()
	if _, err := m.serviceRegistry.GetService(ctx, serviceName); err == nil {
		// The service already exists.
		return nil
	}
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceName,
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{Port: servicePort, Protocol: api.ProtocolTCP}},
			// maintained by this code, not by the pod selector
			Selector:        nil,
			PortalIP:        serviceIP.String(),
			SessionAffinity: api.AffinityTypeNone,
		},
	}
	_, err := m.storage["services"].(rest.Creater).Create(ctx, svc)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// setEndpoints sets the endpoints for the given service.
// TODO: in a multi-master scenario this needs to consider all masters.
func (m *Master) setEndpoints(serviceName string, ip net.IP, port int) error {
	// The setting we want to find.
	want := []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: ip.String()}},
		Ports:     []api.EndpointPort{{Port: port, Protocol: api.ProtocolTCP}},
	}}

	ctx := api.NewDefaultContext()
	e, err := m.endpointRegistry.GetEndpoints(ctx, serviceName)
	if err != nil {
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
	}
	if !reflect.DeepEqual(e.Subsets, want) {
		e.Subsets = want
		glog.Infof("setting endpoints for master service %q to %v", serviceName, e)
		return m.endpointRegistry.UpdateEndpoints(ctx, e)
	}
	// We didn't make any changes, no need to actually call update.
	return nil
}
