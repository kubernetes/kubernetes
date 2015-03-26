/*
Copyright 2014 Google Inc. All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"

	"github.com/golang/glog"
)

func (m *Master) serviceWriterLoop(stop chan struct{}) {
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
			if err := m.ensureEndpointsContain("kubernetes", m.clusterIP, m.publicReadWritePort); err != nil {
				glog.Errorf("Can't create rw endpoints: %v", err)
			}
		}

		select {
		case <-stop:
			return
		case <-time.After(10 * time.Second):
		}
	}
}

func (m *Master) roServiceWriterLoop(stop chan struct{}) {
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
			if err := m.ensureEndpointsContain("kubernetes-ro", m.clusterIP, m.publicReadOnlyPort); err != nil {
				glog.Errorf("Can't create ro endpoints: %v", err)
			}
		}

		select {
		case <-stop:
			return
		case <-time.After(10 * time.Second):
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
			Port: servicePort,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			PortalIP:        serviceIP.String(),
			Protocol:        api.ProtocolTCP,
			SessionAffinity: api.AffinityTypeNone,
		},
	}
	_, err := m.storage["services"].(rest.Creater).Create(ctx, svc)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// ensureEndpointsContain sets the endpoints for the given service. Also removes
// excess endpoints (as determined by m.masterCount). Extra endpoints could appear
// in the list if, for example, the master starts running on a different machine,
// changing IP addresses.
func (m *Master) ensureEndpointsContain(serviceName string, ip net.IP, port int) error {
	ctx := api.NewDefaultContext()
	e, err := m.endpointRegistry.GetEndpoints(ctx, serviceName)
	if err != nil || e.Protocol != api.ProtocolTCP {
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
			Protocol: api.ProtocolTCP,
		}
	}
	found := false
	for i := range e.Endpoints {
		ep := &e.Endpoints[i]
		if ep.IP == ip.String() && ep.Port == port {
			found = true
			break
		}
	}
	if !found {
		e.Endpoints = append(e.Endpoints, api.Endpoint{IP: ip.String(), Port: port})
		if len(e.Endpoints) > m.masterCount {
			// We append to the end and remove from the beginning, so this should
			// converge rapidly with all masters performing this operation.
			e.Endpoints = e.Endpoints[len(e.Endpoints)-m.masterCount:]
		}
		return m.endpointRegistry.UpdateEndpoints(ctx, e)
	}
	// We didn't make any changes, no need to actually call update.
	return nil
}
