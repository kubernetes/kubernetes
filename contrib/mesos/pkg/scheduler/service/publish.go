/*
Copyright 2015 The Kubernetes Authors.

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
	"net"
	"reflect"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/master/ports"

	"github.com/golang/glog"
)

const (
	SCHEDULER_SERVICE_NAME = "k8sm-scheduler"
)

func (m *SchedulerServer) newServiceWriter(publishedAddress net.IP, stop <-chan struct{}) func() {
	return func() {
		for {
			// Update service & endpoint records.
			// TODO(k8s): when it becomes possible to change this stuff,
			// stop polling and start watching.
			if err := m.createSchedulerServiceIfNeeded(SCHEDULER_SERVICE_NAME, ports.SchedulerPort); err != nil {
				glog.Errorf("Can't create scheduler service: %v", err)
			}

			if publishedAddress == nil {
				publishedAddress = net.IP(m.address)
			}
			if err := m.setEndpoints(SCHEDULER_SERVICE_NAME, publishedAddress, m.port); err != nil {
				glog.Errorf("Can't create scheduler endpoints: %v", err)
			}

			select {
			case <-stop:
				return
			case <-time.After(10 * time.Second):
			}
		}
	}
}

// createSchedulerServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (m *SchedulerServer) createSchedulerServiceIfNeeded(serviceName string, servicePort int) error {
	ctx := api.NewDefaultContext()
	if _, err := m.client.Core().Services(api.NamespaceValue(ctx)).Get(serviceName); err == nil {
		// The service already exists.
		return nil
	}
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceName,
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"provider": "k8sm", "component": "scheduler"},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{Port: int32(servicePort), Protocol: api.ProtocolTCP}},
			// maintained by this code, not by the pod selector
			Selector:        nil,
			SessionAffinity: api.ServiceAffinityNone,
		},
	}
	if m.serviceAddress != nil {
		svc.Spec.ClusterIP = m.serviceAddress.String()
	}
	_, err := m.client.Core().Services(api.NamespaceValue(ctx)).Create(svc)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// setEndpoints sets the endpoints for the given service.
// in a multi-master scenario only the master will be publishing an endpoint.
// see SchedulerServer.bootstrap.
func (m *SchedulerServer) setEndpoints(serviceName string, ip net.IP, port int) error {
	// The setting we want to find.
	want := []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: ip.String()}},
		Ports:     []api.EndpointPort{{Port: int32(port), Protocol: api.ProtocolTCP}},
	}}

	ctx := api.NewDefaultContext()
	e, err := m.client.Endpoints(api.NamespaceValue(ctx)).Get(serviceName)
	createOrUpdate := m.client.Endpoints(api.NamespaceValue(ctx)).Update
	if err != nil {
		if errors.IsNotFound(err) {
			createOrUpdate = m.client.Endpoints(api.NamespaceValue(ctx)).Create
		}
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
	}
	if !reflect.DeepEqual(e.Subsets, want) {
		e.Subsets = want
		glog.Infof("Setting endpoints for master service %q to %#v", serviceName, e)
		_, err = createOrUpdate(e)
		return err
	}
	// We didn't make any changes, no need to actually call update.
	return nil
}
