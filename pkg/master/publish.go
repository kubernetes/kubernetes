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
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"

	"github.com/golang/glog"
)

func (m *Master) serviceWriterLoop(stop chan struct{}) {
	for {
		// Update service & endpoint records.
		// TODO: when it becomes possible to change this stuff,
		// stop polling and start watching.
		// TODO: add endpoints of all replicas, not just the elected master.
		if m.readWriteServer != "" {
			if err := m.createMasterServiceIfNeeded("kubernetes", 443); err != nil {
				glog.Errorf("Can't create rw service: %v", err)
			}
			if err := m.ensureEndpointsContain("kubernetes", m.readWriteServer); err != nil {
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
		if m.readOnlyServer != "" {
			if err := m.createMasterServiceIfNeeded("kubernetes-ro", 80); err != nil {
				glog.Errorf("Can't create ro service: %v", err)
			}
			if err := m.ensureEndpointsContain("kubernetes-ro", m.readOnlyServer); err != nil {
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

// createMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (m *Master) createMasterServiceIfNeeded(serviceName string, port int) error {
	ctx := api.NewDefaultContext()
	if _, err := m.serviceRegistry.GetService(ctx, serviceName); err == nil {
		// The service already exists.
		return nil
	}
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceName,
			Namespace: "default",
		},
		Spec: api.ServiceSpec{
			Port: port,
			// We're going to add the endpoints by hand, so this selector is mainly to
			// prevent identification of other pods. This selector will be useful when
			// we start hosting apiserver in a pod.
			Selector: map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
	}
	// Kids, don't do this at home: this is a hack. There's no good way to call the business
	// logic which lives in the REST object from here.
	c, err := m.storage["services"].Create(ctx, svc)
	if err != nil {
		return err
	}
	resp := <-c
	if _, ok := resp.Object.(*api.Service); ok {
		// If all worked, we get back an *api.Service object.
		return nil
	}
	return fmt.Errorf("Unexpected response: %#v", resp)
}

// ensureEndpointsContain sets the endpoints for the given service. Also removes
// excess endpoints (as determined by m.masterCount). Extra endpoints could appear
// in the list if, for example, the master starts running on a different machine,
// changing IP addresses.
func (m *Master) ensureEndpointsContain(serviceName string, endpoint string) error {
	ctx := api.NewDefaultContext()
	e, err := m.endpointRegistry.GetEndpoints(ctx, serviceName)
	if err != nil {
		e = &api.Endpoints{}
		// Fill in ID if it didn't exist already
		e.ObjectMeta.Name = serviceName
		e.ObjectMeta.Namespace = "default"
	}
	found := false
	for i := range e.Endpoints {
		if e.Endpoints[i] == endpoint {
			found = true
			break
		}
	}
	if !found {
		e.Endpoints = append(e.Endpoints, endpoint)
	}
	if len(e.Endpoints) > m.masterCount {
		// We append to the end and remove from the beginning, so this should
		// converge rapidly with all masters performing this operation.
		e.Endpoints = e.Endpoints[len(e.Endpoints)-m.masterCount:]
	} else if found {
		// We didn't make any changes, no need to actually call update.
		return nil
	}
	return m.endpointRegistry.UpdateEndpoints(ctx, e)
}
