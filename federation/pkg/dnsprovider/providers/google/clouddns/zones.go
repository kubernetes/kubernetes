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

package clouddns

import (
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adherence
var _ dnsprovider.Zones = Zones{}

type Zones struct {
	impl       interfaces.ManagedZonesService
	interface_ *Interface
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	response, err := zones.impl.List(zones.project()).Do()
	if err != nil {
		return nil, err
	}
	managedZones := response.ManagedZones()
	zoneList := make([]dnsprovider.Zone, len(managedZones))
	for i, zone := range managedZones {
		zoneList[i] = &Zone{zone, &zones}
	}
	return zoneList, nil
}

func (zones Zones) Add(zone dnsprovider.Zone) (dnsprovider.Zone, error) {
	managedZone := zones.impl.NewManagedZone(zone.Name())
	response, err := zones.impl.Create(zones.project(), managedZone).Do()
	if err != nil {
		return nil, err
	}
	return &Zone{response, &zones}, nil
}

func (zones Zones) Remove(zone dnsprovider.Zone) error {
	if err := zones.impl.Delete(zones.project(), zone.(*Zone).impl.Name()).Do(); err != nil {
		return err
	}
	return nil
}

func (zones Zones) New(name string) (dnsprovider.Zone, error) {
	managedZone := zones.impl.NewManagedZone(name)
	return &Zone{managedZone, &zones}, nil
}

func (zones Zones) project() string {
	return zones.interface_.project()
}
